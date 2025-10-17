// src/http.rs
// Handles downloading and caching of remote models.

use crate::config::{LogLevel, CONFIG};
use crate::error::InferaError;
use crate::log;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime};

/// A guard that guarantees a temporary file is deleted when it goes out of scope.
/// This is used to implement a panic-safe cleanup of partial downloads.
struct TempFileGuard<'a> {
    path: &'a Path,
    committed: bool,
}

impl<'a> TempFileGuard<'a> {
    /// Creates a new guard for the given path.
    fn new(path: &'a Path) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    /// Marks the file as "committed," preventing its deletion on drop.
    /// This should be called only after the file has been successfully and
    /// atomically moved to its final destination.
    fn commit(&mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for TempFileGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(self.path);
        }
    }
}

/// Return the cache directory path used by Infera for remote models.
pub(crate) fn cache_dir() -> PathBuf {
    CONFIG.cache_dir.clone()
}

/// Gets the cache size limit in bytes from environment variable or default.
fn get_cache_size_limit() -> u64 {
    CONFIG.cache_size_limit
}

/// Updates the access time of a cached file by touching it.
fn touch_cache_file(path: &Path) -> Result<(), InferaError> {
    if path.exists() {
        let now = filetime::FileTime::now();
        filetime::set_file_atime(path, now).map_err(|e| InferaError::IoError(e.to_string()))?;
    }
    Ok(())
}

/// Gets metadata about cached files sorted by access time (oldest first).
fn get_cached_files_by_access_time() -> Result<Vec<(PathBuf, SystemTime, u64)>, InferaError> {
    let dir = cache_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in fs::read_dir(&dir)
        .map_err(|e| InferaError::IoError(e.to_string()))?
        .flatten()
    {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("onnx") {
            if let Ok(metadata) = fs::metadata(&path) {
                let accessed = metadata.accessed().unwrap_or_else(|_| SystemTime::now());
                let size = metadata.len();
                files.push((path, accessed, size));
            }
        }
    }

    // Sort by access time, oldest first
    files.sort_by_key(|(_, time, _)| *time);
    Ok(files)
}

/// Calculates total cache size in bytes.
fn get_cache_size() -> Result<u64, InferaError> {
    let files = get_cached_files_by_access_time()?;
    Ok(files.iter().map(|(_, _, size)| size).sum())
}

/// Evicts least recently used cache files until cache size is below limit.
fn evict_cache_if_needed(required_space: u64) -> Result<(), InferaError> {
    let limit = get_cache_size_limit();
    let current_size = get_cache_size()?;

    if current_size + required_space <= limit {
        return Ok(());
    }

    let target_size = limit.saturating_sub(required_space);
    let mut freed_size = 0u64;
    let files = get_cached_files_by_access_time()?;

    for (path, _, size) in files {
        if current_size - freed_size <= target_size {
            break;
        }

        fs::remove_file(&path).map_err(|e| InferaError::IoError(e.to_string()))?;
        freed_size += size;
    }

    Ok(())
}

/// Clears the entire cache directory by deleting its contents.
/// If the directory does not exist, this is a no-op.
pub(crate) fn clear_cache() -> Result<(), InferaError> {
    let dir = cache_dir();
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(&dir)
        .map_err(|e| InferaError::IoError(e.to_string()))?
        .flatten()
    {
        let path = entry.path();
        if path.is_file() {
            fs::remove_file(&path).map_err(|e| InferaError::IoError(e.to_string()))?;
        } else if path.is_dir() {
            fs::remove_dir_all(&path).map_err(|e| InferaError::IoError(e.to_string()))?;
        }
    }
    Ok(())
}

/// Handles the download and caching of a remote model from a URL.
///
/// If the model for the given URL is already present in the local cache, this
/// function updates its access time and returns the path. Otherwise, it downloads
/// the file, evicts old cache entries if needed, stores it in the cache directory,
/// and then returns the path.
///
/// The cache uses an LRU (Least Recently Used) eviction policy with a configurable
/// size limit (default 1GB, configurable via INFERA_CACHE_SIZE_LIMIT env var).
///
/// Downloads support automatic retries with exponential backoff.
///
/// # Arguments
///
/// * `url` - The HTTP/HTTPS URL of the ONNX model to be downloaded.
///
/// # Returns
///
/// A `Result` which is:
/// * `Ok(PathBuf)`: The local file path of the cached model.
/// * `Err(InferaError)`: An error indicating failure in creating the cache directory,
///   making the HTTP request, or writing the file to disk.
pub(crate) fn handle_remote_model(url: &str) -> Result<PathBuf, InferaError> {
    let cache_dir = cache_dir();
    if !cache_dir.exists() {
        log!(LogLevel::Info, "Creating cache directory: {:?}", cache_dir);
        fs::create_dir_all(&cache_dir).map_err(|e| InferaError::CacheDirError(e.to_string()))?;
    }
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let hash_hex = hex::encode(hasher.finalize());
    let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));

    if cached_path.exists() {
        log!(LogLevel::Info, "Cache hit for URL: {}", url);
        // Update access time for LRU tracking
        touch_cache_file(&cached_path)?;
        return Ok(cached_path);
    }

    log!(
        LogLevel::Info,
        "Cache miss for URL: {}, downloading...",
        url
    );

    let temp_path = cached_path.with_extension("onnx.part");
    let mut guard = TempFileGuard::new(&temp_path);

    // Download with retry logic
    let max_attempts = CONFIG.http_retry_attempts;
    let retry_delay_ms = CONFIG.http_retry_delay_ms;
    let timeout_secs = CONFIG.http_timeout_secs;

    let mut last_error = None;

    for attempt in 1..=max_attempts {
        log!(
            LogLevel::Debug,
            "Download attempt {}/{} for {}",
            attempt,
            max_attempts,
            url
        );

        match download_file(url, &temp_path, timeout_secs) {
            Ok(_) => {
                log!(LogLevel::Info, "Successfully downloaded: {}", url);

                // Check file size and evict cache if needed
                let file_size = fs::metadata(&temp_path)
                    .map_err(|e| InferaError::IoError(e.to_string()))?
                    .len();

                log!(LogLevel::Debug, "Downloaded file size: {} bytes", file_size);
                evict_cache_if_needed(file_size)?;

                fs::rename(&temp_path, &cached_path)
                    .map_err(|e| InferaError::IoError(e.to_string()))?;

                guard.commit();
                return Ok(cached_path);
            }
            Err(e) => {
                log!(
                    LogLevel::Warn,
                    "Download attempt {}/{} failed: {}",
                    attempt,
                    max_attempts,
                    e
                );
                last_error = Some(e);

                // Don't sleep after the last attempt
                if attempt < max_attempts {
                    let delay = Duration::from_millis(retry_delay_ms * attempt as u64);
                    log!(LogLevel::Debug, "Waiting {:?} before retry", delay);
                    thread::sleep(delay);
                }
            }
        }
    }

    log!(
        LogLevel::Error,
        "Failed to download after {} attempts: {}",
        max_attempts,
        url
    );
    Err(last_error.unwrap_or_else(|| InferaError::HttpRequestError("Unknown error".to_string())))
}

/// Download a file from a URL to a local path with timeout
fn download_file(url: &str, dest: &Path, timeout_secs: u64) -> Result<(), InferaError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?
        .error_for_status()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;

    let mut file = File::create(dest).map_err(|e| InferaError::IoError(e.to_string()))?;
    io::copy(&mut response, &mut file).map_err(|e| InferaError::IoError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use std::env; // moved here: used in tests only
    use std::thread;
    use tiny_http::{Header, Response, Server as TinyServer};

    #[test]
    fn test_handle_remote_model_cleanup_on_incomplete_download() {
        let server = TinyServer::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let model_url = format!("http://127.0.0.1:{}/incomplete_model.onnx", port);

        let server_handle = thread::spawn(move || {
            if let Ok(request) = server.recv() {
                let mut response = Response::from_string("incomplete data");
                let header = Header::from_bytes(&b"Content-Length"[..], &b"100"[..]).unwrap();
                response.add_header(header);
                let _ = request.respond(response);
            }
        });

        // The download should fail because the response body is shorter than the content-length.
        let result = handle_remote_model(&model_url);
        assert!(result.is_err());

        // Ensure no partial or final file is left in the cache.
        let cache_dir = env::temp_dir().join("infera_cache");
        let mut hasher = Sha256::new();
        hasher.update(model_url.as_bytes());
        let hash_hex = hex::encode(hasher.finalize());
        let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));
        let temp_path = cached_path.with_extension("onnx.part");

        assert!(!cached_path.exists(), "Final cache file should not exist");
        assert!(!temp_path.exists(), "Partial file should be cleaned up");

        server_handle.join().unwrap();
    }

    #[test]
    fn test_handle_remote_model_download_error() {
        // Simulate a server error instead of an interrupted download,
        // as it's more reliable to test the error handling path.
        let mut server = Server::new();
        let _m = server
            .mock("GET", "/server_error_model.onnx")
            .with_status(500)
            .create();

        let url = server.url();
        let model_url = format!("{}/server_error_model.onnx", url);

        let result = handle_remote_model(&model_url);

        // The download should fail because of the server error.
        assert!(
            result.is_err(),
            "handle_remote_model should return an error on 500 status"
        );

        // Ensure no partial or final file is left in the cache.
        let cache_dir = env::temp_dir().join("infera_cache");
        let mut hasher = Sha256::new();
        hasher.update(model_url.as_bytes());
        let hash_hex = hex::encode(hasher.finalize());
        let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));
        let temp_path = cached_path.with_extension("onnx.part");

        assert!(!cached_path.exists(), "Final cache file should not exist");
        assert!(!temp_path.exists(), "Partial file should be cleaned up");
    }

    #[test]
    fn test_handle_remote_model_cleanup_on_connection_drop() {
        let server = TinyServer::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let model_url = format!("http://127.0.0.1:{}/dropped_connection.onnx", port);

        let server_handle = thread::spawn(move || {
            if let Ok(request) = server.recv() {
                // By responding with a Content-Length header but then dropping the
                // request without sending a body, we simulate a connection drop.
                // reqwest will receive the headers and expect a body, but the
                // connection will be closed prematurely, resulting in an I/O error.
                let response = Response::empty(200)
                    .with_header(Header::from_bytes(&b"Content-Length"[..], &b"1024"[..]).unwrap());
                // The `respond` call sends the headers, but the `Request` object
                // is dropped immediately after, closing the connection.
                let _ = request.respond(response);
            }
        });

        // The download should fail because the server closes the connection prematurely.
        let result = handle_remote_model(&model_url);
        assert!(
            result.is_err(),
            "The download should fail on a connection drop"
        );

        // After the failure, the temporary file should be cleaned up.
        let cache_dir = env::temp_dir().join("infera_cache");
        let mut hasher = Sha256::new();
        hasher.update(model_url.as_bytes());
        let hash_hex = hex::encode(hasher.finalize());
        let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));
        let temp_path = cached_path.with_extension("onnx.part");

        assert!(!cached_path.exists(), "Final cache file should not exist");
        assert!(
            !temp_path.exists(),
            "Partial file should be cleaned up after a connection drop"
        );

        server_handle.join().unwrap();
    }

    #[test]
    fn test_handle_remote_model_success_and_cache() {
        // Serve a small body with an accurate Content-Length
        let mut server = Server::new();
        let body = b"onnxdata".to_vec();
        let _m = server
            .mock("GET", "/ok_model.onnx")
            .with_status(200)
            .with_header("Content-Length", &body.len().to_string())
            .with_body(body.clone())
            .create();
        let url = format!("{}/ok_model.onnx", server.url());

        let path1 = handle_remote_model(&url).expect("download should succeed");
        assert!(path1.exists(), "cached file must exist");
        let content1 = fs::read(&path1).expect("read cached file");
        assert_eq!(content1, body);

        // Second call should hit cache and return same path without network
        let path2 = handle_remote_model(&url).expect("cache should hit");
        assert_eq!(path1, path2);
        let temp_path = path1.with_extension("onnx.part");
        assert!(!temp_path.exists(), "no partial file should remain");
    }

    #[test]
    fn test_clear_cache_removes_files() {
        let dir = cache_dir();
        let _ = fs::create_dir_all(&dir);
        let dummy = dir.join("dummy.tmp");
        fs::write(&dummy, b"x").unwrap();
        assert!(dummy.exists());
        clear_cache().unwrap();
        assert!(!dummy.exists());
    }
}
