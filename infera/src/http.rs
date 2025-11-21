use crate::config::{LogLevel, CONFIG};
use crate::error::InferaError;
use crate::log;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime};

use reqwest::header::{ETAG, IF_NONE_MATCH};

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
    println!("evict_cache_if_needed");
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

        println!("consider to remove file: {:?}", path);
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

// / Handles the download and caching of a remote model from a URL.
// /
// / If the model for the given URL is already present in the local cache, this
// / function updates its access time and returns the path. Otherwise, it downloads
// / the file, evicts old cache entries if needed, stores it in the cache directory,
// / and then returns the path.
// /
// / The cache uses an LRU (Least Recently Used) eviction policy with a configurable
// / size limit (default 1GB, configurable via INFERA_CACHE_SIZE_LIMIT env var).
// /
// / Downloads support automatic retries with exponential backoff.
// /
// / # Arguments
// /
// / * `url` - The HTTP/HTTPS URL of the ONNX model to be downloaded.
// /
// / # Returns
// /
// / A `Result` which is:
// / * `Ok(PathBuf)`: The local file path of the cached model.
// / * `Err(InferaError)`: An error indicating failure in creating the cache directory,
// /   making the HTTP request, or writing the file to disk.

pub(crate) fn handle_remote_model(url: &str) -> Result<PathBuf, InferaError> {
    let max_attempts = CONFIG.http_retry_attempts;
    let retry_delay_ms = CONFIG.http_retry_delay_ms;
    let timeout_secs = CONFIG.http_timeout_secs;

    let cache_dir = cache_dir();
    if !cache_dir.exists() {
        log!(LogLevel::Info, "Creating cache directory: {:?}", cache_dir);
        fs::create_dir_all(&cache_dir).map_err(|e| InferaError::CacheDirError(e.to_string()))?;
    }

    // Compute cache key based on URL hash
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let hash_hex = hex::encode(hasher.finalize());
    let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));
    let etag_path = cache_dir.join(format!("{}.etag", hash_hex));

    // Load cached ETag if available
    let etag_trimmed = match fs::read_to_string(&etag_path) {
        Ok(etag_value) => etag_value.trim().to_string(),
        Err(_) => "abc".to_string(),
    };
    println!("Using ETag for request: {}", etag_trimmed);

    let temp_path = cached_path.with_extension("onnx.part");
    let mut guard = TempFileGuard::new(&temp_path);

    let mut last_error = None;

    for attempt in 1..=max_attempts {
        log!(
            LogLevel::Debug,
            "Download attempt {}/{} for {}",
            attempt,
            max_attempts,
            url
        );

        // Perform download using helper function
        match download_file_with_etag(url, &temp_path, timeout_secs, &etag_trimmed) {
            Ok(Some((false, etag_str))) => {
                //  New file was downloaded (first value is false)
                log!(LogLevel::Info, "Download succeeded for URL: {}", url);

                let file_size = fs::metadata(&temp_path)
                    .map_err(|e| InferaError::IoError(e.to_string()))?
                    .len();
                evict_cache_if_needed(file_size)?;
                fs::rename(&temp_path, &cached_path)
                    .map_err(|e| InferaError::IoError(e.to_string()))?;

                // Update ETag file
                fs::write(&etag_path, &etag_str)
                    .map_err(|e| InferaError::IoError(e.to_string()))?;

                guard.commit();
                return Ok(cached_path);
            }
            Ok(Some((true, _))) => {
                // status is 304 Not Modified, use cached file (first value is true)
                log!(LogLevel::Info, "Cache hit for URL: {}", url);

                // Update access time for LRU tracking
                touch_cache_file(&cached_path)?;

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
                if attempt < max_attempts {
                    let delay = Duration::from_millis(retry_delay_ms * attempt as u64);
                    log!(LogLevel::Debug, "Waiting {:?} before retry", delay);
                    thread::sleep(delay);
                }
            }
            Ok(None) => {
                // theoretically unreachable, but necessary to satisfy exhaustiveness
                // Handle as error or log warning, or panic
                log!(LogLevel::Error, "Can't exist None for this matching");
            }
        }
    }

    log!(
        LogLevel::Error,
        "Failed to download after {} attempts: {}",
        max_attempts,
        url
    );

    Err(last_error
        .unwrap_or_else(|| InferaError::HttpRequestError("Unknown download error".to_string())))
}

fn download_file_with_etag(
    url: &str,
    dest: &Path,
    timeout_secs: u64,
    etag: &str,
) -> Result<Option<(bool, String)>, InferaError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;

    let request = client.get(url).header(IF_NONE_MATCH, etag);

    let mut response = request
        .send()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?
        .error_for_status()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;

    println!("response.status(): {:?}", response.status());
    if etag != "" && response.status() == reqwest::StatusCode::NOT_MODIFIED {
        // Not modified, no file write needed, return None etag
        return Ok(Some((true, etag.to_string())));
    }
    println!("prepare to create dest file: {:?}", dest);
    let mut file = File::create(dest).map_err(|e| InferaError::IoError(e.to_string()))?;
    io::copy(&mut response, &mut file).map_err(|e| InferaError::IoError(e.to_string()))?;

    // Extract ETag header if present for updating cache metadata
    let etag_header = response
        .headers()
        .get(ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned());
    let etag_str = etag_header.unwrap_or_else(|| "".to_string());

    return Ok(Some((false, etag_str)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_middleware_etag::Etag;
    use actix_web::{web, App, Error, HttpResponse, HttpServer};
    use bytes::Bytes;
    use mockito::Server;
    use std::env; // moved here: used in tests only
    use std::sync::Arc;
    use std::thread;
    use tiny_http::{Header, Response, Server as TinyServer};
    use tokio::sync::RwLock;

    fn get_file_modification_time(path: &std::path::Path) -> std::io::Result<SystemTime> {
        let metadata = fs::metadata(path)?;
        let modified_time = metadata.modified()?;
        Ok(modified_time)
    }
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
        println!("Cache dir: {:?}", cache_dir);
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

    #[actix_web::test]
    async fn test_handle_remote_model_success_and_cache() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind random port");
        let addr = listener.local_addr().unwrap();

        let file_content = Arc::new(RwLock::new(b"initial content".to_vec()));
        let file_content_server = file_content.clone();

        let server = HttpServer::new(move || {
            let content = file_content_server.clone();
            App::new().wrap(Etag::default()).route(
                "/model.onnx",
                web::get().to(move || {
                    let content = content.clone();
                    async move {
                        let data = content.read().await;
                        let bytes = Bytes::copy_from_slice(&*data);
                        Ok::<_, Error>(
                            HttpResponse::Ok()
                                .content_type("application/octet-stream")
                                .body(bytes),
                        )
                    }
                }),
            )
        })
        .listen(listener)
        .expect("Failed to bind server")
        .run();

        // Spawn server in background
        let srv_handle = actix_web::rt::spawn(server);

        let url = format!("http://{}:{}/model.onnx", addr.ip(), addr.port());
        let second_call_url = url.clone();
        let third_call_url = url.clone();
        // Call your blocking cache-and-download function in blocking task
        let path1 = tokio::task::spawn_blocking(move || handle_remote_model(&url))
            .await
            .expect("Task panicked")
            .expect("handle_remote_model failed");

        assert!(path1.exists());
        let content1 = fs::read(&path1).expect("read cached file");
        let path1_modification_time = get_file_modification_time(&path1).unwrap();

        // Call again, should refresh cache
        let path2 = tokio::task::spawn_blocking(move || handle_remote_model(&second_call_url))
            .await
            .expect("Task panicked")
            .expect("handle_remote_model failed");

        tokio::time::sleep(Duration::from_secs(1)).await;

        let content2 = fs::read(&path2).expect("read cached file");
        let path2_modification_time = get_file_modification_time(&path2).unwrap();

        assert_eq!(path1, path2);
        assert_eq!(content1, content2);
        assert_eq!(path1_modification_time, path2_modification_time);

        // Modify content to simulate update
        {
            let mut content_write = file_content.write().await;
            *content_write = b"updated content".to_vec();
        }

        tokio::time::sleep(Duration::from_secs(1)).await;

        // Call again, should refresh cache
        let path3 = tokio::task::spawn_blocking(move || handle_remote_model(&third_call_url))
            .await
            .expect("Task panicked")
            .expect("handle_remote_model failed");

        let content3 = fs::read(&path3).expect("read cached file");
        let path3_modification_time = get_file_modification_time(&path3).unwrap();
        assert_eq!(path1, path3);
        assert_ne!(content1, content3);
        assert_ne!(path1_modification_time, path3_modification_time);

        // Cleanup server
        srv_handle.abort();
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
