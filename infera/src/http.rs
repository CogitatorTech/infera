// src/http.rs
// Handles downloading and caching of remote models.

use crate::error::InferaError;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

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
    fn commit(mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for TempFileGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            // If the file was not committed, it's a temporary partial file
            // that should be cleaned up. We ignore a potential error here,
            // as we can't do anything about it during a drop.
            let _ = fs::remove_file(self.path);
        }
    }
}

/// Handles the download and caching of a remote model from a URL.
///
/// If the model for the given URL is already present in the local cache, this
/// function returns the path to the cached file directly. Otherwise, it downloads
/// the file, stores it in the cache directory, and then returns the path.
///
/// The cache location is a subdirectory named `infera_cache` inside the system's
/// temporary directory. The filename for the cached model is the SHA256 hash of its URL.
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
    let cache_dir = env::temp_dir().join("infera_cache");
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).map_err(|e| InferaError::CacheDirError(e.to_string()))?;
    }
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let hash_hex = hex::encode(hasher.finalize());
    let cached_path = cache_dir.join(format!("{}.onnx", hash_hex));

    if cached_path.exists() {
        return Ok(cached_path);
    }

    let temp_path = cached_path.with_extension("onnx.part");
    // The guard ensures that the temp file is cleaned up if a panic occurs.
    let guard = TempFileGuard::new(&temp_path);

    let mut response = reqwest::blocking::get(url)
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?
        .error_for_status()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;

    let mut temp_file =
        File::create(&temp_path).map_err(|e| InferaError::IoError(e.to_string()))?;
    io::copy(&mut response, &mut temp_file).map_err(|e| InferaError::IoError(e.to_string()))?;

    fs::rename(&temp_path, &cached_path).map_err(|e| InferaError::IoError(e.to_string()))?;

    // The file has been successfully renamed, so we can disarm the guard.
    guard.commit();

    Ok(cached_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
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
}
