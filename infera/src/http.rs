// src/http.rs
// Handles downloading and caching of remote models.

use crate::error::InferaError;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;

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

    let mut response = reqwest::blocking::get(url)
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?
        .error_for_status()
        .map_err(|e| InferaError::HttpRequestError(e.to_string()))?;
    let mut temp_file =
        File::create(&cached_path).map_err(|e| InferaError::IoError(e.to_string()))?;
    io::copy(&mut response, &mut temp_file).map_err(|e| InferaError::IoError(e.to_string()))?;
    Ok(cached_path)
}
