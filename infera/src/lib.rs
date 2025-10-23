// src/lib.rs
// The public C API layer and module declarations.

use serde_json::json;
use std::ffi::{c_char, CStr, CString};
use std::fs;

// Declare the internal modules
mod config;
mod engine;
mod error;
mod ffi_utils;
mod http;
mod model;

// Re-export the public FFI utility functions and types
pub use error::infera_last_error;
pub use ffi_utils::{infera_free, infera_free_result, InferaInferenceResult};

/// Loads an ONNX model from a local file path or a remote URL and assigns it a unique name.
///
/// If the `path` starts with "http://" or "https://", the model will be downloaded
/// and cached locally. Otherwise, it will be treated as a local file path.
///
/// # Arguments
///
/// * `name` - A pointer to a null-terminated C string representing the unique name for the model.
/// * `path` - A pointer to a null-terminated C string representing the file path or URL of the model.
///
/// # Returns
///
/// * `0` on success.
/// * `-1` on failure. Call `infera_last_error()` to get a descriptive error message.
///
/// # Safety
///
/// * The `name` and `path` pointers must not be null.
/// * The memory pointed to by `name` and `path` must be valid, null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn infera_load_model(name: *const c_char, path: *const c_char) -> i32 {
    let result = (|| -> Result<(), error::InferaError> {
        if name.is_null() || path.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let name_str = CStr::from_ptr(name).to_str()?;
        let path_or_url_str = CStr::from_ptr(path).to_str()?;

        let local_path = if path_or_url_str.starts_with("http") {
            http::handle_remote_model(path_or_url_str)?
        } else {
            path_or_url_str.into()
        };
        let local_path_str = local_path.to_str().ok_or(error::InferaError::Utf8Error)?;

        engine::load_model_impl(name_str, local_path_str)
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            error::set_last_error(&e);
            -1
        }
    }
}

/// Unloads a model, freeing its associated resources.
///
/// # Arguments
///
/// * `name` - A pointer to a null-terminated C string representing the name of the model to unload.
///
/// # Returns
///
/// * `0` on success.
/// * `-1` if the model was not found or an error occurred.
///
/// # Safety
///
/// * The `name` pointer must not be null.
/// * The memory pointed to by `name` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn infera_unload_model(name: *const c_char) -> i32 {
    let result = (|| -> Result<(), error::InferaError> {
        if name.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let name_str = CStr::from_ptr(name).to_str()?;
        if model::MODELS.write().remove(name_str).is_some() {
            Ok(())
        } else {
            Err(error::InferaError::ModelNotFound(name_str.to_string()))
        }
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            error::set_last_error(&e);
            -1
        }
    }
}

/// Runs inference on a loaded model with the given input data.
///
/// The input data is provided as a raw pointer to a flat array of `f32` values.
/// The result of the inference is returned in an `InferaInferenceResult` struct.
/// The caller is responsible for freeing the result using `infera_free_result`.
///
/// # Arguments
///
/// * `model_name` - A pointer to a null-terminated C string for the model's name.
/// * `data` - A pointer to the input tensor data, organized as a flat array of `f32`.
/// * `rows` - The number of rows in the input tensor.
/// * `cols` - The number of columns in the input tensor.
///
/// # Returns
///
/// An `InferaInferenceResult` struct containing the output tensor data and metadata.
/// If an error occurs, the `status` field of the struct will be `-1`.
///
/// # Safety
///
/// * `model_name` and `data` must not be null.
/// * `model_name` must point to a valid, null-terminated C string.
/// * `data` must point to a contiguous block of memory of size `rows * cols * size_of<f32>()`.
#[no_mangle]
pub unsafe extern "C" fn infera_predict(
    model_name: *const c_char,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> InferaInferenceResult {
    let result = (|| -> Result<InferaInferenceResult, error::InferaError> {
        if model_name.is_null() || data.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let name_str = CStr::from_ptr(model_name).to_str()?;
        engine::run_inference_impl(name_str, data, rows, cols)
    })();

    match result {
        Ok(res) => res,
        Err(e) => {
            error::set_last_error(&e);
            InferaInferenceResult::error()
        }
    }
}

/// Runs inference on a loaded model with input data from a raw byte `BLOB`.
///
/// This function is useful when the input tensor is stored as a `BLOB`. The byte data
/// is interpreted as a flat array of `f32` values (native-endian). The function
/// will attempt to infer the batch size based on the model's expected input shape.
///
/// # Arguments
///
/// * `model_name` - A pointer to a null-terminated C string for the model's name.
/// * `blob_data` - A pointer to the input data as a raw byte array.
/// * `blob_len` - The total length of the byte array in `blob_data`.
///
/// # Returns
///
/// An `InferaInferenceResult` struct containing the output. The caller is responsible
/// for freeing this result using `infera_free_result`.
///
/// # Safety
///
/// * `model_name` and `blob_data` must not be null.
/// * `model_name` must point to a valid, null-terminated C string.
/// * `blob_data` must point to a contiguous block of memory of size `blob_len`.
/// * `blob_len` must be a multiple of `std::mem::size_of::<f32>()`.
#[no_mangle]
pub unsafe extern "C" fn infera_predict_from_blob(
    model_name: *const c_char,
    blob_data: *const u8,
    blob_len: usize,
) -> InferaInferenceResult {
    let result = (|| -> Result<InferaInferenceResult, error::InferaError> {
        if model_name.is_null() || blob_data.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let name_str = CStr::from_ptr(model_name).to_str()?;
        engine::run_inference_blob_impl(name_str, blob_data, blob_len)
    })();

    match result {
        Ok(res) => res,
        Err(e) => {
            error::set_last_error(&e);
            InferaInferenceResult::error()
        }
    }
}

/// Retrieves metadata about a specific loaded model as a JSON string.
///
/// The returned JSON string includes the model's name, and its input and output shapes.
///
/// # Arguments
///
/// * `model_name` - A pointer to a null-terminated C string for the model's name.
///
/// # Returns
///
/// A pointer to a heap-allocated, null-terminated C string containing JSON.
/// The caller is responsible for freeing this string using `infera_free`.
/// On error (e.g., model not found), the JSON will contain an "error" key.
///
/// # Safety
///
/// * The `model_name` pointer must not be null and must point to a valid C string.
/// * The returned pointer must be freed with `infera_free` to avoid memory leaks.
#[no_mangle]
pub unsafe extern "C" fn infera_get_model_info(model_name: *const c_char) -> *mut c_char {
    let result = (|| -> Result<String, error::InferaError> {
        if model_name.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let name_str = CStr::from_ptr(model_name).to_str()?;
        engine::get_model_metadata_impl(name_str)
    })();

    match result {
        Ok(json) => CString::new(json).unwrap_or_default().into_raw(),
        Err(e) => {
            error::set_last_error(&e);
            let error_json = json!({ "error": e.to_string() }).to_string();
            CString::new(error_json).unwrap_or_default().into_raw()
        }
    }
}

/// Returns a JSON array of the names of all currently loaded models.
///
/// # Returns
///
/// A pointer to a heap-allocated, null-terminated C string containing a JSON array of strings.
/// The caller is responsible for freeing this string using `infera_free`.
///
/// # Safety
///
/// The returned pointer must be freed with `infera_free` to avoid memory leaks.
#[no_mangle]
pub extern "C" fn infera_get_loaded_models() -> *mut c_char {
    let models = model::MODELS.read();
    let list: Vec<String> = models.keys().cloned().collect();
    let joined = serde_json::to_string(&list).unwrap_or_else(|_| "[]".to_string());
    match CString::new(joined) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => {
            // Fallback to empty JSON array if string contains null bytes
            match CString::new("[]") {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => std::ptr::null_mut(), // This should never happen with "[]"
            }
        }
    }
}

/// Returns a JSON string with version and build information about the Infera library.
///
/// The JSON object includes the library version, the enabled ONNX backend (e.g., "tract"),
/// and the directory used for caching remote models.
///
/// # Returns
///
/// A pointer to a heap-allocated, null-terminated C string containing the version info.
/// The caller is responsible for freeing this string using `infera_free`.
///
/// # Safety
///
/// The returned pointer must be freed with `infera_free` to avoid memory leaks.
#[no_mangle]
pub extern "C" fn infera_get_version() -> *mut c_char {
    let cache_dir_str = config::CONFIG.cache_dir.to_string_lossy().to_string();
    let info = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "onnx_backend": if cfg!(feature = "tract") { "tract" } else { "disabled" },
        "model_cache_dir": cache_dir_str,
    });
    let json_str = serde_json::to_string(&info).unwrap_or_default();
    CString::new(json_str).unwrap_or_default().into_raw()
}

/// Clears the entire model cache directory.
///
/// This removes all cached remote models, freeing up disk space.
///
/// # Returns
///
/// * `0` on success.
/// * `-1` on failure. Call `infera_last_error()` to get a descriptive error message.
///
/// # Safety
///
/// This function is safe to call at any time.
#[no_mangle]
pub extern "C" fn infera_clear_cache() -> i32 {
    match http::clear_cache() {
        Ok(()) => 0,
        Err(e) => {
            error::set_last_error(&e);
            -1
        }
    }
}

/// Returns cache statistics as a JSON string.
///
/// The JSON object includes:
/// * `"cache_dir"`: The path to the cache directory.
/// * `"total_size_bytes"`: Total size of cached models in bytes.
/// * `"file_count"`: Number of cached model files.
/// * `"size_limit_bytes"`: The configured cache size limit.
///
/// # Returns
///
/// A pointer to a heap-allocated, null-terminated C string containing the cache info.
/// The caller is responsible for freeing this string using `infera_free`.
///
/// # Safety
///
/// The returned pointer must be freed with `infera_free` to avoid memory leaks.
#[no_mangle]
pub extern "C" fn infera_get_cache_info() -> *mut c_char {
    let result = (|| -> Result<serde_json::Value, error::InferaError> {
        let cache_dir = http::cache_dir();
        let cache_dir_str = cache_dir.to_string_lossy().to_string();

        let mut total_size = 0u64;
        let mut file_count = 0usize;

        if cache_dir.exists() {
            for entry in fs::read_dir(&cache_dir)
                .map_err(|e| error::InferaError::IoError(e.to_string()))?
                .flatten()
            {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    if let Ok(metadata) = fs::metadata(&path) {
                        total_size += metadata.len();
                        file_count += 1;
                    }
                }
            }
        }

        let size_limit = config::CONFIG.cache_size_limit;

        Ok(json!({
            "cache_dir": cache_dir_str,
            "total_size_bytes": total_size,
            "file_count": file_count,
            "size_limit_bytes": size_limit,
        }))
    })();

    let final_json = result.unwrap_or_else(|e| {
        error::set_last_error(&e);
        json!({"error": e.to_string()})
    });
    let json_str = serde_json::to_string(&final_json).unwrap_or_default();
    CString::new(json_str).unwrap_or_default().into_raw()
}

/// Scans a directory for `.onnx` files and loads them into Infera automatically.
///
/// The name for each model is derived from its filename (without the extension).
///
/// # Arguments
///
/// * `path` - A pointer to a null-terminated C string representing the directory path.
///
/// # Returns
///
/// A pointer to a heap-allocated C string containing a JSON object with two fields:
/// * `"loaded"`: A list of model names that were successfully loaded.
/// * `"errors"`: A list of objects, each detailing a file that failed to load and the reason.
///
/// The caller is responsible for freeing this string using `infera_free`.
///
/// # Safety
///
/// * The `path` pointer must not be null and must point to a valid C string.
/// * The returned pointer must be freed with `infera_free` to avoid memory leaks.
#[no_mangle]
pub unsafe extern "C" fn infera_set_autoload_dir(path: *const c_char) -> *mut c_char {
    let result = (|| -> Result<serde_json::Value, error::InferaError> {
        if path.is_null() {
            return Err(error::InferaError::NullPointer);
        }
        let path_str = CStr::from_ptr(path).to_str()?;

        let mut loaded = Vec::new();
        let mut errors = Vec::new();

        let entries =
            fs::read_dir(path_str).map_err(|e| error::InferaError::IoError(e.to_string()))?;
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.is_file() && file_path.extension().is_some_and(|ext| ext == "onnx") {
                if let Some(name) = file_path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(full_path) = file_path.to_str() {
                        match engine::load_model_impl(name, full_path) {
                            Ok(_) => loaded.push(name.to_string()),
                            Err(e) => {
                                errors.push(json!({ "file": full_path, "error": e.to_string() }))
                            }
                        }
                    }
                }
            }
        }
        Ok(json!({"loaded": loaded, "errors": errors}))
    })();

    let final_json = result.unwrap_or_else(|e| {
        error::set_last_error(&e);
        json!({"error": e.to_string()})
    });
    let json_str = serde_json::to_string(&final_json).unwrap_or_default();
    CString::new(json_str).unwrap_or_default().into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_infera_get_version() {
        let version_ptr = infera_get_version();
        let version_json = unsafe { CStr::from_ptr(version_ptr).to_str().unwrap() };
        let version_data: serde_json::Value = serde_json::from_str(version_json).unwrap();

        assert!(version_data["version"].is_string());
        assert!(version_data["onnx_backend"].is_string());
        assert!(version_data["model_cache_dir"].is_string());

        unsafe { infera_free(version_ptr) };
    }

    #[test]
    fn test_infera_set_autoload_dir() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("linear.onnx");
        fs::copy("../test/models/linear.onnx", &model_path).unwrap();

        let path_cstr = CString::new(dir.path().to_str().unwrap()).unwrap();
        let result_ptr = unsafe { infera_set_autoload_dir(path_cstr.as_ptr()) };
        let result_json = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let result_data: serde_json::Value = serde_json::from_str(result_json).unwrap();

        assert_eq!(result_data["loaded"].as_array().unwrap().len(), 1);
        assert_eq!(result_data["loaded"][0], "linear");
        assert_eq!(result_data["errors"].as_array().unwrap().len(), 0);

        unsafe { infera_free(result_ptr) };
    }

    #[test]
    fn test_infera_set_autoload_dir_non_existent() {
        let dir = tempdir().unwrap();
        let non_existent_path = dir.path().join("non_existent");
        let path_cstr = CString::new(non_existent_path.to_str().unwrap()).unwrap();
        let result_ptr = unsafe { infera_set_autoload_dir(path_cstr.as_ptr()) };
        let result_json = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let result_data: serde_json::Value = serde_json::from_str(result_json).unwrap();

        assert!(result_data["error"].is_string());

        unsafe { infera_free(result_ptr) };
    }

    #[test]
    fn test_infera_set_autoload_dir_invalid_model() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("invalid.onnx");
        fs::write(&model_path, "invalid onnx data").unwrap();

        let path_cstr = CString::new(dir.path().to_str().unwrap()).unwrap();
        let result_ptr = unsafe { infera_set_autoload_dir(path_cstr.as_ptr()) };
        let result_json = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let result_data: serde_json::Value = serde_json::from_str(result_json).unwrap();

        assert_eq!(result_data["loaded"].as_array().unwrap().len(), 0);
        assert_eq!(result_data["errors"].as_array().unwrap().len(), 1);
        assert_eq!(
            result_data["errors"][0]["file"],
            model_path.to_str().unwrap()
        );

        unsafe { infera_free(result_ptr) };
    }

    #[test]
    fn test_ffi_null_pointers() {
        let model_name = CString::new("test").unwrap();
        let path = CString::new("path").unwrap();
        let null_ptr = std::ptr::null();

        // Test infera_load_model
        unsafe {
            assert_eq!(infera_load_model(null_ptr, path.as_ptr()), -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));

            assert_eq!(infera_load_model(model_name.as_ptr(), null_ptr), -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));
        }

        // Test infera_unload_model
        unsafe {
            assert_eq!(infera_unload_model(null_ptr), -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));
        }

        // Test infera_predict
        let data: [f32; 1] = [0.0];
        unsafe {
            let result = infera_predict(null_ptr, data.as_ptr(), 1, 1);
            assert_eq!(result.status, -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));

            let result = infera_predict(model_name.as_ptr(), std::ptr::null(), 1, 1);
            assert_eq!(result.status, -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));
        }

        // Test infera_predict_from_blob
        let blob: [u8; 4] = [0; 4];
        unsafe {
            let result = infera_predict_from_blob(null_ptr, blob.as_ptr(), 4);
            assert_eq!(result.status, -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));

            let result = infera_predict_from_blob(model_name.as_ptr(), std::ptr::null(), 4);
            assert_eq!(result.status, -1);
            let error = CStr::from_ptr(infera_last_error());
            assert!(error.to_str().unwrap().contains("Null pointer passed"));
        }

        // Test infera_get_model_info
        unsafe {
            let info_ptr = infera_get_model_info(null_ptr);
            let info_json = CStr::from_ptr(info_ptr).to_str().unwrap();
            let info_data: serde_json::Value = serde_json::from_str(info_json).unwrap();
            assert!(info_data["error"].is_string());
            assert!(info_data["error"]
                .as_str()
                .unwrap()
                .contains("Null pointer passed"));
            infera_free(info_ptr);
        }

        // Test infera_set_autoload_dir
        unsafe {
            let result_ptr = infera_set_autoload_dir(null_ptr);
            let result_json = CStr::from_ptr(result_ptr).to_str().unwrap();
            let result_data: serde_json::Value = serde_json::from_str(result_json).unwrap();
            assert!(result_data["error"].is_string());
            assert!(result_data["error"]
                .as_str()
                .unwrap()
                .contains("Null pointer passed"));
            infera_free(result_ptr);
        }
    }

    #[test]
    fn test_infera_predict_from_blob_invalid_size() {
        let model_name = CString::new("test_model").unwrap();
        let model_path = CString::new("../test/models/linear.onnx").unwrap();
        unsafe {
            infera_load_model(model_name.as_ptr(), model_path.as_ptr());
        }

        // Blob size is 5, which is not a multiple of 4 (size of f32)
        let blob: [u8; 5] = [0; 5];
        let result = unsafe { infera_predict_from_blob(model_name.as_ptr(), blob.as_ptr(), 5) };
        assert_eq!(result.status, -1);

        let error = unsafe { CStr::from_ptr(infera_last_error()) };
        assert!(error
            .to_str()
            .unwrap()
            .contains("Invalid BLOB size: length must be a multiple of 4"));

        unsafe {
            infera_unload_model(model_name.as_ptr());
        }
    }

    #[test]
    fn test_infera_predict_invalid_shape() {
        // Load a simple model that expects input shape [1,3]
        let model_name = CString::new("shape_check").unwrap();
        let model_path = CString::new("../test/models/linear.onnx").unwrap();
        unsafe {
            assert_eq!(
                infera_load_model(model_name.as_ptr(), model_path.as_ptr()),
                0
            );
        }

        // Provide rows=1, cols=2 while model expects 3 features -> should error
        let data: [f32; 2] = [0.0, 0.0];
        let res = unsafe { infera_predict(model_name.as_ptr(), data.as_ptr(), 1, 2) };
        assert_eq!(res.status, -1);
        let err = unsafe { CStr::from_ptr(infera_last_error()) };
        let msg = err.to_str().unwrap();
        assert!(
            msg.contains("Invalid input shape"),
            "unexpected error: {}",
            msg
        );

        unsafe {
            infera_unload_model(model_name.as_ptr());
        }
    }

    #[test]
    fn test_infera_get_model_info_nonexistent_returns_error_json() {
        let name = CString::new("__missing_model__").unwrap();
        let info_ptr = unsafe { infera_get_model_info(name.as_ptr()) };
        let info_json = unsafe { CStr::from_ptr(info_ptr).to_str().unwrap() };
        let value: serde_json::Value = serde_json::from_str(info_json).unwrap();
        assert!(
            value.get("error").is_some(),
            "expected error field in JSON: {}",
            info_json
        );
        unsafe { infera_free(info_ptr) };
    }

    #[test]
    fn test_infera_get_cache_info_includes_configured_limit() {
        let cache_info_ptr = infera_get_cache_info();
        let cache_info_json = unsafe { CStr::from_ptr(cache_info_ptr).to_str().unwrap() };
        let value: serde_json::Value = serde_json::from_str(cache_info_json).unwrap();
        let size_limit = value["size_limit_bytes"]
            .as_u64()
            .expect("size_limit_bytes should be u64");
        assert_eq!(size_limit, crate::config::CONFIG.cache_size_limit);
        unsafe { infera_free(cache_info_ptr) };
    }
}
