// src/lib.rs
// The public C API layer and module declarations.

use serde_json::json;
use std::env;
use std::ffi::{c_char, CStr, CString};
use std::fs;

// Declare the internal modules
mod engine;
mod error;
mod ffi_utils;
mod http;
mod model;

// Re-export the public FFI utility functions and types
pub use error::infera_last_error;
pub use ffi_utils::{infera_free, infera_free_result, InferaInferenceResult};

/// Loads a model from a file or URL.
///
/// # Safety
/// The `name` and `path` pointers must be valid, null-terminated C strings.
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

/// Unloads a model.
///
/// # Safety
/// The `name` pointer must be a valid, null-terminated C string.
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

/// Runs inference on a model with the given input data.
///
/// # Safety
/// The `model_name` and `data` pointers must be valid. `model_name` must be a
/// null-terminated C string. `data` must point to a contiguous block of memory
/// of size `rows * cols * size_of<f32>()`.
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

/// Runs inference on a model with input data from a BLOB.
///
/// # Safety
/// The `model_name` and `blob_data` pointers must be valid. `model_name` must be a
/// null-terminated C string. `blob_data` must point to a contiguous block of
/// memory of size `blob_len`.
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

/// Gets information about a loaded model.
///
/// # Safety
/// The `model_name` pointer must be a valid, null-terminated C string.
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

#[no_mangle]
pub extern "C" fn infera_get_loaded_models() -> *mut c_char {
    let models = model::MODELS.read();
    let list: Vec<String> = models.keys().cloned().collect();
    let joined = serde_json::to_string(&list).unwrap_or_else(|_| "[]".to_string());
    CString::new(joined).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn infera_get_version() -> *mut c_char {
    let cache_dir = env::temp_dir().join("infera_cache");
    let info = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "onnx_backend": if cfg!(feature = "tract") { "tract" } else { "disabled" },
        "model_cache_dir": cache_dir.to_str(),
    });
    let json_str = serde_json::to_string(&info).unwrap_or_default();
    CString::new(json_str).unwrap_or_default().into_raw()
}

/// Sets a directory to automatically load models from.
///
/// # Safety
/// The `path` pointer must be a valid, null-terminated C string.
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

    let final_json = match result {
        Ok(json) => json,
        Err(e) => {
            error::set_last_error(&e);
            json!({"error": e.to_string()})
        }
    };
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
        fs::copy("../tests/models/linear.onnx", &model_path).unwrap();

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
}
