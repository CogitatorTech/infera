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
