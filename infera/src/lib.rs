#![allow(unsafe_op_in_unsafe_fn)]
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicPtr, Ordering};

// Model management and inference
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json;
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

#[derive(Error, Debug)]
pub enum InferaError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape { expected: String, actual: String },
    #[error("ONNX error: {0}")]
    OnnxError(String),
    #[error("Memory allocation error")]
    MemoryError,
    #[error("Invalid UTF-8 string")]
    Utf8Error,
    #[error("Null pointer passed")]
    NullPointer,
    #[error("IO error: {0}")]
    IoError(String),
    #[error("JSON serialization error: {0}")]
    JsonError(String),
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

static LAST_ERROR: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::new()));

fn set_last_error(err: &InferaError) {
    *LAST_ERROR.write() = err.to_string();
}

#[cfg(feature = "tract")]
struct OnnxModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_shape: Vec<i64>,
    output_shape: Vec<i64>,
    name: String,
}

#[cfg(not(feature = "tract"))]
struct OnnxModel {
    name: String,
}

static MODELS: Lazy<RwLock<HashMap<String, OnnxModel>>> = Lazy::new(|| RwLock::new(HashMap::new()));

#[unsafe(no_mangle)]
pub unsafe extern "C" fn infera_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = CString::from_raw(ptr);
}

#[unsafe(no_mangle)]
pub extern "C" fn infera_last_error() -> *const c_char {
    let guard = LAST_ERROR.read();
    static LAST_ERROR_CSTR: AtomicPtr<c_char> = AtomicPtr::new(std::ptr::null_mut());

    unsafe {
        if let Some(old) = std::ptr::NonNull::new(LAST_ERROR_CSTR.load(Ordering::SeqCst)) {
            drop(CString::from_raw(old.as_ptr()));
        }
        let c =
            CString::new(guard.as_str()).unwrap_or_else(|_| CString::new("invalid utf8").unwrap());
        let ptr = c.into_raw();
        LAST_ERROR_CSTR.store(ptr, Ordering::SeqCst);
        ptr as *const c_char
    }
}

#[repr(C)]
pub struct InferaInferenceResult {
    pub data: *mut f32,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub status: i32,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn infera_free_result(res: InferaInferenceResult) {
    if !res.data.is_null() && res.len > 0 {
        let _ = Vec::from_raw_parts(res.data, res.len, res.len);
    }
}

#[repr(C)]
pub struct ModelMetadata {
    pub input_shape: *mut i64,
    pub input_shape_len: usize,
    pub output_shape: *mut i64,
    pub output_shape_len: usize,
    pub input_count: usize,
    pub output_count: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn infera_free_metadata(meta: ModelMetadata) {
    if !meta.input_shape.is_null() && meta.input_shape_len > 0 {
        let _ = Vec::from_raw_parts(meta.input_shape, meta.input_shape_len, meta.input_shape_len);
    }
    if !meta.output_shape.is_null() && meta.output_shape_len > 0 {
        let _ = Vec::from_raw_parts(
            meta.output_shape,
            meta.output_shape_len,
            meta.output_shape_len,
        );
    }
}

// Core model management functions
#[unsafe(no_mangle)]
pub extern "C" fn infera_load_onnx_model(name: *const c_char, path: *const c_char) -> i32 {
    let result = (|| -> Result<(), InferaError> {
        if name.is_null() || path.is_null() {
            return Err(InferaError::NullPointer);
        }

        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;
        let path_str = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;

        load_model_impl(name_str, path_str)
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e);
            -1
        }
    }
}

#[cfg(feature = "tract")]
fn load_model_impl(name: &str, path: &str) -> Result<(), InferaError> {
    // Load ONNX model using tract
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_optimized()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_runnable()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    // Get input/output shapes for validation
    let input_facts = model
        .model()
        .input_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_facts = model
        .model()
        .output_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    let input_shape: Vec<i64> = input_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();
    let output_shape: Vec<i64> = output_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();

    let onnx_model = OnnxModel {
        model,
        input_shape,
        output_shape,
        name: name.to_string(),
    };

    // Put the model in thread-safe map
    let mut models = MODELS.write();
    models.insert(name.to_string(), onnx_model);

    Ok(())
}

#[cfg(not(feature = "tract"))]
fn load_model_impl(name: &str, _path: &str) -> Result<(), InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn infera_unload_onnx_model(name: *const c_char) -> i32 {
    let result = (|| -> Result<(), InferaError> {
        if name.is_null() {
            return Err(InferaError::NullPointer);
        }

        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;

        let mut models = MODELS.write();
        if models.remove(name_str).is_some() {
            Ok(())
        } else {
            Err(InferaError::ModelNotFound(name_str.to_string()))
        }
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn infera_run_inference(
    model_name: *const c_char,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> InferaInferenceResult {
    let result = (|| -> Result<InferaInferenceResult, InferaError> {
        if model_name.is_null() || data.is_null() {
            return Err(InferaError::NullPointer);
        }

        let name_str = unsafe { CStr::from_ptr(model_name) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;

        run_inference_impl(name_str, data, rows, cols)
    })();

    match result {
        Ok(res) => res,
        Err(e) => {
            set_last_error(&e);
            InferaInferenceResult {
                data: std::ptr::null_mut(),
                len: 0,
                rows: 0,
                cols: 0,
                status: -1,
            }
        }
    }
}

#[cfg(feature = "tract")]
fn run_inference_impl(
    model_name: &str,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;

    // Convert input data to an owned vector
    let input_data = unsafe { std::slice::from_raw_parts(data, rows * cols) };
    let input_vec = input_data.to_vec();

    // Create Tract tensor directly from raw data using tensor construction
    let input_tensor = tract_onnx::prelude::Tensor::from_shape(&[rows, cols], &input_vec)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let outputs = model
        .model
        .run(tvec!(input_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    // Extract output data
    let output_tensor = outputs
        .into_iter()
        .next()
        .ok_or_else(|| InferaError::OnnxError("No output tensor".to_string()))?;

    let output_array = output_tensor
        .to_array_view::<f32>()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    let output_shape = output_array.shape();
    let output_rows = output_shape[0];
    let output_cols = if output_shape.len() > 1 {
        output_shape[1]
    } else {
        1
    };

    // Copy output data to an owned vector
    let output_data: Vec<f32> = output_array.iter().cloned().collect();
    let output_len = output_data.len();

    // Convert to raw pointer (caller must free this) - use correct Box syntax
    let output_ptr = Box::into_raw(output_data.into_boxed_slice()) as *mut f32;

    Ok(InferaInferenceResult {
        data: output_ptr,
        len: output_len,
        rows: output_rows,
        cols: output_cols,
        status: 0,
    })
}

#[cfg(not(feature = "tract"))]
fn run_inference_impl(
    _model_name: &str,
    _data: *const f32,
    _rows: usize,
    _cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

// Model metadata functions
#[unsafe(no_mangle)]
pub extern "C" fn infera_get_model_metadata(model_name: *const c_char) -> ModelMetadata {
    let result = (|| -> Result<ModelMetadata, InferaError> {
        if model_name.is_null() {
            return Err(InferaError::NullPointer);
        }

        let name_str = unsafe { CStr::from_ptr(model_name) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;

        get_model_metadata_impl(name_str)
    })();

    match result {
        Ok(meta) => meta,
        Err(e) => {
            set_last_error(&e);
            ModelMetadata {
                input_shape: std::ptr::null_mut(),
                input_shape_len: 0,
                output_shape: std::ptr::null_mut(),
                output_shape_len: 0,
                input_count: 0,
                output_count: 0,
            }
        }
    }
}

#[cfg(feature = "tract")]
fn get_model_metadata_impl(model_name: &str) -> Result<ModelMetadata, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;

    // Clone the shape vectors to transfer ownership
    let input_shape = model.input_shape.clone();
    let output_shape = model.output_shape.clone();

    let input_shape_len = input_shape.len();
    let output_shape_len = output_shape.len();

    // Convert to raw pointers (caller must free these) - use correct Box syntax
    let input_shape_ptr = Box::into_raw(input_shape.into_boxed_slice()) as *mut i64;
    let output_shape_ptr = Box::into_raw(output_shape.into_boxed_slice()) as *mut i64;

    Ok(ModelMetadata {
        input_shape: input_shape_ptr,
        input_shape_len,
        output_shape: output_shape_ptr,
        output_shape_len,
        input_count: 1,  // Assuming single input for now
        output_count: 1, // Assuming single output for now
    })
}

#[cfg(not(feature = "tract"))]
fn get_model_metadata_impl(_model_name: &str) -> Result<ModelMetadata, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

// List all loaded models
#[unsafe(no_mangle)]
pub extern "C" fn infera_list_models() -> *mut c_char {
    let result = (|| -> Result<String, InferaError> {
        let models = MODELS.read();
        let model_names: Vec<&String> = models.keys().collect();
        serde_json::to_string(&model_names).map_err(|e| InferaError::JsonError(e.to_string()))
    })();

    match result {
        Ok(json) => {
            let s = CString::new(json).unwrap_or_else(|_| CString::new("[]").unwrap());
            s.into_raw()
        }
        Err(e) => {
            set_last_error(&e);
            let s = CString::new("[]").unwrap();
            s.into_raw()
        }
    }
}

// Get detailed model information as JSON
#[unsafe(no_mangle)]
pub extern "C" fn infera_model_info(model_name: *const c_char) -> *mut c_char {
    let result = (|| -> Result<String, InferaError> {
        if model_name.is_null() {
            return Err(InferaError::NullPointer);
        }

        let name_str = unsafe { CStr::from_ptr(model_name) }
            .to_str()
            .map_err(|_| InferaError::Utf8Error)?;

        get_model_info_impl(name_str)
    })();

    match result {
        Ok(json) => {
            let s = CString::new(json).unwrap_or_else(|_| CString::new("{}").unwrap());
            s.into_raw()
        }
        Err(e) => {
            set_last_error(&e);
            let error_json = format!("{{\"error\": \"{}\"}}", e);
            let s = CString::new(error_json).unwrap_or_else(|_| CString::new("{}").unwrap());
            s.into_raw()
        }
    }
}

#[cfg(feature = "tract")]
fn get_model_info_impl(model_name: &str) -> Result<String, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;

    let info = serde_json::json!({
        "name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "loaded": true
    });

    serde_json::to_string(&info).map_err(|e| InferaError::JsonError(e.to_string()))
}

#[cfg(not(feature = "tract"))]
fn get_model_info_impl(_model_name: &str) -> Result<String, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}
