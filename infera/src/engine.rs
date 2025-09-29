// src/engine.rs
// Contains the core ONNX inference logic using the Tract library.

use crate::error::InferaError;
use crate::ffi_utils::InferaInferenceResult;
use crate::model::{OnnxModel, MODELS};
use serde_json::json;
use std::convert::TryInto;
use std::mem;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

#[cfg(feature = "tract")]
pub(crate) fn load_model_impl(name: &str, path: &str) -> Result<(), InferaError> {
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_optimized()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_runnable()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
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
    MODELS.write().insert(name.to_string(), onnx_model);
    Ok(())
}

#[cfg(not(feature = "tract"))]
pub(crate) fn load_model_impl(_name: &str, _path: &str) -> Result<(), InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

#[cfg(feature = "tract")]
pub(crate) fn run_inference_impl(
    model_name: &str,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    let input_data = unsafe { std::slice::from_raw_parts(data, rows * cols) };
    let input_tensor = Tensor::from_shape(&[rows, cols], input_data)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let outputs = model
        .model
        .run(tvec!(input_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
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
    let output_data: Vec<f32> = output_array.iter().cloned().collect();
    let output_len = output_data.len();
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
pub(crate) fn run_inference_impl(
    _model_name: &str,
    _data: *const f32,
    _rows: usize,
    _cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

#[cfg(feature = "tract")]
pub(crate) fn run_inference_blob_impl(
    model_name: &str,
    blob_data: *const u8,
    blob_len: usize,
) -> Result<InferaInferenceResult, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    if !blob_len.is_multiple_of(mem::size_of::<f32>()) {
        return Err(InferaError::InvalidBlobSize);
    }
    let blob_bytes = unsafe { std::slice::from_raw_parts(blob_data, blob_len) };
    let float_vec: Vec<f32> = blob_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    let expected_elements: usize = model
        .input_shape
        .iter()
        .filter(|&&d| d > 0)
        .map(|&d| d as usize)
        .product();
    if !float_vec.len().is_multiple_of(expected_elements) {
        return Err(InferaError::BlobShapeMismatch {
            expected: expected_elements,
            actual: float_vec.len(),
        });
    }
    let batch_size = float_vec.len() / expected_elements;
    let final_shape: Vec<usize> = model
        .input_shape
        .iter()
        .map(|&d| if d == -1 { batch_size } else { d as usize })
        .collect();
    let input_tensor = Tensor::from_shape(&final_shape, &float_vec)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let outputs = model
        .model
        .run(tvec!(input_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
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
    let output_data: Vec<f32> = output_array.iter().cloned().collect();
    let output_len = output_data.len();
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
pub(crate) fn run_inference_blob_impl(
    _model_name: &str,
    _blob_data: *const u8,
    _blob_len: usize,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

#[cfg(feature = "tract")]
pub(crate) fn get_model_metadata_impl(model_name: &str) -> Result<String, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    let info = json!({
        "name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "loaded": true
    });
    serde_json::to_string(&info).map_err(|e| InferaError::JsonError(e.to_string()))
}

#[cfg(not(feature = "tract"))]
pub(crate) fn get_model_metadata_impl(_model_name: &str) -> Result<String, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}
