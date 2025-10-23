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

/// Compute (rows, cols) from a tensor shape by flattening all dimensions after the first into cols.
///
/// Rules:
/// - Scalar (shape = []): (1, 1)
/// - 1D (shape = [n]): (n, 1)
/// - N-D (shape = [d0, d1, d2, ...]): (d0, product(d1..))
pub(crate) fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => {
            let rows = shape[0];
            let cols = shape[1..].iter().copied().product::<usize>().max(1);
            (rows, cols)
        }
    }
}

/// Loads, compiles, and stores an ONNX model.
///
/// This function reads an ONNX model from the given path, uses the Tract library
/// to parse, optimize, and compile it into a runnable plan. It also extracts
/// metadata like input/output shapes. The resulting `OnnxModel` is then
/// inserted into the global `MODELS` map.
///
/// # Arguments
///
/// * `name` - The name to assign to the loaded model.
/// * `path` - The file system path to the `.onnx` model file.
///
/// # Returns
///
/// * `Ok(())` on successful loading and compilation.
/// * `Err(InferaError)` if the model cannot be found, parsed, or compiled.
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

/// A stub for `load_model_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn load_model_impl(_name: &str, _path: &str) -> Result<(), InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Runs inference with a given model and input tensor data.
///
/// This function retrieves the specified model from the global `MODELS` store,
/// constructs a Tract tensor from the raw input data, runs the model,
/// and packages the output into an `InferaInferenceResult`.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model to use for inference.
/// * `data` - A pointer to the raw f32 tensor data.
/// * `rows` - The number of rows in the input tensor.
/// * `cols` - The number of columns in the input tensor.
///
/// # Returns
///
/// * `Ok(InferaInferenceResult)` containing the output tensor data and metadata.
/// * `Err(InferaError)` if the model is not found or if an error occurs during inference.
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

    // If model inner dimensions (after the first/batch dim) are all known (>0),
    // validate that the provided `cols` matches their product. This yields clearer errors
    // than deferring to the backend.
    if !model.input_shape.is_empty() {
        let inner_dims = &model.input_shape[1..];
        if inner_dims.iter().all(|&d| d > 0) {
            let expected_inner: usize = inner_dims.iter().map(|&d| d as usize).product();
            if cols != expected_inner {
                return Err(InferaError::InvalidInputShape {
                    expected: format!("batch x {:?}", inner_dims),
                    actual: format!("{} x {}", rows, cols),
                });
            }
        }
    }

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
    let (output_rows, output_cols) = shape_rows_cols(output_array.shape());
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

/// A stub for `run_inference_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
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

/// Runs inference with a given model and raw BLOB input data.
///
/// This function is similar to `run_inference_impl` but takes a raw byte slice (`BLOB`)
/// as input. It converts the bytes to `f32` values, validates the shape against the
/// model's expected input, and attempts to infer the batch size for models with
/// dynamic input dimensions.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model to use for inference.
/// * `blob_data` - A pointer to the raw byte data.
/// * `blob_len` - The length of the byte data slice.
///
/// # Returns
///
/// * `Ok(InferaInferenceResult)` containing the output tensor data and metadata.
/// * `Err(InferaError)` if the model is not found, the blob size is invalid,
///   the shape does not match, or an error occurs during inference.
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
        .map(|chunk| {
            // SAFETY: chunks_exact(4) guarantees exactly 4 bytes, so this conversion cannot fail
            let array: [u8; 4] = match chunk.try_into() {
                Ok(arr) => arr,
                Err(_) => [0u8; 4], // This branch is unreachable but satisfies clippy
            };
            f32::from_ne_bytes(array)
        })
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
    let (output_rows, output_cols) = shape_rows_cols(output_array.shape());
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

/// A stub for `run_inference_blob_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
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

/// Retrieves metadata for a loaded model as a JSON string.
///
/// This function looks up the model by name and serializes its metadata
/// (name, input shape, output shape) into a JSON string.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model.
///
/// # Returns
///
/// * `Ok(String)` containing the JSON metadata.
/// * `Err(InferaError)` if the model is not found or if JSON serialization fails.
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

/// A stub for `get_model_metadata_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn get_model_metadata_impl(_model_name: &str) -> Result<String, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_rows_cols() {
        assert_eq!(shape_rows_cols(&[]), (1, 1));
        assert_eq!(shape_rows_cols(&[5]), (5, 1));
        assert_eq!(shape_rows_cols(&[2, 3]), (2, 3));
        assert_eq!(shape_rows_cols(&[2, 3, 4]), (2, 12));
        assert_eq!(shape_rows_cols(&[1, 1, 1, 1]), (1, 1));
    }
}
