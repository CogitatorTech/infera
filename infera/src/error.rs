// src/error.rs
// Contains the InferaError enum and thread-local error handling logic.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::str::Utf8Error as StdUtf8Error;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
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
    #[error("HTTP request failed: {0}")]
    HttpRequestError(String),
    #[error("Failed to create cache directory: {0}")]
    CacheDirError(String),
    #[error("Invalid BLOB size: length must be a multiple of 4")]
    InvalidBlobSize,
    #[error("BLOB data does not match model's expected input shape. Expected {expected} elements, but BLOB contained {actual}.")]
    BlobShapeMismatch { expected: usize, actual: usize },
}

impl From<StdUtf8Error> for InferaError {
    fn from(_: StdUtf8Error) -> Self {
        InferaError::Utf8Error
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(err: &InferaError) {
    if let Ok(c_string) = CString::new(err.to_string()) {
        LAST_ERROR.with(|cell| {
            *cell.borrow_mut() = Some(c_string);
        });
    }
}

#[no_mangle]
pub extern "C" fn infera_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match *cell.borrow() {
        Some(ref c_string) => c_string.as_ptr(),
        None => std::ptr::null(),
    })
}
