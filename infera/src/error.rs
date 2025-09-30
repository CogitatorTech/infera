// src/error.rs
// Contains the InferaError enum and thread-local error handling logic.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::str::Utf8Error as StdUtf8Error;
use thiserror::Error;

/// Represents all possible errors that can occur within the Infera library.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum InferaError {
    /// Error indicating that a requested model could not be found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    /// Error for when the provided input tensor shape does not match the model's expected shape.
    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape {
        /// The expected shape.
        expected: String,
        /// The actual shape provided.
        actual: String,
    },
    /// An error originating from the underlying ONNX inference engine (e.g., Tract).
    #[error("ONNX error: {0}")]
    OnnxError(String),
    /// Error indicating a failure in memory allocation.
    #[error("Memory allocation error")]
    MemoryError,
    /// Error for when a C string from the FFI boundary is not valid UTF-8.
    #[error("Invalid UTF-8 string")]
    Utf8Error,
    /// Error for when a null pointer is passed as an argument to an FFI function.
    #[error("Null pointer passed")]
    NullPointer,
    /// An I/O error that occurred while reading a file or making a network request.
    #[error("IO error: {0}")]
    IoError(String),
    /// An error during the serialization or deserialization of JSON data.
    #[error("JSON serialization error: {0}")]
    JsonError(String),
    /// Error for when a feature is required but not enabled at compile time (e.g., "tract").
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
    /// An error that occurred during an HTTP request to fetch a remote model.
    #[error("HTTP request failed: {0}")]
    HttpRequestError(String),
    /// Error for when the model cache directory cannot be created.
    #[error("Failed to create cache directory: {0}")]
    CacheDirError(String),
    /// Error for when `infera_predict_from_blob` receives a blob whose size is not a multiple of 4.
    #[error("Invalid BLOB size: length must be a multiple of 4")]
    InvalidBlobSize,
    /// Error indicating a mismatch between the number of elements in a blob and the model's expected input tensor size.
    #[error("BLOB data does not match model's expected input shape. Expected {expected} elements, but BLOB contained {actual}."
    )]
    BlobShapeMismatch {
        /// The number of elements the model expected.
        expected: usize,
        /// The actual number of elements found in the blob.
        actual: usize,
    },
}

impl From<StdUtf8Error> for InferaError {
    fn from(_: StdUtf8Error) -> Self {
        InferaError::Utf8Error
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Sets the last error for the current thread.
///
/// This stores the given error in a thread-local variable so it can be retrieved
/// later by FFI clients using `infera_last_error`.
pub(crate) fn set_last_error(err: &InferaError) {
    if let Ok(c_string) = CString::new(err.to_string()) {
        LAST_ERROR.with(|cell| {
            *cell.borrow_mut() = Some(c_string);
        });
    }
}

/// Retrieves the last error message set in the current thread.
///
/// After an FFI function returns an error code, this function can be called
/// to get a more descriptive, human-readable error message.
///
/// # Returns
///
/// A pointer to a null-terminated C string containing the last error message.
/// Returns a null pointer if no error has occurred since the last call.
/// The caller **must not** free this pointer, as it is managed by a thread-local static variable.
#[no_mangle]
pub extern "C" fn infera_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match *cell.borrow() {
        Some(ref c_string) => c_string.as_ptr(),
        None => std::ptr::null(),
    })
}
