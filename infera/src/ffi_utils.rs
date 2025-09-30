// src/ffi_utils.rs
// Contains C-compatible structs and memory management functions for the FFI boundary.

use std::ffi::{c_char, CString};

#[repr(C)]
pub struct InferaInferenceResult {
    pub data: *mut f32,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub status: i32,
}

impl InferaInferenceResult {
    // Helper function to create an error result
    pub fn error() -> Self {
        InferaInferenceResult {
            data: std::ptr::null_mut(),
            len: 0,
            rows: 0,
            cols: 0,
            status: -1,
        }
    }
}

/// Frees a C string that was allocated by Rust.
///
/// # Safety
///
/// The `ptr` must be a pointer to a C string that was allocated by Rust's
/// `CString::into_raw`. Calling this function with a pointer that was not
/// allocated by `CString::into_raw` will result in undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn infera_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Frees the memory allocated for an `InferaInferenceResult`.
///
/// # Safety
///
/// The `res.data` pointer must have been allocated by Rust's `Vec` and the `res.len`
/// must be the correct length of the allocated memory. Calling this function
/// with a result that was not created by this library can lead to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn infera_free_result(res: InferaInferenceResult) {
    if !res.data.is_null() && res.len > 0 {
        let _ = Vec::from_raw_parts(res.data, res.len, res.len);
    }
}
