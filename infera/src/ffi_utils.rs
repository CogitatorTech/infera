// Contains C-compatible structs and memory management functions for the FFI boundary.

use std::ffi::{c_char, CString};

/// A C-compatible struct that holds the result of an inference operation.
///
/// This struct is returned by `infera_predict` and `infera_predict_from_blob`.
/// The caller is responsible for freeing the `data` pointer by passing the entire
/// struct to `infera_free_result`.
#[repr(C)]
pub struct InferaInferenceResult {
    /// A pointer to the raw output data of the model, stored as a flat array of `f32`.
    pub data: *mut f32,
    /// The total number of elements in the `data` array.
    pub len: usize,
    /// The number of rows in the output tensor.
    pub rows: usize,
    /// The number of columns in the output tensor.
    pub cols: usize,
    /// The status of the inference operation. `0` for success, `-1` for failure.
    pub status: i32,
}

impl InferaInferenceResult {
    /// Creates a new `InferaInferenceResult` representing an error state.
    ///
    /// The `data` pointer is null and the `status` is set to -1.
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

/// Frees a heap-allocated C string that was returned by an Infera FFI function.
///
/// This function should be used to free the memory for strings returned by functions
/// like `infera_get_model_info`, `infera_get_loaded_models`, and `infera_get_version`.
///
/// # Safety
///
/// The `ptr` must be a non-null pointer to a C string that was previously allocated
/// by Rust's `CString::into_raw`. Passing any other pointer (e.g., a string literal,
/// a pointer from a different allocator, or a null pointer) will result in undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn infera_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Frees the data buffer within an `InferaInferenceResult`.
///
/// This function must be called on every `InferaInferenceResult` returned from
/// `infera_predict` or `infera_predict_from_blob` to prevent memory leaks.
///
/// # Safety
///
/// * The `res` struct must be a value that was returned by an Infera prediction function.
/// * The `res.data` pointer must have been allocated by Rust's `Vec<f32>` and not yet freed.
/// * The `res.len` field must accurately represent the capacity of the allocated buffer.
///
/// Calling this function on a manually-created struct or calling it more than once
/// on the same result will lead to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn infera_free_result(res: InferaInferenceResult) {
    if !res.data.is_null() {
        // SAFETY: `res.data` was allocated from a Box<[f32]> via `into_raw` with length `res.len`.
        // Reconstruct the slice pointer and drop it to free the allocation correctly.
        let slice_ptr: *mut [f32] = std::ptr::slice_from_raw_parts_mut(res.data, res.len);
        let _ = Box::from_raw(slice_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infera_free_result_zero_len_non_null() {
        // Allocate an empty slice on heap and convert to raw
        let empty: Box<[f32]> = Vec::<f32>::new().into_boxed_slice();
        let ptr = Box::into_raw(empty) as *mut f32;
        let res = InferaInferenceResult {
            data: ptr,
            len: 0,
            rows: 0,
            cols: 0,
            status: 0,
        };
        unsafe { infera_free_result(res) }; // should not panic or leak
    }

    #[test]
    fn test_infera_free_result_non_empty() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let len = data.len();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut f32;
        let res = InferaInferenceResult {
            data: ptr,
            len,
            rows: 1,
            cols: len,
            status: 0,
        };
        unsafe { infera_free_result(res) }; // should free without UB
    }
}
