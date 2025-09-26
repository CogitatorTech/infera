use std::ffi::{c_char, CString};
use std::ptr;

/// # Safety
///
/// This function returns a raw C string and assumes the caller will handle memory management.
#[no_mangle]
pub unsafe extern "C" fn hello_infera() -> *mut c_char {
    let s = CString::new("Hello from Infera!").unwrap();
    s.into_raw()
}

/// # Safety
/// Frees a string previously returned by `hello_infera`. Passing any other pointer is UB.
#[no_mangle]
pub unsafe extern "C" fn infera_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // Reconstruct the CString so it gets dropped and deallocated.
    let _ = CString::from_raw(ptr);
}
