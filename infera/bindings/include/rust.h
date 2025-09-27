#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>



extern "C" {

/// # Safety
///
/// This function returns a raw C string and assumes the caller will handle memory management.
char *hello_infera();

/// # Safety
/// Frees a string previously returned by `hello_infera`. Passing any other pointer is UB.
void infera_free(char *ptr);

} // extern "C"
