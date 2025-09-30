/* Generated with cbindgen */
/* DO NOT EDIT */


#ifndef INFERA_H
#define INFERA_H

#pragma once

/* Generated with cbindgen:0.26.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
namespace infera {
#endif // __cplusplus

typedef struct InferaInferenceResult {
  float *data;
  uintptr_t len;
  uintptr_t rows;
  uintptr_t cols;
  int32_t status;
} InferaInferenceResult;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Loads a model from a file or URL.
 *
 * # Safety
 * The `name` and `path` pointers must be valid, null-terminated C strings.
 */
 int32_t infera_load_model(const char *name, const char *path);

/**
 * Unloads a model.
 *
 * # Safety
 * The `name` pointer must be a valid, null-terminated C string.
 */
 int32_t infera_unload_model(const char *name);

/**
 * Runs inference on a model with the given input data.
 *
 * # Safety
 * The `model_name` and `data` pointers must be valid. `model_name` must be a
 * null-terminated C string. `data` must point to a contiguous block of memory
 * of size `rows * cols * size_of<f32>()`.
 */

struct InferaInferenceResult infera_predict(const char *model_name,
                                            const float *data,
                                            uintptr_t rows,
                                            uintptr_t cols);

/**
 * Runs inference on a model with input data from a BLOB.
 *
 * # Safety
 * The `model_name` and `blob_data` pointers must be valid. `model_name` must be a
 * null-terminated C string. `blob_data` must point to a contiguous block of
 * memory of size `blob_len`.
 */

struct InferaInferenceResult infera_predict_from_blob(const char *model_name,
                                                      const uint8_t *blob_data,
                                                      uintptr_t blob_len);

/**
 * Gets information about a loaded model.
 *
 * # Safety
 * The `model_name` pointer must be a valid, null-terminated C string.
 */
 char *infera_get_model_info(const char *model_name);

 char *infera_get_loaded_models(void);

 char *infera_get_version(void);

/**
 * Sets a directory to automatically load models from.
 *
 * # Safety
 * The `path` pointer must be a valid, null-terminated C string.
 */
 char *infera_set_autoload_dir(const char *path);

 const char *infera_last_error(void);

/**
 * Frees a C string that was allocated by Rust.
 *
 * # Safety
 *
 * The `ptr` must be a pointer to a C string that was allocated by Rust's
 * `CString::into_raw`. Calling this function with a pointer that was not
 * allocated by `CString::into_raw` will result in undefined behavior.
 */
 void infera_free(char *ptr);

/**
 * Frees the memory allocated for an `InferaInferenceResult`.
 *
 * # Safety
 *
 * The `res.data` pointer must have been allocated by Rust's `Vec` and the `res.len`
 * must be the correct length of the allocated memory. Calling this function
 * with a result that was not created by this library can lead to undefined behavior.
 */
 void infera_free_result(struct InferaInferenceResult res);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#ifdef __cplusplus
} // namespace infera
#endif // __cplusplus

#endif /* INFERA_H */

/* End of generated bindings */
