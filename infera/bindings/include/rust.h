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

/**
 * A C-compatible struct that holds the result of an inference operation.
 *
 * This struct is returned by `infera_predict` and `infera_predict_from_blob`.
 * The caller is responsible for freeing the `data` pointer by passing the entire
 * struct to `infera_free_result`.
 */
typedef struct InferaInferenceResult {
  /**
   * A pointer to the raw output data of the model, stored as a flat array of `f32`.
   */
  float *data;
  /**
   * The total number of elements in the `data` array.
   */
  uintptr_t len;
  /**
   * The number of rows in the output tensor.
   */
  uintptr_t rows;
  /**
   * The number of columns in the output tensor.
   */
  uintptr_t cols;
  /**
   * The status of the inference operation. `0` for success, `-1` for failure.
   */
  int32_t status;
} InferaInferenceResult;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Loads an ONNX model from a local file path or a remote URL and assigns it a unique name.
 *
 * If the `path` starts with "http://" or "https://", the model will be downloaded
 * and cached locally. Otherwise, it will be treated as a local file path.
 *
 * # Arguments
 *
 * * `name` - A pointer to a null-terminated C string representing the unique name for the model.
 * * `path` - A pointer to a null-terminated C string representing the file path or URL of the model.
 *
 * # Returns
 *
 * * `0` on success.
 * * `-1` on failure. Call `infera_last_error()` to get a descriptive error message.
 *
 * # Safety
 *
 * * The `name` and `path` pointers must not be null.
 * * The memory pointed to by `name` and `path` must be valid, null-terminated C strings.
 */

int32_t infera_load_model(const char *name,
                          const char *path);

/**
 * Unloads a model, freeing its associated resources.
 *
 * # Arguments
 *
 * * `name` - A pointer to a null-terminated C string representing the name of the model to unload.
 *
 * # Returns
 *
 * * `0` on success.
 * * `-1` if the model was not found or an error occurred.
 *
 * # Safety
 *
 * * The `name` pointer must not be null.
 * * The memory pointed to by `name` must be a valid, null-terminated C string.
 */
 int32_t infera_unload_model(const char *name);

/**
 * Runs inference on a loaded model with the given input data.
 *
 * The input data is provided as a raw pointer to a flat array of `f32` values.
 * The result of the inference is returned in an `InferaInferenceResult` struct.
 * The caller is responsible for freeing the result using `infera_free_result`.
 *
 * # Arguments
 *
 * * `model_name` - A pointer to a null-terminated C string for the model's name.
 * * `data` - A pointer to the input tensor data, organized as a flat array of `f32`.
 * * `rows` - The number of rows in the input tensor.
 * * `cols` - The number of columns in the input tensor.
 *
 * # Returns
 *
 * An `InferaInferenceResult` struct containing the output tensor data and metadata.
 * If an error occurs, the `status` field of the struct will be `-1`.
 *
 * # Safety
 *
 * * `model_name` and `data` must not be null.
 * * `model_name` must point to a valid, null-terminated C string.
 * * `data` must point to a contiguous block of memory of size `rows * cols * size_of<f32>()`.
 */

struct InferaInferenceResult infera_predict(const char *model_name,
                                            const float *data,
                                            uintptr_t rows,
                                            uintptr_t cols);

/**
 * Runs inference on a loaded model with input data from a raw byte `BLOB`.
 *
 * This function is useful when the input tensor is stored as a `BLOB`. The byte data
 * is interpreted as a flat array of `f32` values (native-endian). The function
 * will attempt to infer the batch size based on the model's expected input shape.
 *
 * # Arguments
 *
 * * `model_name` - A pointer to a null-terminated C string for the model's name.
 * * `blob_data` - A pointer to the input data as a raw byte array.
 * * `blob_len` - The total length of the byte array in `blob_data`.
 *
 * # Returns
 *
 * An `InferaInferenceResult` struct containing the output. The caller is responsible
 * for freeing this result using `infera_free_result`.
 *
 * # Safety
 *
 * * `model_name` and `blob_data` must not be null.
 * * `model_name` must point to a valid, null-terminated C string.
 * * `blob_data` must point to a contiguous block of memory of size `blob_len`.
 * * `blob_len` must be a multiple of `std::mem::size_of::<f32>()`.
 */

struct InferaInferenceResult infera_predict_from_blob(const char *model_name,
                                                      const uint8_t *blob_data,
                                                      uintptr_t blob_len);

/**
 * Retrieves metadata about a specific loaded model as a JSON string.
 *
 * The returned JSON string includes the model's name, and its input and output shapes.
 *
 * # Arguments
 *
 * * `model_name` - A pointer to a null-terminated C string for the model's name.
 *
 * # Returns
 *
 * A pointer to a heap-allocated, null-terminated C string containing JSON.
 * The caller is responsible for freeing this string using `infera_free`.
 * On error (e.g., model not found), the JSON will contain an "error" key.
 *
 * # Safety
 *
 * * The `model_name` pointer must not be null and must point to a valid C string.
 * * The returned pointer must be freed with `infera_free` to avoid memory leaks.
 */
 char *infera_get_model_info(const char *model_name);

/**
 * Returns a JSON array of the names of all currently loaded models.
 *
 * # Returns
 *
 * A pointer to a heap-allocated, null-terminated C string containing a JSON array of strings.
 * The caller is responsible for freeing this string using `infera_free`.
 *
 * # Safety
 *
 * The returned pointer must be freed with `infera_free` to avoid memory leaks.
 */
 char *infera_get_loaded_models(void);

/**
 * Returns a JSON string with version and build information about the Infera library.
 *
 * The JSON object includes the library version, the enabled ONNX backend (e.g., "tract"),
 * and the directory used for caching remote models.
 *
 * # Returns
 *
 * A pointer to a heap-allocated, null-terminated C string containing the version info.
 * The caller is responsible for freeing this string using `infera_free`.
 *
 * # Safety
 *
 * The returned pointer must be freed with `infera_free` to avoid memory leaks.
 */
 char *infera_get_version(void);

/**
 * Clears the entire model cache directory.
 *
 * This removes all cached remote models, freeing up disk space.
 *
 * # Returns
 *
 * * `0` on success.
 * * `-1` on failure. Call `infera_last_error()` to get a descriptive error message.
 *
 * # Safety
 *
 * This function is safe to call at any time.
 */
 int32_t infera_clear_cache(void);

/**
 * Returns cache statistics as a JSON string.
 *
 * The JSON object includes:
 * * `"cache_dir"`: The path to the cache directory.
 * * `"total_size_bytes"`: Total size of cached models in bytes.
 * * `"file_count"`: Number of cached model files.
 * * `"size_limit_bytes"`: The configured cache size limit.
 *
 * # Returns
 *
 * A pointer to a heap-allocated, null-terminated C string containing the cache info.
 * The caller is responsible for freeing this string using `infera_free`.
 *
 * # Safety
 *
 * The returned pointer must be freed with `infera_free` to avoid memory leaks.
 */
 char *infera_get_cache_info(void);

/**
 * Scans a directory for `.onnx` files and loads them into Infera automatically.
 *
 * The name for each model is derived from its filename (without the extension).
 *
 * # Arguments
 *
 * * `path` - A pointer to a null-terminated C string representing the directory path.
 *
 * # Returns
 *
 * A pointer to a heap-allocated C string containing a JSON object with two fields:
 * * `"loaded"`: A list of model names that were successfully loaded.
 * * `"errors"`: A list of objects, each detailing a file that failed to load and the reason.
 *
 * The caller is responsible for freeing this string using `infera_free`.
 *
 * # Safety
 *
 * * The `path` pointer must not be null and must point to a valid C string.
 * * The returned pointer must be freed with `infera_free` to avoid memory leaks.
 */
 char *infera_set_autoload_dir(const char *path);

/**
 * Retrieves the last error message set in the current thread.
 *
 * After an FFI function returns an error code, this function can be called
 * to get a more descriptive, human-readable error message.
 *
 * # Returns
 *
 * A pointer to a null-terminated C string containing the last error message.
 * Returns a null pointer if no error has occurred since the last call.
 * The caller **must not** free this pointer, as it is managed by a thread-local static variable.
 */
 const char *infera_last_error(void);

/**
 * Frees a heap-allocated C string that was returned by an Infera FFI function.
 *
 * This function should be used to free the memory for strings returned by functions
 * like `infera_get_model_info`, `infera_get_loaded_models`, and `infera_get_version`.
 *
 * # Safety
 *
 * The `ptr` must be a non-null pointer to a C string that was previously allocated
 * by Rust's `CString::into_raw`. Passing any other pointer (e.g., a string literal,
 * a pointer from a different allocator, or a null pointer) will result in undefined behavior.
 */
 void infera_free(char *ptr);

/**
 * Frees the data buffer within an `InferaInferenceResult`.
 *
 * This function must be called on every `InferaInferenceResult` returned from
 * `infera_predict` or `infera_predict_from_blob` to prevent memory leaks.
 *
 * # Safety
 *
 * * The `res` struct must be a value that was returned by an Infera prediction function.
 * * The `res.data` pointer must have been allocated by Rust's `Vec<f32>` and not yet freed.
 * * The `res.len` field must accurately represent the capacity of the allocated buffer.
 *
 * Calling this function on a manually-created struct or calling it more than once
 * on the same result will lead to undefined behavior.
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
