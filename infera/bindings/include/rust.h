#ifndef INFERA_H
#define INFERA_H

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic types and structures
typedef struct {
  float *data;
  size_t len;
  size_t rows;
  size_t cols;
  int32_t status;
} InferaInferenceResult;

// Basic functions
void infera_free(char *ptr);
const char *infera_last_error(void);

// Version and autoload functions
char *infera_version(void);
char *infera_autoload_dir(const char *path);

// Model management functions
int32_t infera_load_onnx_model(const char *name, const char *path);
int32_t infera_unload_onnx_model(const char *name);

// Inference functions
InferaInferenceResult infera_run_inference(const char *model_name,
                                           const float *data, size_t rows,
                                           size_t cols);
InferaInferenceResult infera_predict_blob(const char *model_name,
                                          const uint8_t *blob_data,
                                          size_t blob_len);

// Utility functions
char *infera_list_models(void);
char *infera_model_info(const char *model_name);
char *infera_get_model_metadata(const char *model_name); // changed return type

// Memory cleanup functions
void infera_free_result(InferaInferenceResult result);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* INFERA_H */
