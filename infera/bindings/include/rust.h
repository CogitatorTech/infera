#ifndef INFERA_H
#define INFERA_H

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float *data;
  size_t len;
  size_t rows;
  size_t cols;
  int32_t status;
} InferaInferenceResult;

void infera_free(char *ptr);
void infera_free_result(InferaInferenceResult result);
const char *infera_last_error(void);
char *infera_get_loaded_models(void);
char *infera_get_model_info(const char *model_name);
char *infera_get_version(void);
char *infera_set_autoload_dir(const char *path);
int32_t infera_load_model(const char *name, const char *path);
int32_t infera_unload_model(const char *name);
InferaInferenceResult infera_predict(const char *model_name, const float *data,
                                     size_t rows, size_t cols);
InferaInferenceResult infera_predict_from_blob(const char *model_name,
                                               const uint8_t *blob_data,
                                               size_t blob_len);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* INFERA_H */
