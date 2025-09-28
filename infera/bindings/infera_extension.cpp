#define DUCKDB_EXTENSION_MAIN

#include "infera_extension.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Include the generated Rust header
#include "rust.h"

namespace duckdb {

// Utility function to get and format error messages
static std::string GetInferaError() {
  const char *err = infera_last_error();
  return err ? std::string(err)
             : std::string("unknown error"); // Should not happen
}

// Model loading function
static void LoadOnnxModel(DataChunk &args, ExpressionState &state,
                          Vector &result) {
  if (args.ColumnCount() != 2) {
    throw InvalidInputException(
        "load_onnx_model(model_name, path) expects exactly 2 arguments");
  }

  auto &model_name_vec = args.data[0];
  auto &path_vec = args.data[1];

  if (args.size() == 0) {
    return;
  }

  // Process first row only
  auto model_name = model_name_vec.GetValue(0);
  auto path = path_vec.GetValue(0);

  if (model_name.IsNull() || path.IsNull()) {
    throw InvalidInputException("Model name and path cannot be NULL");
  }

  std::string model_name_str = model_name.ToString();
  std::string path_str = path.ToString();

  if (model_name_str.empty()) {
    throw InvalidInputException("Model name cannot be empty");
  }

  int rc = infera_load_onnx_model(model_name_str.c_str(), path_str.c_str());
  bool success = rc == 0;

  if (!success) {
    throw InvalidInputException("Failed to load ONNX model '" + model_name_str +
                                "': " + GetInferaError());
  }

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<bool>(result)[0] = success;
  ConstantVector::SetNull(result, false);
}

// Model unloading function
static void UnloadOnnxModel(DataChunk &args, ExpressionState &state,
                            Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException(
        "unload_onnx_model(model_name) expects exactly 1 argument");
  }

  auto &model_name_vec = args.data[0];
  if (args.size() == 0) {
    return;
  }

  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }

  std::string model_name_str = model_name.ToString();
  int rc = infera_unload_onnx_model(model_name_str.c_str());
  bool success = (rc == 0);

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<bool>(result)[0] = success;
  ConstantVector::SetNull(result, false);
}

// Single-output ONNX prediction function using batching
static void OnnxPredict(DataChunk &args, ExpressionState &state,
                        Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict(model_name, feature1, ...) "
                                "requires at least 2 arguments");
  }

  if (args.size() == 0) {
    return;
  }

  auto &model_name_vec = args.data[0];
  auto model_name_val = model_name_vec.GetValue(0);
  if (model_name_val.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name_val.ToString();

  // Collect all features for the batch
  const idx_t batch_size = args.size();
  const idx_t feature_count = args.ColumnCount() - 1;
  std::vector<float> features;
  features.reserve(batch_size * feature_count);

  for (idx_t row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (idx_t col_idx = 1; col_idx <= feature_count; ++col_idx) {
      auto feature_val = args.data[col_idx].GetValue(row_idx);
      if (feature_val.IsNull()) {
        throw InvalidInputException("Feature values cannot be NULL");
      }

      float feature_float;
      switch (feature_val.type().id()) {
      case LogicalTypeId::FLOAT:
        feature_float = feature_val.GetValue<float>();
        break;
      case LogicalTypeId::DOUBLE:
        feature_float = static_cast<float>(feature_val.GetValue<double>());
        break;
      case LogicalTypeId::INTEGER:
        feature_float = static_cast<float>(feature_val.GetValue<int32_t>());
        break;
      case LogicalTypeId::BIGINT:
        feature_float = static_cast<float>(feature_val.GetValue<int64_t>());
        break;
      default:
        throw InvalidInputException("Unsupported feature type: " +
                                    feature_val.type().ToString());
      }
      features.push_back(feature_float);
    }
  }

  // Run inference on the entire batch
  InferaInferenceResult res = infera_run_inference(
      model_name_str.c_str(), features.data(), batch_size, feature_count);

  if (res.status != 0) {
    throw InvalidInputException("Inference failed for model '" +
                                model_name_str + "': " + GetInferaError());
  }

  // Validate result shape for this single-output function
  if (res.rows != batch_size || res.cols != 1) {
    std::string err_msg = StringUtil::Format(
        "Model output shape mismatch. Expected (%d, 1), but got (%d, %d).",
        batch_size, res.rows, res.cols);
    infera_free_result(res);
    throw InvalidInputException(err_msg);
  }

  // Populate the result vector
  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<float>(result);
  for (idx_t i = 0; i < batch_size; i++) {
    result_data[i] = res.data[i];
  }

  infera_free_result(res);
}

// List all loaded models (returns a JSON array)
static void ListModels(DataChunk &args, ExpressionState &state,
                       Vector &result) {
  char *models_json = infera_list_models();

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, models_json);
  ConstantVector::SetNull(result, false);

  infera_free(models_json);
}

// Get model information
static void ModelInfo(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException(
        "model_info(model_name) expects exactly 1 argument");
  }

  auto &model_name_vec = args.data[0];
  if (args.size() == 0) {
    return;
  }

  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }

  std::string model_name_str = model_name.ToString();
  char *info_json = infera_model_info(model_name_str.c_str());

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, info_json);
  ConstantVector::SetNull(result, false);

  infera_free(info_json);
}

// Multi-output prediction function (returns a JSON array)
static void OnnxPredictMulti(DataChunk &args, ExpressionState &state,
                             Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict_multi(model_name, feature1, ...) "
                                "requires at least 2 arguments");
  }

  if (args.size() == 0) {
    return;
  }

  auto &model_name_vec = args.data[0];
  auto model_name_val = model_name_vec.GetValue(0);
  if (model_name_val.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name_val.ToString();

  // Collect all features for the batch
  const idx_t batch_size = args.size();
  const idx_t feature_count = args.ColumnCount() - 1;
  std::vector<float> features;
  features.reserve(batch_size * feature_count);

  for (idx_t row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (idx_t col_idx = 1; col_idx <= feature_count; ++col_idx) {
      auto feature_val = args.data[col_idx].GetValue(row_idx);
      if (feature_val.IsNull()) {
        throw InvalidInputException("Feature values cannot be NULL");
      }

      float feature_float;
      switch (feature_val.type().id()) {
      case LogicalTypeId::FLOAT:
        feature_float = feature_val.GetValue<float>();
        break;
      case LogicalTypeId::DOUBLE:
        feature_float = static_cast<float>(feature_val.GetValue<double>());
        break;
      case LogicalTypeId::INTEGER:
        feature_float = static_cast<float>(feature_val.GetValue<int32_t>());
        break;
      case LogicalTypeId::BIGINT:
        feature_float = static_cast<float>(feature_val.GetValue<int64_t>());
        break;
      default:
        throw InvalidInputException("Unsupported feature type: " +
                                    feature_val.type().ToString());
      }
      features.push_back(feature_float);
    }
  }

  // Run inference on the entire batch
  InferaInferenceResult res = infera_run_inference(
      model_name_str.c_str(), features.data(), batch_size, feature_count);

  if (res.status != 0) {
    throw InvalidInputException("Inference failed for model '" +
                                model_name_str + "': " + GetInferaError());
  }

  // Validate result shape
  if (res.rows != batch_size) {
    std::string err_msg = StringUtil::Format(
        "Model output row count mismatch. Expected %d, but got %d.",
        batch_size, res.rows);
    infera_free_result(res);
    throw InvalidInputException(err_msg);
  }

  // Populate the result vector with JSON strings
  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<string_t>(result);
  const size_t output_cols = res.cols;

  for (idx_t row_idx = 0; row_idx < batch_size; row_idx++) {
    std::string json_result = "[";
    for (size_t col_idx = 0; col_idx < output_cols; col_idx++) {
      if (col_idx > 0) {
        json_result += ",";
      }
      json_result += std::to_string(res.data[row_idx * output_cols + col_idx]);
    }
    json_result += "]";
    result_data[row_idx] = StringVector::AddString(result, json_result);
  }

  infera_free_result(res);
}

// Model metadata function returning JSON
static void ModelMetadataFunc(DataChunk &args, ExpressionState &state,
                              Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException(
        "model_metadata(model_name) expects exactly 1 argument");
  }
  auto &model_name_vec = args.data[0];
  if (args.size() == 0) {
    return;
  }
  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name.ToString();

  ModelMetadata meta = infera_get_model_metadata(model_name_str.c_str());
  std::string json;
  if (meta.input_shape_len == 0 && meta.output_shape_len == 0) {
    // On error, fetch last error message
    json = std::string("{\"error\": \"") + GetInferaError() + "\"}";
  } else {
    json = "{";
    json += "\"name\":\"" + model_name_str + "\",";
    json += "\"input_shape\":[";
    for (size_t i = 0; i < meta.input_shape_len; i++) {
      if (i > 0)
        json += ",";
      json += std::to_string(meta.input_shape[i]);
    }
    json += "],";
    json += "\"output_shape\":[";
    for (size_t i = 0; i < meta.output_shape_len; i++) {
      if (i > 0)
        json += ",";
      json += std::to_string(meta.output_shape[i]);
    }
    json += "],";
    json += "\"input_count\":" + std::to_string(meta.input_count) + ",";
    json += "\"output_count\":" + std::to_string(meta.output_count);
    json += "}";
  }
  infera_free_metadata(meta);

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, json);
  ConstantVector::SetNull(result, false);
}

// Internal function to register all functions
static void LoadInternal(ExtensionLoader &loader) {
  ScalarFunction load_onnx_model_func(
      "load_onnx_model", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::BOOLEAN, LoadOnnxModel);
  loader.RegisterFunction(load_onnx_model_func);

  ScalarFunction unload_onnx_model_func("unload_onnx_model",
                                        {LogicalType::VARCHAR},
                                        LogicalType::BOOLEAN, UnloadOnnxModel);
  loader.RegisterFunction(unload_onnx_model_func);

  // Register inference functions with variable number of arguments
  ScalarFunctionSet onnx_predict_set("onnx_predict");
  for (idx_t param_count = 2; param_count <= 10; param_count++) {
    vector<LogicalType> arguments;
    arguments.push_back(LogicalType::VARCHAR); // model_name
    for (idx_t i = 1; i < param_count; i++) {
      arguments.push_back(LogicalType::FLOAT); // features
    }
    onnx_predict_set.AddFunction(
        ScalarFunction(arguments, LogicalType::FLOAT, OnnxPredict));
  }
  loader.RegisterFunction(onnx_predict_set);

  // Multi-output prediction function
  ScalarFunctionSet onnx_predict_multi_set("onnx_predict_multi");
  for (idx_t param_count = 2; param_count <= 10; param_count++) {
    vector<LogicalType> arguments;
    arguments.push_back(LogicalType::VARCHAR); // model_name
    for (idx_t i = 1; i < param_count; i++) {
      arguments.push_back(LogicalType::FLOAT); // features
    }
    onnx_predict_multi_set.AddFunction(
        ScalarFunction(arguments, LogicalType::VARCHAR, OnnxPredictMulti));
  }
  loader.RegisterFunction(onnx_predict_multi_set);

  // Utility functions
  ScalarFunction list_models_func("list_models", {}, LogicalType::VARCHAR,
                                  ListModels);
  loader.RegisterFunction(list_models_func);

  ScalarFunction model_info_func("model_info", {LogicalType::VARCHAR},
                                 LogicalType::VARCHAR, ModelInfo);
  loader.RegisterFunction(model_info_func);

  // Register model_metadata JSON function
  ScalarFunction model_metadata_func("model_metadata", {LogicalType::VARCHAR},
                                     LogicalType::VARCHAR, ModelMetadataFunc);
  loader.RegisterFunction(model_metadata_func);
}

void InferaExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }

std::string InferaExtension::Name() { return "infera"; }
std::string InferaExtension::Version() const { return "v0.1.0"; }

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void infera_init(duckdb::DatabaseInstance &db) {
  duckdb::ExtensionLoader loader(db, "infera");
  LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *infera_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}
