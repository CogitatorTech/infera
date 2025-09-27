#define DUCKDB_EXTENSION_MAIN

#include "infera_extension.hpp"
#include "duckdb/common/exception.hpp"
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
             : std::string("unknown error"); // This should never happen
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
    throw InvalidInputException("load_onnx_model requires at least one row");
  }

  // Process first row only (model name and path should be same for all rows)
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

  // Return success status for all rows (even if only one failed)
  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<bool>(result);
  for (idx_t i = 0; i < args.size(); i++) {
    result_data[i] = success;
  }
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
    throw InvalidInputException("unload_onnx_model requires at least one row");
  }

  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }

  std::string model_name_str = model_name.ToString();
  int rc = infera_unload_onnx_model(model_name_str.c_str());
  bool success = (rc == 0);

  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<bool>(result);
  for (idx_t i = 0; i < args.size(); i++) {
    result_data[i] = success;
  }
}

// ONNX prediction function with variable number of arguments
static void OnnxPredict(DataChunk &args, ExpressionState &state,
                        Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict(model_name, feature1, ...) "
                                "requires at least 2 arguments");
  }

  auto &model_name_vec = args.data[0];

  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<float>(result);

  for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
    auto model_name = model_name_vec.GetValue(row_idx);

    if (model_name.IsNull()) {
      throw InvalidInputException("Model name cannot be NULL");
    }

    std::string model_name_str = model_name.ToString();

    // Collect feature values
    std::vector<float> features;
    features.reserve(args.ColumnCount() - 1);

    for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
      auto feature_val = args.data[col_idx].GetValue(row_idx);
      if (feature_val.IsNull()) {
        throw InvalidInputException("Feature values cannot be NULL");
      }

      // Convert to float based on type
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

    // Run inference
    InferaInferenceResult inference_result =
        infera_run_inference(model_name_str.c_str(), features.data(),
                             1, // single row
                             features.size());

    if (inference_result.status != 0) {
      throw InvalidInputException("Inference failed for model '" +
                                  model_name_str + "': " + GetInferaError());
    }

    if (inference_result.len == 0 || inference_result.data == nullptr) {
      throw InvalidInputException("No prediction returned from model '" +
                                  model_name_str + "'");
    }

    // Take the first prediction value
    result_data[row_idx] = inference_result.data[0];

    // Clean up the result
    infera_free_result(inference_result);
  }
}

// List all loaded models (returns a JSON array)
static void ListModels(DataChunk &args, ExpressionState &state,
                       Vector &result) {
  char *models_json = infera_list_models();
  result.SetValue(0, Value(models_json));
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
    throw InvalidInputException("model_info requires at least one row");
  }

  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }

  std::string model_name_str = model_name.ToString();
  char *info_json = infera_model_info(model_name_str.c_str());

  result.SetValue(0, Value(info_json));
  infera_free(info_json);
}

// Multi-output prediction function (returns a JSON array)
static void OnnxPredictMulti(DataChunk &args, ExpressionState &state,
                             Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict_multi(model_name, feature1, ...) "
                                "requires at least 2 arguments");
  }

  auto &model_name_vec = args.data[0];

  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<string_t>(result);

  for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
    auto model_name = model_name_vec.GetValue(row_idx);

    if (model_name.IsNull()) {
      throw InvalidInputException("Model name cannot be NULL");
    }

    std::string model_name_str = model_name.ToString();

    // Collect feature values
    std::vector<float> features;
    features.reserve(args.ColumnCount() - 1);

    for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
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

    // Run inference
    InferaInferenceResult inference_result =
        infera_run_inference(model_name_str.c_str(), features.data(),
                             1, // single row
                             features.size());

    if (inference_result.status != 0) {
      throw InvalidInputException("Inference failed for model '" +
                                  model_name_str + "': " + GetInferaError());
    }

    if (inference_result.len == 0 || inference_result.data == nullptr) {
      throw InvalidInputException("No prediction returned from model '" +
                                  model_name_str + "'");
    }

    // Convert all outputs to JSON array
    std::string json_result = "[";
    for (size_t i = 0; i < inference_result.len; i++) {
      if (i > 0)
        json_result += ",";
      json_result += std::to_string(inference_result.data[i]);
    }
    json_result += "]";

    result_data[row_idx] = StringVector::AddString(result, json_result);

    // Clean up the result
    infera_free_result(inference_result);
  }
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
    throw InvalidInputException("model_metadata requires at least one row");
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

  result.SetValue(0, Value(json));
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
//std::string InferaExtension::Description() const {
//  return "Infera extension allows running ML models on data stored in DuckDB";
//}

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
