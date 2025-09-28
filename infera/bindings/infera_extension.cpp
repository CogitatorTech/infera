#define DUCKDB_EXTENSION_MAIN

#include "include/infera_extension.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/pragma_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rust.h"

namespace duckdb {

static std::string GetInferaError() {
  const char *err = infera_last_error();
  return err ? std::string(err) : std::string("unknown error");
}

static void PragmaAutoloadDir(ClientContext &context, const FunctionParameters &parameters) {
  if (parameters.values.empty() || parameters.values[0].IsNull()) {
    return;
  }
  std::string path = parameters.values[0].ToString();
  char *result_json_c = infera_autoload_dir(path.c_str());
  infera_free(result_json_c);
}

static void InferaVersion(DataChunk &args, ExpressionState &state, Vector &result) {
  char *info_json_c = infera_version();
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, info_json_c);
  ConstantVector::SetNull(result, false);
  infera_free(info_json_c);
}

static void LoadOnnxModel(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 2) {
    throw InvalidInputException("load_onnx_model(model_name, path) expects exactly 2 arguments");
  }
  auto &model_name_vec = args.data[0];
  auto &path_vec = args.data[1];
  if (args.size() == 0) { return; }
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
    throw InvalidInputException("Failed to load ONNX model '" + model_name_str + "': " + GetInferaError());
  }
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<bool>(result)[0] = success;
  ConstantVector::SetNull(result, false);
}

static void UnloadOnnxModel(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException("unload_onnx_model(model_name) expects exactly 1 argument");
  }
  auto &model_name_vec = args.data[0];
  if (args.size() == 0) { return; }
  auto model_name = model_name_vec.GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name.ToString();
  int rc = infera_unload_onnx_model(model_name_str.c_str());
  bool success = (rc == 0);
  if (!success) {
      throw InvalidInputException("Failed to unload ONNX model '" + model_name_str + "': " + GetInferaError());
  }
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<bool>(result)[0] = success;
  ConstantVector::SetNull(result, false);
}

static void ExtractFeatures(DataChunk &args, std::vector<float> &features) {
  const idx_t batch_size = args.size();
  const idx_t feature_count = args.ColumnCount() - 1;
  features.reserve(batch_size * feature_count);

  for (idx_t row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (idx_t col_idx = 1; col_idx <= feature_count; ++col_idx) {
      auto feature_val = args.data[col_idx].GetValue(row_idx);
      if (feature_val.IsNull()) {
        throw InvalidInputException("Feature values cannot be NULL");
      }
      float feature_float;
      switch (feature_val.type().id()) {
      case LogicalTypeId::FLOAT: feature_float = feature_val.GetValue<float>(); break;
      case LogicalTypeId::DOUBLE: feature_float = static_cast<float>(feature_val.GetValue<double>()); break;
      case LogicalTypeId::INTEGER: feature_float = static_cast<float>(feature_val.GetValue<int32_t>()); break;
      case LogicalTypeId::BIGINT: feature_float = static_cast<float>(feature_val.GetValue<int64_t>()); break;
      default: throw InvalidInputException("Unsupported feature type: " + feature_val.type().ToString());
      }
      features.push_back(feature_float);
    }
  }
}

static void OnnxPredict(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict(model_name, feature1, ...) requires at least 2 arguments");
  }
  if (args.size() == 0) { return; }
  auto &model_name_vec = args.data[0];
  auto model_name_val = model_name_vec.GetValue(0);
  if (model_name_val.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name_val.ToString();
  const idx_t batch_size = args.size();
  const idx_t feature_count = args.ColumnCount() - 1;

  std::vector<float> features;
  ExtractFeatures(args, features);

  InferaInferenceResult res = infera_run_inference(model_name_str.c_str(), features.data(), batch_size, feature_count);
  if (res.status != 0) {
    throw InvalidInputException("Inference failed for model '" + model_name_str + "': " + GetInferaError());
  }
  if (res.rows != batch_size || res.cols != 1) {
    std::string err_msg = StringUtil::Format("Model output shape mismatch. Expected (%d, 1), but got (%d, %d).", batch_size, res.rows, res.cols);
    infera_free_result(res);
    throw InvalidInputException(err_msg);
  }
  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<float>(result);
  for (idx_t i = 0; i < batch_size; i++) {
    result_data[i] = res.data[i];
  }
  infera_free_result(res);
}

static void InferaPredictBlob(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 2) {
    throw InvalidInputException("infera_predict_blob(model_name, input_blob) requires 2 arguments");
  }
  if (args.size() == 0) { return; }
  result.SetVectorType(VectorType::FLAT_VECTOR);
  for (idx_t i = 0; i < args.size(); i++) {
    auto model_name_val = args.data[0].GetValue(i);
    auto blob_val = args.data[1].GetValue(i);
    if (model_name_val.IsNull() || blob_val.IsNull()) {
      result.SetValue(i, Value());
      continue;
    }
    std::string model_name_str = model_name_val.ToString();
    string_t blob_str_t = blob_val.GetValueUnsafe<string_t>();
    auto blob_ptr = reinterpret_cast<const uint8_t *>(blob_str_t.GetDataUnsafe());
    auto blob_len = blob_str_t.GetSize();
    InferaInferenceResult res = infera_predict_blob(model_name_str.c_str(), blob_ptr, blob_len);
    if (res.status != 0) {
      infera_free_result(res);
      throw InvalidInputException("Inference failed for model '" + model_name_str + "': " + GetInferaError());
    }
    std::vector<Value> elems;
    elems.reserve(res.len);
    for (size_t j = 0; j < res.len; ++j) {
      elems.emplace_back(Value::FLOAT(res.data[j]));
    }
    result.SetValue(i, Value::LIST(std::move(elems)));
    infera_free_result(res);
  }
  result.Verify(args.size());
}

static void ListModels(DataChunk &args, ExpressionState &state, Vector &result) {
  char *models_json = infera_list_models();
  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, models_json);
  ConstantVector::SetNull(result, false);
  infera_free(models_json);
}

static void ModelInfo(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException("model_info(model_name) expects exactly 1 argument");
  }
  if (args.size() == 0) { return; }
  auto model_name = args.data[0].GetValue(0);
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

static void OnnxPredictMulti(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() < 2) {
    throw InvalidInputException("onnx_predict_multi(model_name, feature1, ...) requires at least 2 arguments");
  }
  if (args.size() == 0) { return; }
  auto &model_name_vec = args.data[0];
  auto model_name_val = model_name_vec.GetValue(0);
  if (model_name_val.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name_val.ToString();
  const idx_t batch_size = args.size();
  const idx_t feature_count = args.ColumnCount() - 1;

  std::vector<float> features;
  ExtractFeatures(args, features);

  InferaInferenceResult res = infera_run_inference(model_name_str.c_str(), features.data(), batch_size, feature_count);
  if (res.status != 0) {
    infera_free_result(res);
    throw InvalidInputException("Inference failed for model '" + model_name_str + "': " + GetInferaError());
  }
  if (res.rows != batch_size) {
    std::string err_msg = StringUtil::Format("Model output row count mismatch. Expected %d, but got %d.", batch_size, res.rows);
    infera_free_result(res);
    throw InvalidInputException(err_msg);
  }
  result.SetVectorType(VectorType::FLAT_VECTOR);
  auto result_data = FlatVector::GetData<string_t>(result);
  const size_t output_cols = res.cols;
  for (idx_t row_idx = 0; row_idx < batch_size; row_idx++) {
    std::string json_result = "[";
    for (size_t col_idx = 0; col_idx < output_cols; col_idx++) {
      if (col_idx > 0) { json_result += ","; }
      json_result += std::to_string(res.data[row_idx * output_cols + col_idx]);
    }
    json_result += "]";
    result_data[row_idx] = StringVector::AddString(result, json_result);
  }
  infera_free_result(res);
}

static void ModelMetadataFunc(DataChunk &args, ExpressionState &state, Vector &result) {
  if (args.ColumnCount() != 1) {
    throw InvalidInputException("model_metadata(model_name) expects exactly 1 argument");
  }
  if (args.size() == 0) { return; }
  auto model_name = args.data[0].GetValue(0);
  if (model_name.IsNull()) {
    throw InvalidInputException("Model name cannot be NULL");
  }
  std::string model_name_str = model_name.ToString();
  char *json_meta = infera_get_model_metadata(model_name_str.c_str());

  result.SetVectorType(VectorType::CONSTANT_VECTOR);
  ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, json_meta);
  ConstantVector::SetNull(result, false);
  infera_free(json_meta);
}

static void LoadInternal(ExtensionLoader &loader) {
  ScalarFunction load_onnx_model_func("load_onnx_model", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN, LoadOnnxModel);
  loader.RegisterFunction(load_onnx_model_func);

  ScalarFunction unload_onnx_model_func("unload_onnx_model", {LogicalType::VARCHAR}, LogicalType::BOOLEAN, UnloadOnnxModel);
  loader.RegisterFunction(unload_onnx_model_func);

  // Removed deprecated LogicalType::VARARG usage. Register multiple arities instead.
  const idx_t MAX_FEATURES = 63; // features (total args = 1 + features)
  for (idx_t feature_count = 1; feature_count <= MAX_FEATURES; feature_count++) {
    vector<LogicalType> arg_types;
    arg_types.reserve(feature_count + 1);
    arg_types.push_back(LogicalType::VARCHAR); // model name
    for (idx_t i = 0; i < feature_count; i++) {
      arg_types.push_back(LogicalType::FLOAT); // DuckDB will auto-cast other numerics
    }
    loader.RegisterFunction(ScalarFunction("onnx_predict", arg_types, LogicalType::FLOAT, OnnxPredict));
    loader.RegisterFunction(ScalarFunction("onnx_predict_multi", arg_types, LogicalType::VARCHAR, OnnxPredictMulti));
  }

  ScalarFunction infera_predict_blob_func("infera_predict_blob", {LogicalType::VARCHAR, LogicalType::BLOB}, LogicalType::LIST(LogicalType::FLOAT), InferaPredictBlob);
  loader.RegisterFunction(infera_predict_blob_func);

  ScalarFunction list_models_func("list_models", {}, LogicalType::VARCHAR, ListModels);
  loader.RegisterFunction(list_models_func);

  ScalarFunction model_info_func("model_info", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ModelInfo);
  loader.RegisterFunction(model_info_func);

  ScalarFunction model_metadata_func("model_metadata", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ModelMetadataFunc);
  loader.RegisterFunction(model_metadata_func);

  ScalarFunction infera_version_func("infera_version", {}, LogicalType::VARCHAR, InferaVersion);
  loader.RegisterFunction(infera_version_func);

  auto autoload_pragma = PragmaFunction::PragmaCall("infera_autoload_dir", PragmaAutoloadDir,
                                                    {LogicalType::VARCHAR}, LogicalType::INVALID);
  loader.RegisterFunction(autoload_pragma);
}

void InferaExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string InferaExtension::Name() { return "infera"; }
std::string InferaExtension::Version() const { return "v0.1.0"; }

} // namespace duckdb

extern "C" {
DUCKDB_EXTENSION_API void infera_init(duckdb::DatabaseInstance &db) {
  duckdb::ExtensionLoader loader(db, "infera");
  duckdb::LoadInternal(loader);
}
}
