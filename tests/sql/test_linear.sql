-- Tiny ONNX model load/predict/unload roundtrip test
.echo on
LOAD infera;

-- Load sample model
SELECT load_onnx_model('linear', 'tests/models/linear.onnx') AS loaded;

-- Verify model appears in list
SELECT list_models() AS after_load;

-- Retrieve metadata and validate it contains expected input shape
SELECT model_metadata('linear') AS metadata;
SELECT position('"input_shape":[1,3]' IN model_metadata('linear')) > 0 AS metadata_has_input_shape;

-- Run deterministic prediction; model implements: y = 2*x1 -1*x2 + 0.5*x3 + 0.25
-- Using features (1.0, 2.0, 3.0) expected y = 1.75
SELECT onnx_predict('linear', 1.0, 2.0, 3.0) AS prediction;
SELECT abs(onnx_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS correct_prediction;

-- Unload model and confirm removal
SELECT unload_onnx_model('linear') AS unloaded;
SELECT list_models() AS after_unload;
.echo off
