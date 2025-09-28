-- Tiny ONNX model load/predict/unload roundtrip test
.echo on
LOAD infera;

-- Load sample model
SELECT infera_load_model('linear', 'tests/models/linear.onnx') AS loaded;

-- Verify model appears in list
SELECT infera_get_loaded_models() AS after_load;

-- Retrieve metadata and validate it contains expected input shape
SELECT infera_get_model_info('linear') AS metadata;
SELECT position('"input_shape":[1,3]' IN infera_get_model_info('linear')) > 0 AS metadata_has_input_shape;

-- Run deterministic prediction; model implements: y = 2*x1 -1*x2 + 0.5*x3 + 0.25
-- Using features (1.0, 2.0, 3.0) expected y = 1.75
SELECT infera_predict('linear', 1.0, 2.0, 3.0) AS prediction;
SELECT abs(infera_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS correct_prediction;

-- Unload model and confirm removal
SELECT infera_unload_model('linear') AS unloaded;
SELECT infera_get_loaded_models() AS after_unload;
.echo off
