-- Tests the core, end-to-end functionality of the extension.
.echo on
LOAD infera;

-- =============================================================================
-- Test 1: Version and Initial State
-- =============================================================================
SELECT '--- Testing Version and Initial State ---';

-- Check that the version function returns a valid JSON.
SELECT infera_get_version();

-- Check that no models are loaded initially.
SELECT infera_get_loaded_models() AS initial_models;


-- =============================================================================
-- Test 2: Local Model Roundtrip (Load -> Info -> Predict -> Unload)
-- =============================================================================
SELECT '--- Testing Local Model Roundtrip ---';

-- Load the simple linear model.
SELECT infera_load_model('linear', 'tests/models/linear.onnx') AS loaded;

-- Verify the model appears in the list.
SELECT instr(infera_get_loaded_models(), 'linear') > 0 AS after_load;

-- Retrieve model info and check for expected input shape.
SELECT infera_get_model_info('linear') AS metadata;
SELECT position('"input_shape":[1,3]' IN infera_get_model_info('linear')) > 0 AS metadata_ok;

-- Run deterministic predictions. Model is y = 2*x1 - 1*x2 + 0.5*x3 + 0.25
-- For (1.0, 2.0, 3.0), expected y = 1.75
SELECT infera_predict('linear', 1.0, 2.0, 3.0) AS single_prediction;
SELECT abs(infera_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS single_predict_ok;

-- Test the multi-output prediction function.
SELECT infera_predict_multi('linear', 1.0, 2.0, 3.0) as multi_prediction;
SELECT instr(infera_predict_multi('linear', 1.0, 2.0, 3.0), '1.75') > 0 AS multi_predict_ok;

-- Unload the model and confirm its removal.
SELECT infera_unload_model('linear') AS unloaded;
SELECT infera_get_loaded_models() AS after_unload;


-- =============================================================================
-- Test 3: Autoload Directory
-- =============================================================================
SELECT '--- Testing Autoload Directory ---';

-- Create a temporary directory and copy the model into it.
.shell mkdir -p tests/temp_models
.shell cp tests/models/linear.onnx tests/temp_models/

-- Run the autoload function.
SELECT infera_set_autoload_dir('tests/temp_models');

-- Verify the model was loaded automatically.
SELECT instr(infera_get_loaded_models(), 'linear') > 0 AS autoloaded_model_is_listed;

-- Run a prediction to confirm it's functional.
SELECT abs(infera_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS autoload_predict_ok;

-- Clean up.
SELECT infera_unload_model('linear');
.shell rm -rf tests/temp_models


.echo off
