-- Tests the autoload and version functions
.echo on

LOAD infera;

-- 1. Test the version function
-- This should return a JSON object with version info.
SELECT infera_version();

-- 2. Test the autoload PRAGMA
-- Create a temporary directory for models
.shell mkdir -p tests/temp_models
-- Copy the test model into it
.shell cp tests/models/linear.onnx tests/temp_models/

-- Run the PRAGMA to load all models from the directory
-- Note that `infera_autoload_dir` is a function, not a configuration parameter.
PRAGMA infera_autoload_dir('tests/temp_models');

-- Verify the 'linear' model was loaded automatically
SELECT list_models();
SELECT instr(list_models(), 'linear') > 0 AS model_is_listed;

-- Run a prediction to confirm it's functional
SELECT abs(onnx_predict('linear', 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS correct_prediction;

-- Clean up
SELECT unload_onnx_model('linear');
.shell rm -rf tests/temp_models

.echo off
