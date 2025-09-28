-- Tests advanced features like remote model loading and BLOB inputs.
.echo on
LOAD infera;

-- =============================================================================
-- Test 1: Remote Model Loading
-- =============================================================================
SELECT '--- Testing Remote Model Loading ---';

-- Define macros for the model name and URL for clarity.
CREATE OR REPLACE MACRO model_name() AS 'remote_linear_model';
CREATE OR REPLACE MACRO model_url() AS 'https://github.com/CogitatorTech/infera/raw/refs/heads/main/tests/models/linear.onnx';

-- Load the model from the URL.
SELECT infera_load_model(model_name(), model_url()) AS loaded_from_url;

-- Verify the model appears in the list.
SELECT instr(infera_get_loaded_models(), model_name()) > 0 AS model_is_listed;

-- Run a prediction to confirm it's functional (y = 1.75 for these features).
SELECT abs(infera_predict(model_name(), 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS prediction_is_correct;

-- Unload the model to clean up.
SELECT infera_unload_model(model_name()) AS unloaded;
SELECT instr(infera_get_loaded_models(), model_name()) = 0 AS model_is_removed;


-- =============================================================================
-- Test 2: BLOB Input Prediction
-- =============================================================================
SELECT '--- Testing BLOB Input Prediction ---';

-- Load a model that expects a large tensor input.
SELECT infera_load_model(
    'mobilenet',
    'https://huggingface.co/onnxmodelzoo/tf_mobilenetv3_small_075_Opset17/resolve/main/tf_mobilenetv3_small_075_Opset17.onnx'
);

-- Test error handling with an incorrectly sized BLOB.
-- This is expected to fail.
SELECT infera_predict_from_blob('mobilenet', 'dummy_bytes');

-- Test with a correctly sized, zero-filled BLOB.
-- Model input is 1*224*224*3 floats. A float is 4 bytes. Total size = 602112 bytes.
WITH const AS (
  SELECT CAST(REPEAT(CHR(0), 602112) AS BLOB) AS zero_blob
)
SELECT len(infera_predict_from_blob('mobilenet', zero_blob)) as output_length
FROM const;

-- Clean up.
SELECT infera_unload_model('mobilenet');

.echo off
