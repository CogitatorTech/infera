-- Tests loading an ONNX model from a public URL.
.echo on

-- 1. Load the infera extension
LOAD infera;

-- 2. Define the model name and a stable public URL for the test model
CREATE OR REPLACE MACRO model_name() AS 'remote_linear_model';
CREATE OR REPLACE MACRO model_url() AS 'https://github.com/CogitatorTech/infera/raw/refs/heads/main/tests/models/linear.onnx';

-- 3. Load the model from the URL
-- The first time this is run, the model will be downloaded.
-- Subsequent runs will be faster because they use the local cache.
SELECT load_onnx_model(model_name(), model_url()) AS loaded_from_url;

-- 4. Verify the model appears in the list of loaded models
SELECT list_models() AS models_after_load;
SELECT instr(list_models(), model_name()) > 0 AS model_is_listed;

-- 5. Run a prediction to confirm the remotely-loaded model is functional
-- The model implements: y = 2*x1 - 1*x2 + 0.5*x3 + 0.25
-- Using features (1.0, 2.0, 3.0), the expected result is y = 1.75
SELECT onnx_predict(model_name(), 1.0, 2.0, 3.0) AS prediction;
SELECT abs(onnx_predict(model_name(), 1.0, 2.0, 3.0) - 1.75) < 1e-5 AS prediction_is_correct;

-- 6. Unload the model to clean up the environment
SELECT unload_onnx_model(model_name()) AS unloaded;

-- 7. Verify the model has been removed from the list
SELECT list_models() AS models_after_unload;
SELECT instr(list_models(), model_name()) = 0 AS model_is_removed;

.echo off
