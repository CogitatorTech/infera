-- Tests the BLOB input API
.echo on

-- 1. Load the extension
LOAD infera;

-- 2. Load a model that expects a tensor input (e.g., MobileNetV3)
SELECT load_onnx_model(
    'mobilenet',
    'https://huggingface.co/onnxmodelzoo/tf_mobilenetv3_small_075_Opset17/resolve/main/tf_mobilenetv3_small_075_Opset17.onnx'
);

-- 3. Create a table with a BLOB column
CREATE TABLE image_data(id INTEGER, image_tensor BLOB);

-- 4. Insert a dummy BLOB to test error handling
INSERT INTO image_data VALUES (1, 'dummy_bytes');

-- 5. Call the function with an incorrectly sized BLOB
-- This correctly fails with an "Invalid BLOB size" error.
SELECT infera_predict_blob('mobilenet', image_tensor) FROM image_data;

-- 6. Test with a correctly sized but zero-filled BLOB
-- This is the most efficient way to generate a zero-filled BLOB in SQL.
-- This call will now succeed.
WITH const AS (
  -- 1*224*224*3 = 150528 floats. Each float is 4 bytes. Total size = 602112 bytes.
  SELECT CAST(REPEAT(CHR(0), 602112) AS BLOB) AS zero_blob
)
SELECT len(infera_predict_blob('mobilenet', zero_blob)) as output_length
FROM const;

-- 7. Clean up
DROP TABLE image_data;
SELECT unload_onnx_model('mobilenet');

.echo off
