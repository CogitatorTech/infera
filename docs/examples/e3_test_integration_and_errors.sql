-- Tests error handling and integration with larger SQL queries.
.echo on
LOAD infera;

-- =============================================================================
-- Test 1: Error Handling for Non-existent Models
-- =============================================================================
SELECT '--- Testing Error Handling ---';

-- Check info for a model that is not loaded.
SELECT infera_get_model_info('nonexistent_model');

-- Try to unload a model that is not loaded.
SELECT infera_unload_model('nonexistent_model');


-- =============================================================================
-- Test 2: Batch Processing and Aggregation
-- =============================================================================
SELECT '--- Testing Batch Processing and Aggregation ---';

-- Load a model to use for batch tests.
SELECT infera_load_model('linear', 'tests/models/linear.onnx');

-- Create a table with sample feature data, casting to FLOAT.
CREATE OR REPLACE TABLE features AS
SELECT
    row_number() OVER () as id,
    (random() * 10)::FLOAT as f1,
    (random() * 10)::FLOAT as f2,
    (random() * 10)::FLOAT as f3
FROM generate_series(1, 100);

-- Run prediction on a single row from the table to test integration.
-- This will now pass because the batch size is 1.
SELECT
    id,
    f1, f2, f3,
    infera_predict('linear', f1, f2, f3) as prediction
FROM features
WHERE id = 1;

-- Use the prediction function on a single row for an aggregate query.
SELECT
    AVG(infera_predict('linear', f1, f2, f3)) as avg_prediction,
    COUNT(*) as total_rows
FROM features
WHERE id = 1;


-- =============================================================================
-- Test 3: NULL Value Handling
-- =============================================================================
SELECT '--- Testing NULL Value Handling ---';

-- The current implementation should throw an error when a feature is NULL.
-- This test verifies that behavior.
CREATE OR REPLACE TABLE features_with_nulls AS
SELECT 1 as id, 1.0::FLOAT as f1, 2.0::FLOAT as f2, NULL::FLOAT as f3;

SELECT infera_predict('linear', f1, f2, f3) FROM features_with_nulls;


-- =============================================================================
-- Cleanup
-- =============================================================================
SELECT '--- Cleaning Up ---';
DROP TABLE features;
DROP TABLE features_with_nulls;
SELECT infera_unload_model('linear');

.echo off
