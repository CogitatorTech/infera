-- More comprehensive tests for Infera DuckDB extension

.echo on

-- =============================================================================
-- Basic functionality tests
-- =============================================================================

-- Test 1: Basic extension loading (verify list_models returns JSON)
SELECT list_models() as models_initial;

-- Test 2: List models (should be empty initially)
SELECT list_models() as empty_models;

-- =============================================================================
-- Model management tests
-- =============================================================================

-- Test 3: Load a simple ONNX model (example paths - adjust for your models)
-- SELECT load_onnx_model('iris_classifier', '/path/to/iris_model.onnx') as load_success;

-- Test 4: List loaded models
-- SELECT list_models() as loaded_models;

-- Test 5: Get model information
-- SELECT model_info('iris_classifier') as model_details;

-- =============================================================================
-- Batch inference tests with sample data
-- =============================================================================

-- Create sample feature data for testing
CREATE TABLE sample_features AS
SELECT
    row_number() OVER () as id,
    5.1 + (random() * 2.0) as sepal_length,
    3.5 + (random() * 1.0) as sepal_width,
    1.4 + (random() * 5.0) as petal_length,
    0.2 + (random() * 2.0) as petal_width
FROM generate_series(1, 100);

-- Test 6: Single-output prediction (uncomment when model is loaded)
-- SELECT
--     id,
--     sepal_length,
--     sepal_width,
--     petal_length,
--     petal_width,
--     onnx_predict('iris_classifier', sepal_length, sepal_width, petal_length, petal_width) as prediction
-- FROM sample_features
-- LIMIT 10;

-- Test 7: Multi-output prediction (for models with multiple outputs)
-- SELECT
--     id,
--     onnx_predict_multi('iris_classifier', sepal_length, sepal_width, petal_length, petal_width) as predictions_json
-- FROM sample_features
-- LIMIT 5;

-- =============================================================================
-- Aggregation and analytics tests
-- =============================================================================

-- Test 8: Aggregate predictions (uncomment when model is loaded)
-- SELECT
--     CASE
--         WHEN sepal_length < 5.5 THEN 'small'
--         WHEN sepal_length < 6.5 THEN 'medium'
--         ELSE 'large'
--     END as size_category,
--     COUNT(*) as count,
--     AVG(onnx_predict('iris_classifier', sepal_length, sepal_width, petal_length, petal_width)) as avg_prediction,
--     MIN(onnx_predict('iris_classifier', sepal_length, sepal_width, petal_length, petal_width)) as min_prediction,
--     MAX(onnx_predict('iris_classifier', sepal_length, sepal_width, petal_length, petal_width)) as max_prediction
-- FROM sample_features
-- GROUP BY size_category;

-- =============================================================================
-- Performance and batch processing tests
-- =============================================================================

-- Test 9: Large batch inference performance
CREATE TABLE large_batch AS
SELECT
    row_number() OVER () as id,
    5.0 + (random() * 3.0) as feature1,
    3.0 + (random() * 2.0) as feature2,
    2.0 + (random() * 4.0) as feature3,
    1.0 + (random() * 2.0) as feature4
FROM generate_series(1, 10000);

-- Measure batch inference time (uncomment when model is loaded)
-- .timer on
-- SELECT COUNT(*), AVG(onnx_predict('iris_classifier', feature1, feature2, feature3, feature4)) as avg_pred
-- FROM large_batch;
-- .timer off

-- =============================================================================
-- Error handling and edge cases
-- =============================================================================

-- Test 10: Extension availability check via list_models()
SELECT
    CASE
        WHEN list_models() IS NOT NULL THEN 'Extension loaded successfully'
        ELSE 'Extension load failed'
    END as status;

-- Test 11: NULL feature handling (should show proper error handling)
CREATE TABLE test_nulls AS
SELECT
    1 as id, 5.1 as f1, 3.5 as f2, NULL as f3, 0.2 as f4
UNION ALL
SELECT
    2 as id, 5.9 as f1, 3.0 as f2, 4.2 as f3, 1.5 as f4;

-- This should handle NULLs gracefully (uncomment when model is loaded)
-- SELECT
--     id,
--     CASE
--         WHEN f3 IS NULL THEN 'Skipped due to NULL'
--         ELSE CAST(onnx_predict('iris_classifier', f1, f2, f3, f4) as VARCHAR)
--     END as result
-- FROM test_nulls;

-- =============================================================================
-- Model lifecycle tests
-- =============================================================================

-- Test 12: Model unloading (uncomment when model is loaded)
-- SELECT unload_onnx_model('iris_classifier') as unload_success;

-- Test 13: Verify model is unloaded
-- SELECT list_models() as models_after_unload;

-- =============================================================================
-- Cleanup
-- =============================================================================

DROP TABLE IF EXISTS sample_features;
DROP TABLE IF EXISTS large_batch;
DROP TABLE IF EXISTS test_nulls;

.echo off

-- =============================================================================
-- Summary
-- =============================================================================
SELECT 'Tests Completed' as test_status;
