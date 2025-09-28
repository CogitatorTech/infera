-- Tests for the core APIs of the Infera DuckDB extension

.echo on

LOAD infera;

-- Test 1: Extension Loading
SELECT 'Testing Infera ML Extension Loading' as test_name;

-- Test 2: Model Management Functions
SELECT infera_get_loaded_models() as initial_models_list;
SELECT typeof(infera_get_loaded_models()) as list_models_return_type;

-- Test 3: Error Handling
SELECT infera_get_model_info('nonexistent_model') as nonexistent_model_info;
SELECT infera_unload_model('nonexistent_model') as unload_nonexistent_result;

-- Test 4: Use in complex queries
SELECT
    'Test ID: ' || id as test_name,
    infera_get_loaded_models() as available_models
FROM (VALUES (1), (2), (3)) t(id);

-- Test 5: Function compatibility
SELECT
    length(infera_get_loaded_models()) as models_json_length,
    typeof(infera_get_model_info('test')) as model_info_type;

.echo off
