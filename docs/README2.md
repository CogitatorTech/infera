# Infera - Production-Grade ONNX Inference Extension for DuckDB

Infera is a high-performance DuckDB extension that enables ONNX model inference directly within SQL queries. Built in
Rust with C++ bindings, it provides a robust, thread-safe, and memory-efficient solution for machine learning inference
in analytical workloads.

## Features

- **ONNX Model Support**: Load and run inference on ONNX models using the Tract inference engine
- **Thread-Safe**: Concurrent model access and inference with optimized locking
- **Memory Efficient**: Zero-copy data transfer and automatic memory management
- **Production Ready**: Comprehensive error handling, logging, and resource cleanup
- **Simple API**: Easy-to-use SQL functions for model management and inference
- **Type Safety**: Robust type conversion and validation for SQL data types
- **Variable Inputs**: Support for models with different numbers of input features

## Installation

### Prerequisites

- DuckDB (version 0.9.0 or later)
- Rust toolchain with Cargo
- CMake and C++ compiler
- ONNX models for inference

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd infera

# Build the Rust library
cd infera
cargo build --release --features tract

# Build the DuckDB extension
cd ..
make release

# Install the extension
make install
```

## API Reference

### Model Management Functions

#### `load_onnx_model(model_name VARCHAR, path VARCHAR) → BOOLEAN`

Loads an ONNX model from the filesystem.

```sql
SELECT load_onnx_model('iris_classifier', '/path/to/iris_model.onnx');
```

**Parameters:**

- `model_name`: Unique identifier for the model
- `path`: Filesystem path to the ONNX model file

**Returns:** `TRUE` if successful, throws exception on error

#### `unload_onnx_model(model_name VARCHAR) → BOOLEAN`

Unloads a previously loaded model from memory.

```sql
SELECT unload_onnx_model('iris_classifier');
```

#### `list_models() → VARCHAR`

Returns a JSON array of all loaded model names.

```sql
SELECT list_models();
-- Output: ["iris_classifier", "housing_predictor"]
```

#### `model_info(model_name VARCHAR) → VARCHAR`

Returns detailed information about a loaded model as JSON.

```sql
SELECT model_info('iris_classifier');
-- Output: {"name":"iris_classifier","input_shape":[1,4],"output_shape":[1,3],"loaded":true}
```

### Inference Functions

#### `onnx_predict(model_name VARCHAR, feature1 FLOAT, ..., featureN FLOAT) → FLOAT`

Performs single-output inference on the specified model.

```sql
-- Example with iris dataset (4 features)
SELECT onnx_predict('iris_classifier', 5.1, 3.5, 1.4, 0.2) as prediction;

-- Batch inference
SELECT sepal_length,
       sepal_width,
       petal_length,
       petal_width,
       onnx_predict('iris_classifier', sepal_length, sepal_width, petal_length, petal_width) as species_prediction
FROM iris_data;
```

**Parameters:**

- `model_name`: Name of the loaded model
- `feature1, ..., featureN`: Feature values (supports up to 9 features)

**Returns:** Single prediction value (first output of the model)

#### `onnx_predict_multi(model_name VARCHAR, feature1 FLOAT, ..., featureN FLOAT) → VARCHAR`

Performs multi-output inference, returning all outputs as a JSON array.

```sql
SELECT onnx_predict_multi('multi_output_model', 1.0, 2.0, 3.0) as all_predictions;
-- Output: [0.85, 0.12, 0.03]
```

## Usage Examples

### Basic Workflow

```sql
-- 1. Load the extension
LOAD
infera;

-- 2. Verify extension is working (empty array or list of models)
SELECT list_models();

-- 3. Load an ONNX model
SELECT load_onnx_model('my_model', '/path/to/model.onnx');

-- 4. Check model information
SELECT model_info('my_model');

-- 5. Run inference on data
SELECT id,
       feature1,
       feature2,
       feature3,
       onnx_predict('my_model', feature1, feature2, feature3) as prediction
FROM my_table;

-- 6. Clean up when done
SELECT unload_onnx_model('my_model');
```

### Advanced Examples

#### Classification with Post-Processing

```sql
-- Load a classification model
SELECT load_onnx_model('classifier', '/models/iris_classifier.onnx');

-- Run classification with probability interpretation
WITH predictions AS (SELECT *,
                            onnx_predict_multi('classifier', sepal_length, sepal_width, petal_length,
                                               petal_width) as raw_output
                     FROM iris_test_data)
SELECT *,
       CASE
           WHEN json_extract_path_text(raw_output, '$[0]')::FLOAT > 0.5 THEN 'setosa'
        WHEN json_extract_path_text(raw_output, '$[1]')::FLOAT > 0.5 THEN 'versicolor'
        ELSE 'virginica'
END
as predicted_species
FROM predictions;
```

#### Regression with Feature Engineering

```sql
-- Load a regression model
SELECT load_onnx_model('housing_model', '/models/housing_predictor.onnx');

-- Feature engineering and prediction
SELECT house_id,
       bedrooms,
       bathrooms,
       sqft,
       age,
       -- Feature engineering
       sqft / bedrooms                         as sqft_per_bedroom,
       -- Prediction
       onnx_predict('housing_model',
                    bedrooms::FLOAT,
                    bathrooms::FLOAT,
                    sqft::FLOAT,
                    age::FLOAT,
                    (sqft / bedrooms):: FLOAT) as predicted_price
FROM housing_data
WHERE bedrooms > 0;
```

#### Batch Processing Large Datasets

```sql
-- Efficient batch processing with model management
SELECT load_onnx_model('batch_model', '/models/large_model.onnx');

-- Process in chunks to manage memory
CREATE TABLE predictions AS
SELECT chunk_id,
       COUNT(*)                                         as records_processed,
       AVG(onnx_predict('batch_model', f1, f2, f3, f4)) as avg_prediction
FROM (SELECT (ROW_NUMBER() OVER() - 1) // 10000 as chunk_id,
             feature1                           as f1,
             feature2                           as f2,
             feature3                           as f3,
             feature4                           as f4
      FROM large_dataset) chunked_data
GROUP BY chunk_id
ORDER BY chunk_id;

-- Clean up
SELECT unload_onnx_model('batch_model');
```

## Performance Considerations

### Memory Management

- Models are loaded into memory once and shared across queries
- Automatic cleanup of inference results prevents memory leaks
- Use `unload_onnx_model()` to free memory when models are no longer needed

### Concurrency

- Thread-safe model access allows concurrent inference
- Read-write locks minimize contention during model operations
- Consider loading models once and reusing across multiple queries

### Optimization Tips

1. **Batch Processing**: Process multiple rows in single queries rather than row-by-row
2. **Model Reuse**: Load models once and use them for multiple inference operations
3. **Feature Preparation**: Prepare features in SQL before inference to minimize overhead
4. **Memory Monitoring**: Monitor memory usage when working with large models or datasets

## Error Handling

The extension provides comprehensive error handling:

```sql
-- Invalid model name
SELECT onnx_predict('nonexistent_model', 1.0, 2.0);
-- Error: Model not found: nonexistent_model

-- Invalid feature count
SELECT onnx_predict('iris_model', 1.0, 2.0);
-- Model expects 4 features
-- Error: Invalid input shape: expected [1,4], got [1,2]

-- Invalid file path
SELECT load_onnx_model('test', '/invalid/path.onnx');
-- Error: Failed to load ONNX model 'test': No such file or directory
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
    - Verify the ONNX file path is correct and accessible
    - Check that the model is valid ONNX format
    - Ensure sufficient memory is available

2. **Inference Errors**
    - Verify the number of features matches model requirements
    - Check feature data types (should be numeric)
    - Use `model_info()` to inspect model requirements

3. **Performance Issues**
    - Consider model complexity and size
    - Monitor memory usage during large batch operations
    - Use appropriate chunk sizes for large datasets

### Debug Information

```sql
-- Check loaded models
SELECT list_models();

-- Get model details
SELECT model_info('model_name');

-- Verify extension is loaded (should return a JSON array, usually [] if none loaded)
SELECT list_models();
```
