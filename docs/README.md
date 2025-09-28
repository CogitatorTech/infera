### API Reference

#### Model Management

##### `infera_load_model(model_name VARCHAR, path_or_url VARCHAR) → BOOLEAN`

Loads an ONNX model from a local filesystem path or a remote URL.

```sql
-- Load from a local file
SELECT infera_load_model('my_model', '/path/to/model.onnx');

-- Load from a URL
SELECT infera_load_model('remote_model', 'https://.../model.onnx');
```

##### `infera_unload_model(model_name VARCHAR) → BOOLEAN`

Unloads a previously loaded model from memory.

```sql
SELECT infera_unload_model('my_model');
```

##### `infera_set_autoload_dir(path VARCHAR) → VARCHAR`

Scans a directory, loading all valid `.onnx` files. The model name is derived from the filename.

```sql
SELECT infera_set_autoload_dir('/path/to/models/');
-- Returns a JSON report of loaded and failed models.
```

##### `infera_get_loaded_models() → VARCHAR`

Returns a JSON array of all loaded model names.

```sql
SELECT infera_get_loaded_models();
-- Output: ["my_model", "remote_model"]
```

##### `infera_get_model_info(model_name VARCHAR) → VARCHAR`

Returns detailed information about a loaded model as a JSON object, including its input and output shapes.

```sql
SELECT infera_get_model_info('my_model');
-- Output: {"name":"my_model","input_shape":[-1,3],"output_shape":[-1,1],"loaded":true}
```

#### Inference Functions

##### `infera_predict(model_name VARCHAR, feature1 FLOAT, ..., featureN FLOAT) → FLOAT`

Performs single-output inference. Can be used with literal values or columns.

```sql
-- Predict using columns from a table
SELECT
  infera_predict('my_model', f1, f2, f3) as prediction
FROM features_table;
```

##### `infera_predict_multi(model_name VARCHAR, feature1 FLOAT, ..., featureN FLOAT) → VARCHAR`

Performs multi-output inference, returning all outputs as a JSON array.

```sql
SELECT infera_predict_multi('multi_output_model', 1.0, 2.0);
-- Output: [0.85, 0.12, 0.03]
```

##### `infera_predict_from_blob(model_name VARCHAR, data BLOB) → LIST[FLOAT]`

Performs inference using a `BLOB` of raw floating-point data as input.

```sql
-- The BLOB must contain the raw bytes of the input tensor.
SELECT infera_predict_from_blob('image_model', read_blob('image_tensor.bin'));
```

#### Utility Functions

##### `infera_get_version() → VARCHAR`

Returns a JSON object containing the extension version and build details.

```sql
SELECT infera_get_version();
```

-----

### Building from Source

#### Prerequisites

- Rust toolchain
- CMake 3.21+
- C++ compiler (GCC/Clang)
- `cbindgen` for generating C headers (`cargo install cbindgen`)

#### Build Steps

1.  Build the Rust library:
    ```bash
    cargo build --release --manifest-path infera/Cargo.toml
    ```
2.  Generate C header bindings (if needed):
    ```bash
    make rust-binding-headers
    ```
3.  Build the complete extension:
    ```bash
    make release
    ```

-----

### Testing the Extension

After building, you can run the included SQL test suite.

```bash
# Run all core tests
./build/release/duckdb < tests/sql/test_core_functionality.sql

# Run advanced feature tests
./build/release/duckdb < tests/sql/test_advanced_features.sql

# Run integration and error tests
./build/release/duckdb < tests/sql/test_integration_and_errors.sql
```

-----

### Development Workflow

#### Adding New Rust Functions

1.  Add function to Rust code (`infera/src/lib.rs`):

    ```rust
    #[no_mangle]
    pub unsafe extern "C" fn my_new_function() -> *mut c_char {
        // ... implementation ...
    }
    ```

2.  Add the function name to the `[export]` list in `infera/cbindgen.toml`.

3.  Update C header (regenerate bindings):

    ```bash
    make rust-binding-headers
    ```

4.  Register in C++ extension (`bindings/infera_extension.cpp`):

    ```cpp
    // Add a C++ wrapper and register it in LoadInternal()
    ScalarFunction my_func("my_new_function", {}, LogicalType::VARCHAR, MyNewFunctionWrapper);
    loader.RegisterFunction(my_func);
    ```

5.  Rebuild and test:

    ```bash
    make release
    ./build/release/duckdb -c "SELECT my_new_function();"
    ```
