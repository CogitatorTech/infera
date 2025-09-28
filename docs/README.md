### API Reference

| Function                                                | Return Type      | Description                                                                                                                                                               |
|:--------------------------------------------------------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `infera_load_model(name VARCHAR, path_or_url VARCHAR)`  | `BOOLEAN`        | Loads an ONNX model from a local path or remote URL.                                                                                                                      |
| `infera_unload_model(name VARCHAR)`                     | `BOOLEAN`        | Unloads a model from memory to free resources.                                                                                                                            |
| `infera_set_autoload_dir(path VARCHAR)`                 | `VARCHAR (JSON)` | Scans a directory, loads all the `.onnx` files in it (models), and returns a JSON report.                                                                                 |
| `infera_get_loaded_models()`                            | `VARCHAR (JSON)` | Returns a JSON array containing the names of models that are currently loaded and are ready to be used.                                                                   |
| `infera_get_model_info(name VARCHAR)`                   | `VARCHAR (JSON)` | Returns a JSON object with information about a specific (loaded) model.                                                                                                   |
| `infera_predict(name VARCHAR, features... FLOAT)`       | `FLOAT`          | Performs inference and returns a single float value.                                                                                                                      |
| `infert_predict_multi(name VARCHAR, features... FLOAT)` | `VARCHAR (JSON)` | Performs inference and returns all outputs as a JSON array. This is useful for models that prodcue more than one predictions per sample, like in multi-target regression. |
| `infera_predict_from_blob(name VARCHAR, data BLOB)`     | `LIST[FLOAT]`    | Performs inference on a raw `BLOB` of tensor data, like image data stored in the database.                                                                                |
| `infera_get_version()`                                  | `VARCHAR (JSON)` | Returns a JSON object with the information about the current version of the extension.                                                                                    |

-----

### Usage Examples

#### Model Management

```sql
-- Load a model from a local file
SELECT infera_load_model('local_model', '/path/to/model.onnx');

-- Load a model from a remote URL
SELECT infera_load_model('remote_model', 'https://.../model.onnx');

-- List all loaded models
SELECT infera_get_loaded_models();
-- Output: ["local_model", "remote_model"]

-- Get information about a specific model
SELECT infera_get_model_info('local_model');
-- Output: {"name":"local_model","input_shape":[-1,3],"output_shape":[-1,1],"loaded":true}

-- Unload a model
SELECT infera_unload_model('remote_model');
```

#### Inference

```sql
-- Predict using literal feature values
SELECT infera_predict('my_model', 1.0, 2.5, 3.0) as prediction;

-- Predict using columns from a table
SELECT id,
       infera_predict('my_model', feature1, feature2, feature3) as prediction
FROM features_table;

-- Get multiple outputs as a JSON array
SELECT infera_predict_multi('multi_output_model', 1.0, 2.0);
-- Output: [0.85, 0.12, 0.03]
```

-----

### Building from Source

#### Prerequisites

- Rust toolchain
- CMake 3.21+
- C++ compiler (GCC/Clang)
- `cbindgen` for generating C headers (`cargo install cbindgen`)

#### Build Steps

1. Build the Rust library:
   ```bash
   cargo build --release --manifest-path infera/Cargo.toml
   ```
2. Generate C header bindings (if needed):
   ```bash
   make rust-binding-headers
   ```
3. Build the complete extension:
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

1. Add function to Rust code (`infera/src/lib.rs`):

   ```rust
   #[no_mangle]
   pub unsafe extern "C" fn my_new_function() -> *mut c_char {
       // ... implementation ...
   }
   ```

2. Add the function name to the `[export]` list in `infera/cbindgen.toml`.

3. Update C header (regenerate bindings):

   ```bash
   make rust-binding-headers
   ```

4. Register in C++ extension (`bindings/infera_extension.cpp`):

   ```cpp
   // Add a C++ wrapper and register it in LoadInternal()
   ScalarFunction my_func("my_new_function", {}, LogicalType::VARCHAR, MyNewFunctionWrapper);
   loader.RegisterFunction(my_func);
   ```

5. Rebuild and test:

   ```bash
   make release
   ./build/release/duckdb -c "SELECT my_new_function();"
   ```
