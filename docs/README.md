### API Reference

Table below includes the information about all SQL functions exposed by the Infera.

| Function                                                | Return Type      | Description                                                                                                                                 |
|:--------------------------------------------------------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| `infera_load_model(name VARCHAR, path_or_url VARCHAR)`  | `BOOLEAN`        | Loads an ONNX model from a local file path or a remote URL and assigns it a unique name. Returns `true` on success.                         |
| `infera_unload_model(name VARCHAR)`                     | `BOOLEAN`        | Unloads a model, freeing its associated resources. Returns `true` on success.                                                               |
| `infera_set_autoload_dir(path VARCHAR)`                 | `VARCHAR (JSON)` | Scans a directory for `.onnx` files, loads them automatically, and returns a JSON report of loaded models and any errors.                   |
| `infera_get_loaded_models()`                            | `VARCHAR (JSON)` | Returns a JSON array containing the names of all currently loaded models.                                                                   |
| `infera_get_model_info(name VARCHAR)`                   | `VARCHAR (JSON)` | Returns a JSON object with metadata about a specific loaded model, including its name, input/output shapes, and status.                     |
| `infera_predict(name VARCHAR, features... FLOAT)`       | `FLOAT`          | Performs inference on a batch of data, returning a single float value for each input row.                                                   |
| `infera_predict_multi(name VARCHAR, features... FLOAT)` | `VARCHAR (JSON)` | Performs inference and returns all outputs as a JSON-encoded array. This is useful for models that produce multiple predictions per sample. |
| `infera_predict_from_blob(name VARCHAR, data BLOB)`     | `LIST[FLOAT]`    | Performs inference on raw `BLOB` data (for example, used for an image tensor), returning the result as a list of floats.                    |
| `infera_get_version()`                                  | `VARCHAR (JSON)` | Returns a JSON object with version and build information for the Infera extension.                                                          |

---

### Usage Examples

This section includes some examples of how to use the Infera functions.

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

-- Unload a loaded model
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

-- Get multiple outputs as a JSON array.
-- This is useful models that return multiple outputs per prediction (like a non-binary classifier)
SELECT infera_predict_multi('multi_output_model', 1.0, 2.0);
-- Output: [0.85, 0.12, 0.03]

-- Predict using raw BLOB data (like tensor data)
SELECT infera_predict_from_blob('my_model', my_blob_column)
FROM my_table;
-- Expected output: [0.1, 0.2, 0.3, ...] (as a LIST<FLOAT>)
```

> [!IMPORTANT]
> When you use a model model for inference, in essence it will be executed on your machine.
> So make sure you download and use models from trusted sources only.

#### Utility Functions

```sql
-- Get a JSON list of all loaded models
SELECT infera_get_loaded_models();
-- Output: ["linear_model", "squeezenet"]

-- Get detailed metadata for a specific model
SELECT infera_get_model_info('squeezenet');
/* Output:
{
  "name": "squeezenet",
  "input_shape": [1, 3, 224, 224],
  "output_shape": [1, 1000],
  "loaded": true
}
*/

-- Load all models from the 'models/' directory
SELECT infera_set_autoload_dir('path/to/your/models');
/* Output:
{
  "loaded": ["model1", "model2"],
  "errors": []
}
*/
```

---

### Building Infera from Source

To build Infera from source, you need to have GNU Make, CMake, and a C++ compiler (like GCC or Clang) installed.
You also need to have Rust (nightly) and Cargo installed via `rustup`.

1. **Clone the repository:**

   ```bash
   git clone --recursive https://github.com/CogitatorTech/infera.git
   cd infera
   ```

> [!NOTE]
> The `--recursive` flag is important to clone the required submodules (like DuckDB).

2. **Install dependencies:**

   The project includes a `Makefile` target to help set up the development environment. For Debian-based systems, you
   can run:
   ```bash
   make install-deps
   ```
   This will install necessary system packages, Rust tools, and Python dependencies. For other operating systems, please
   refer to the `Makefile` to see the list of dependencies and install them manually.

3. **Build the extension:**

   Run the following command to build the DuckDB shell with the Infera extension included:
   ```bash
   make release
   ```
   This will create a `duckdb` executable inside the `build/release/` directory.

4. **Run the custom DuckDB shell:**

   You can now run the custom-built DuckDB shell:
   ```bash
   ./build/release/duckdb
   ```
   The Infera extension will be automatically available, and you can start using the `infera_*` functions right away
   without needing to run the `LOAD` command.

> [!NOTE]
> After a successful build, you can run the following binaries:
> - `./build/release/duckdb`: this is the newest stable version of duckdb with Infera statically linked to it.
> - `./build/release/test/unittest`: this is the test runner of duckdb (for `.test` files).
> - `./build/release/extension/infera/infera.duckdb_extension`: this is the loadable binary that is a `.so`,
    `.dylib`, or `.dll` file based on your platform.

---

### Architecture

Infera is made up of two main components:

1. **Rust Core (`infera/src/`)**: The core logic is implemented in Rust. This component is responsible for:
    * Loading ONNX models from local files or remote URLs.
    * Caching remote models for efficient reuse.
    * Managing the lifecycle of loaded models.
    * Performing the actual model inference using the [Tract](https://github.com/sonos/tract) engine.
    * Exposing a C-compatible Foreign Function Interface (FFI) so that it can be called from other languages.

2. **C++ DuckDB Bindings (`infera/bindings/`)**: A C++ layer that connects the Rust core and DuckDB. Its
   responsibilities include:
    * Defining the custom SQL functions (like `infera_load_model` and `infera_predict`).
    * Translating data from DuckDB's internal vector-based format into the raw data pointers expected by the Rust FFI.
    * Calling the Rust functions and handling the returned results.
    * Integrating with DuckDB's extension loading mechanism.
