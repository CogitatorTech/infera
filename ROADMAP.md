## Feature Roadmap

This document includes the roadmap for the Infera DuckDB extension.
It outlines features to be implemented and their current status.

> [!IMPORTANT]
> This roadmap is a work in progress and is subject to change.

### 1. Inference Interface

* **Input Data Types**
    * [x] `FLOAT` features from table columns.
    * [x] Type casting from `INTEGER`, `BIGINT`, and `DOUBLE` columns.
    * [x] `BLOB` input for tensor data.
    * [ ] `STRUCT` or `MAP` input for named features.
* **Output Data Types**
    * [x] Single `FLOAT` scalar output.
    * [x] Multiple `FLOAT` outputs as a `VARCHAR` containing JSON.
    * [x] Multiple `FLOAT` outputs as a `LIST[FLOAT]`.
    * [ ] Return multiple outputs as a `STRUCT`.
* **Batch Processing**
    * [x] Inference on batches for models with dynamic dimensions.
    * [ ] Automatic batch splitting for models with a fixed batch size.

### 2. Model Management API

* **Model Loading**
    * [x] Load models from local file paths.
    * [x] Load models from URLs with local caching.
    * [x] Load all `.onnx` models from a directory.
* **Model Lifecycle**
    * [x] Unload models from memory.
    * [x] List loaded models.
    * [x] Get model metadata as a JSON object.
    * [ ] Cache eviction policies for remote models.

### 3. Performance and Concurrency

* **Concurrency Control**
    * [x] Thread-safe model store for concurrent queries.
* **Data Transfer**
    * [ ] Process `BLOB` columns in a single FFI call.
    * [ ] Zero-copy data transfer between DuckDB and Rust.
* **Hardware Support**
    * [ ] GPU support for inference via an alternative backend.

### 4. Backend and Format Support

* **ONNX Standard**
    * [x] Support for ONNX operators via the `tract` engine.
    * [ ] Support for models with named inputs and outputs.
* **Alternative Backends**
    * [ ] An optional build using the ONNX Runtime backend.
* **Other Formats**
    * [ ] Support for other model formats like TorchScript or TensorFlow Lite.

### 5. Miscellaneous

* **SQL Functions**
    * [x] Consistent function names for the public API.
* **Error Handling**
    * [x] Error messages for missing models, incorrect argument counts, and NULL inputs.
    * [ ] Error messages with more specific details.
* **Documentation**
    * [x] `README.md` file with API reference and examples.
    * [ ] An official DuckDB extension documentation page.
* **Distribution**
    * [ ] Pre-compiled extension binaries for Linux, macOS, and Windows.
    * [ ] Submission to the DuckDB Community Extensions repository.
