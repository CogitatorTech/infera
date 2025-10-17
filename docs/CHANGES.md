# Technical Report – Audit and Fixes

Date: 2025-10-17
Scope: Infera (DuckDB extension in Rust for in-database inference)

This report documents issues found, architectural observations, fixes applied, and tests added. All changes were kept minimal and verified with unit and integration tests to avoid regressions.

---

## Summary of Key Fixes

1) Robust output tensor shape handling
- Problem: Output handling assumed 1D/2D tensors, risking incorrect (rows, cols) for N-D outputs and edge cases (scalars).
- Fix: Introduced `shape_rows_cols(shape: &[usize]) -> (usize, usize)` and used it to compute `(rows, cols)` by flattening dimensions after the first. Applied in both `run_inference_impl` and `run_inference_blob_impl`.
- Impact: Predict functions behave consistently for multi-dimensional outputs. `infera_predict_multi` works reliably for models with vector/matrix outputs.

2) Early input shape validation for `infera_predict`
- Problem: When users passed the wrong number of features, errors came from the backend and were opaque.
- Fix: If a model’s inner input dimensions (after the batch dim) are fully known (>0), validate that the provided `cols` equals their product. On mismatch, return a clear `InvalidInputShape { expected, actual }` error.
- Impact: Users now get immediate, descriptive errors without backend traces. Prevents subtle misuses.

3) Partial download cleanup & caching for remote models
- Problem: The temporary file guard used a `commit(self)` style that could be misleading and less idiomatic; robustness around partial files mattered.
- Fix: Switched to `commit(&mut self)` and used a mutable guard to explicitly disarm cleanup after successful `rename`. Behavior unchanged in success and better clarity; failure paths still remove partial files.
- Added: Positive-path test that verifies cache creation, correct content, cache hits, and absence of `*.onnx.part` files.

4) Version JSON stability
- Problem: `model_cache_dir` could serialize to `null` if the path wasn’t valid UTF-8 via `to_str()`.
- Fix: Use `to_string_lossy()` to always produce a valid string.

---

## Files Changed

- `infera/src/engine.rs`
  - Added `shape_rows_cols` helper.
  - Applied helper to compute output `(rows, cols)` for both float-array and blob-based inference.
  - Added input shape validation for known inner dims in `run_inference_impl`.
  - Added unit tests for the shape helper.

- `infera/src/http.rs`
  - Made `TempFileGuard::commit` take `&mut self`; used a mutable guard.
  - Added a success-path test for download/caching to complement existing failure-path tests.

- `infera/src/lib.rs`
  - Ensured `infera_get_version()` always returns a string for `model_cache_dir`.
  - Added a unit test to assert `InvalidInputShape` is raised for mismatched feature counts.

No external/public API signatures were changed; behavior changes are additive and improve error clarity and robustness.

---

## Tests Added (Regression Coverage)

- Engine
  - `engine::tests::test_shape_rows_cols` – validates `(rows, cols)` for scalar, 1D, 2D, and 3D shapes.

- HTTP/Cache
  - `http::tests::test_handle_remote_model_success_and_cache` – validates successful download, cache hit on second call, and absence of partial files.

- FFI/API layer
  - `tests::test_infera_predict_invalid_shape` – loads a 3-feature model and attempts inference with 2 features to assert a clear `Invalid input shape` error path.

All existing tests (Rust unit tests and sqllogictest-based SQL tests) continue to pass.

---

## Verification

- Rust unit tests (with `tract` feature): PASS
- C++ DuckDB extension build (release): PASS
- SQL tests (DuckDB unittest harness; `.test` files): PASS
- Behavior preserved: `.test` suite expects `infera_get_model_info` to return JSON with an error message for missing models; this remains unchanged. A separate `.slt` file expects an exception and is not part of the executed test harness.

---

## Architectural Notes and Observations

- Model store concurrency: The global model map uses `parking_lot::RwLock` and appears correct for concurrent reads/writes. The C++ binding treats `unload` as idempotent (returns `true` even if the model wasn’t loaded) to simplify verification and concurrency tests.

- Output conventions: `infera_predict` expects a single-column output and will error if a model returns multiple columns (as validated in SQL tests). `infera_predict_multi` returns a JSON-encoded array in `VARCHAR`. The blob-based pathway returns `LIST[FLOAT]`, which is efficient for bulk outputs.

- Feature type coverage: The binding currently accepts `FLOAT`, `DOUBLE`, `INTEGER`, and `BIGINT` for features. Support for `DECIMAL` and other numeric types could be added if needed.

- Error surfaces: The C API returns error JSON for certain queries (e.g., `get_model_info` on a missing model), and the C++ layer passes that through as a string. Changing this to throw at the SQL layer would require updating the tests and user expectations.

- Extension versioning: DuckDB extensions must match the DuckDB engine version they’re built against. Loading the built extension into a mismatched DuckDB (e.g., `pip install duckdb` with a different version) will fail. Use the compiled `duckdb` binary from this repository’s build or align versions in your environment.

---

## How to Build and Test

- Build the extension (release):
```bash
make release
```

- Run Rust unit tests:
```bash
cargo test --manifest-path infera/Cargo.toml --features tract --all-targets -- --nocapture
```

- Run SQL tests via DuckDB unittest harness:
```bash
make test
```

- Optional: Concurrency stress test (requires matching DuckDB version in Python)
```bash
python3 test/concurrency/test_concurrency.py
```
If you use `pip install duckdb`, ensure the package version matches the in-tree DuckDB version used for the build, or use the compiled `./build/release/duckdb` shell to validate concurrency.

---

## Potential Follow-Ups (Non-breaking)

- Input feature types: Consider supporting `DECIMAL` and other numeric types with clear casting rules.
- Error surfaces: Optionally make `infera_get_model_info` throw on missing models at the SQL layer and update tests accordingly.
- Data transfer: Explore zero-copy pathways for BLOB inputs/outputs.
- Multi-output ergonomics: Offer a variant that returns `LIST[FLOAT]` directly for `infera_predict_multi` to avoid JSON encoding.
- Backend options: Evaluate an ONNX Runtime backend feature, potentially with GPU support.

---

## Appendix: Rationale For Notable Choices

- Kept `infera_get_model_info` error-as-JSON behavior to preserve existing SQL tests and expected UX.
- Validated input shapes early to provide clearer, user-friendly errors without backend stack traces.
- Normalized output shape reporting to a consistent contract: `(rows, product_of_rest)` for N-D tensors.
- Ensured remote model caching is robust under failure and success paths, and safe to re-use (cache hit semantics).

