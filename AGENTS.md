# AGENTS.md

This file provides guidance to coding agents collaborating on this repository.

## Mission

Infera is a DuckDB extension for running machine learning inference (on ONNX models) directly from SQL.
It has two tightly coupled layers:

1. A Rust core that loads and caches ONNX models, runs inference through Tract, manages engine state, and exposes a C ABI.
2. A C++ DuckDB extension layer that registers SQL functions and table functions on top of that Rust ABI.

Priorities, in order:

1. Correctness and safety of the SQL-facing inference behavior.
2. Compatibility with supported DuckDB versions and extension CI.
3. Reliable model loading, cache behavior, and error propagation.
4. Small, well-tested changes that preserve the existing Rust/C++ boundary.

## Core Rules

- Use English for code, comments, docs, tests, and commit messages.
- Prefer focused fixes over broad refactoring.
- Preserve the existing Rust/C ABI unless the task explicitly requires changing it.
- Treat `infera/bindings/include/rust.h` as generated code. If Rust FFI signatures change, regenerate it with `make create-bindings`.
- Do not edit vendored code under `external/` unless the task is explicitly about updating or patching a vendored dependency.
- Do not add new dependencies, network behavior, or background processes unless the requirement clearly calls for them.
- Keep docs and examples aligned with user-visible SQL behavior.

## Writing Style

- Use Oxford commas in inline lists: "a, b, and c" not "a, b, c".
- Do not use em dashes. Restructure the sentence, or use a colon or semicolon instead.
- Avoid colorful adjectives and adverbs. Write "TCP proxy" not "lightweight TCP proxy", "scoring components" not "transparent scoring components".
- Use noun phrases for checklist items, not imperative verbs. Write "redundant index detection" not "detect redundant indexes".
- Headings in Markdown files must be in the title case: "Build from Source" not "Build from source". Minor words (a, an, the, and, but, or, for, in,
  on, at, to, by, of, is, are, was, were, be) stay lowercase unless they are the first word.

## Repository Layout

- `infera/src/lib.rs`: Rust crate entry point and public exports for the C ABI surface.
- `infera/src/engine.rs`: Inference engine state, model registry, and execution paths.
- `infera/src/model.rs`: ONNX model loading, metadata, and representation.
- `infera/src/config.rs`: Runtime configuration and environment-driven settings.
- `infera/src/http.rs`: Remote model fetching.
- `infera/src/error.rs`: Error types and last-error plumbing shared across the FFI boundary.
- `infera/src/ffi_utils.rs`: Shared helpers for the FFI boundary.
- `infera/bindings/infera_extension.cpp`: DuckDB extension implementation that maps SQL calls to the Rust ABI.
- `infera/bindings/include/infera_extension.hpp`: C++ extension declarations.
- `infera/bindings/include/rust.h`: Generated C header for the Rust ABI.
- `CMakeLists.txt`: Top-level CMake integration, platform detection, and Corrosion setup.
- `extension_config.cmake`: DuckDB extension wiring and linkage to the prebuilt Rust static library.
- `test/sql/`: Sqllogictest files for SQL-level extension behavior.
- `test/models/`: Sample ONNX models used by SQL tests.
- `test/concurrency/`: Concurrency and stress tests.
- `docs/examples/`: SQL examples that should remain runnable against a local build.
- `.github/workflows/tests.yml`: Rust tests and SQL tests in CI.
- `.github/workflows/lints.yml`: Rust formatting and clippy checks in CI.
- `.github/workflows/dist_pipeline.yml`: cross-platform extension packaging against DuckDB `main` and `v1.5.2`.

## Architecture Notes

### Rust Core

The Rust crate owns inference-facing behavior: model loading from local paths or URLs, ONNX parsing through Tract, engine state, model caching, tensor
shaping, JSON output formatting, and error handling.
All SQL-visible behavior should ultimately reduce to deterministic Rust operations exposed through the FFI layer.

### FFI Boundary

The boundary between Rust and C++ is intentionally narrow:

- Rust returns primitive values, heap-allocated C strings, or result structs that own their own buffers.
- C++ is responsible for converting those values into DuckDB vectors and freeing Rust-allocated memory with the matching `infera_free_*` functions.
- Errors should cross the boundary through the existing last-error mechanism instead of ad hoc conventions.

When changing anything on one side of the boundary, inspect the matching code on the other side in the same change.

### DuckDB Layer

`infera/bindings/infera_extension.cpp` registers scalar functions such as `infera_predict` and `infera_predict_from_blob`, plus table-style behavior
for
listing and inspecting loaded models.
DuckDB API compatibility matters here. If a change touches vector access, function registration, or scans, verify against the vendored DuckDB headers
in `external/duckdb`.

### Build Integration

`make release` and `make debug` build the Rust crate first, then build DuckDB plus the extension.
`extension_config.cmake` expects a prebuilt Rust static library and links it into the DuckDB extension targets.
`CMakeLists.txt` also contains platform and Rust-target selection logic used by local builds and CI distribution builds.

## Generated and Derived Files

- `infera/bindings/include/rust.h` is generated from the Rust crate via `cbindgen`.
- `infera/target/`, `build/`, and coverage outputs such as `infera/cobertura.xml` are build artifacts, not source.
- Do not hand-edit generated artifacts unless the task explicitly requires it, and you explain why.

## Rust Conventions

- Edition: Rust 2021.
- Format with `cargo fmt` through `make rust-format`.
- Lint with `cargo clippy` through `make rust-lint`.
- Follow the existing error style: return typed errors internally, then translate them once at the FFI boundary.
- Avoid `unwrap()` and `expect()` in production code. CI denies them via clippy.
- Prefer existing crates and helpers already in use before introducing new abstractions.

## C++ and DuckDB Conventions

- Keep C++ changes narrowly scoped to DuckDB integration concerns.
- Match current DuckDB APIs used by the vendored headers in `external/duckdb/src/include`.
- Be careful with vector mutability and ownership. Many DuckDB helpers have separate const and mutable accessors.
- Keep user-facing SQL function names, signatures, and error messages stable unless the task explicitly changes them.

## Required Validation

Run the narrowest relevant checks, then expand if the change crosses layers.

| Area                 | Command            | Use When                                                            |
|----------------------|--------------------|---------------------------------------------------------------------|
| Rust formatting      | `make rust-format` | Any Rust code changed                                               |
| Rust lint            | `make rust-lint`   | Any Rust code changed                                               |
| Rust tests           | `make rust-test`   | Rust logic, FFI, model loading, inference, or HTTP behavior changed |
| Extension build      | `make release`     | C++, CMake, linkage, or SQL-facing behavior changed                 |
| SQL tests            | `make test`        | SQL functions, table functions, or DuckDB integration changed       |
| Examples             | `make examples`    | User-visible SQL behavior or docs/examples changed                  |
| Combined local check | `make check`       | Small Rust-only changes                                             |

Minimum expectations:

- Rust-only logic changes: `make rust-test` and `make rust-lint`.
- FFI changes: `make create-bindings`, `make rust-test`, and `make release`.
- C++ or SQL-surface changes: `make release` and `make test`.
- Docs/examples updates that affect runnable SQL: `make examples`.

## Testing Expectations

- Rust tests in `infera/src/` cover crate behavior, engine state, model loading, and error paths.
- SQL tests in `test/sql/*.test` validate the extension from DuckDB's side and are the right place for user-visible SQL regressions.
- Concurrency tests in `test/concurrency/` cover thread-safety of the inference engine.
- Prefer local or offline-friendly tests for model-facing logic. Do not make CI depend on remote model downloads unless the repository already does so
  for that path.
- If you change model loading, cache semantics, inference output shape, or SQL function output shape, add or update tests in the layer where the
  regression would be caught first.

## Change Design Checklist

Before coding:

1. Identify whether the task belongs to Rust core, FFI boundary, C++ DuckDB layer, build wiring, or tests.
2. Check whether the change affects one DuckDB version or both `main` and `v1.5.2`.
3. Decide whether `rust.h`, docs examples, or SQL tests need to move with the code.
4. Confirm whether the change is safe under offline or model-free test conditions.

Before submitting:

1. Relevant build and test commands pass locally, or any gaps are explicitly called out.
2. Generated headers are refreshed if the Rust ABI is changed.
3. User-visible SQL changes are covered by `test/sql` and reflected in docs/examples where appropriate.
4. Changes to DuckDB integration are reviewed against the vendored headers, not memory of older APIs.

## Commit and PR Hygiene

- Keep commits scoped to one logical change.
- Mention both layers when relevant: for example, "update Rust FFI and DuckDB binding for X".
- PR descriptions should include:
    1. Behavioral change summary.
    2. Validation runs locally.
    3. Whether the change affects Rust only, SQL surface, or cross-version DuckDB compatibility.
