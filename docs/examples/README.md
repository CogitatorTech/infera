# Infera Examples

This directory contains DuckDB SQL scripts that show usage patterns of the Infera extension.
Each file is self‑contained and can be executed in the DuckDB shell (or via `duckdb < file.sql`).

## Prerequisites

1. Build the extension:
   ```bash
   make release
   ```
2. Start the DuckDB shell from the project root directory:
   ```bash
   ./build/release/duckdb
   ```
3. Inside the shell, load a script:
   ```sql
   .read docs/examples/e1_core_functionality.sql
   ```

## Example Index

| File                            | Topic                                | Functionalities                                                                                           |
|---------------------------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `e1_core_functionality.sql`     | Core lifecycle                       | Version, load, inspect model info, single prediction, unload, autoload directory reuse                    |
| `e2_advanced_features.sql`      | Remote and BLOB inference            | Remote model load (GitHub), large vision model, constructing a zero‑filled BLOB for inference             |
| `e3_integration_and_errors.sql` | Integration patterns and errors      | Deterministic batch table, aggregation, null feature detection, missing model handling                    |
| `e4_multi_output.sql`           | Multi‑output models                  | Uses `multi_output.onnx`, shows `infera_predict_multi` vs (commented) single‑output mismatch              |
| `e5_autoload_and_json.sql`      | Autoload and lightweight JSON checks | Error JSON from missing dir, loading multiple models via directory scan, simple substring presence checks |
| `e6_blob_diagnostics.sql`       | BLOB diagnostics                     | Shows correct BLOB sizing for mobilenet; commented invalid case to illustrate error path                  |

## Running All Examples

```bash
make examples
```
