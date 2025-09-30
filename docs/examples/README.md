## Examples

This directory contains DuckDB SQL scripts that show usage patterns of the Infera extension.
Each file is self‑contained and can be executed in the DuckDB shell (or via `duckdb < file.sql`).

### Prerequisites

1. Build the extension
   ```bash
   make release
   ```
2. Start the DuckDB shell from the project root directory
   ```bash
   ./build/release/duckdb
   ```
3. Inside the shell, load a script
   ```sql
   .read docs/examples/e1_core_functionality.sql
   ```

### Running All Examples

```bash
make examples
```
