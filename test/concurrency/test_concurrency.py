#!/usr/bin/env python3
"""Concurrent stress test for the Infera DuckDB extension.

This is NOT part of the sqllogictest harness. Run manually:

    python3 test/concurrency/test_concurrency.py

It will:
  * Load the extension once
  * Spawn multiple threads each rapidly loading, predicting, and unloading a model
  * Verify at the end that no models remain loaded

Exit code 0 indicates success; non-zero indicates a failure.
"""
import sys
import threading
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
extension_path = ROOT / 'build' / 'release' / 'extension' / 'infera' / 'infera.duckdb_extension'
model_path = ROOT / 'test' / 'models' / 'linear.onnx'

THREADS = 8
ITERATIONS = 10

lock = threading.Lock()
errors: list[str] = []


def worker(idx: int):
    con = duckdb.connect(database=':memory:')
    try:
        con.execute(f"load '{extension_path.as_posix()}'")
        for i in range(ITERATIONS):
            name = f"lin_{idx}_{i}"
            con.execute("select infera_load_model(?, ?)", [name, model_path.as_posix()])
            # simple deterministic prediction (1.0,2.0,3.0) -> 1.75
            res = con.execute("select infera_predict(?, 1.0, 2.0, 3.0)", [name]).fetchone()[0]
            if abs(res - 1.75) > 1e-5:
                raise AssertionError(f"unexpected prediction {res}")
            con.execute("select infera_unload_model(?)", [name])
        # final safety: should unload again harmlessly
        con.execute("select infera_unload_model('non_existent_again')")
    except Exception as e:  # pragma: no cover - debugging output
        with lock:
            errors.append(f"Thread {idx} error: {e}")
    finally:
        con.close()


def main():
    if not extension_path.exists():
        print(f"Extension not found at {extension_path}. Build first with 'make release'", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"Model file missing: {model_path}", file=sys.stderr)
        return 2

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()

    if errors:
        print("FAIL: concurrency errors detected:")
        for e in errors:
            print(" -", e)
        return 1

    # Final global check using a fresh connection
    con = duckdb.connect(database=':memory:')
    con.execute(f"load '{extension_path.as_posix()}'")
    loaded_json = con.execute("select infera_get_loaded_models()").fetchone()[0]
    con.close()
    if loaded_json not in ('[]', '[]\n'):
        print(f"FAIL: expected no loaded models, got {loaded_json}")
        return 1

    print("PASS: concurrency test completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
