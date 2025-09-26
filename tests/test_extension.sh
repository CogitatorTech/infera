#!/bin/bash
# Test script for the Infera DuckDB extension

set -e

echo "=== Testing Infera DuckDB Extension ==="

# Path to the built DuckDB executable
DUCKDB_PATH="./build/release/duckdb"

# Check if DuckDB was built
if [ ! -f "$DUCKDB_PATH" ]; then
    echo "❌ DuckDB executable not found at $DUCKDB_PATH"
    echo "Please run 'make release' first to build the extension"
    exit 1
fi

echo "✅ Found DuckDB executable"

# Test 1: Basic function test
echo ""
echo "🧪 Test 1: Basic hello_infera() function call"
$DUCKDB_PATH -c "SELECT hello_infera() as greeting;"

# Test 2: Function metadata
echo ""
echo "🧪 Test 2: Check function return type"
$DUCKDB_PATH -c "SELECT typeof(hello_infera()) as return_type;"

# Test 3: Multiple calls
echo ""
echo "🧪 Test 3: Multiple function calls in one query"
$DUCKDB_PATH -c "SELECT hello_infera() as call1, hello_infera() as call2;"

# Test 4: Use in subquery
echo ""
echo "🧪 Test 4: Function in subquery and aggregation"
$DUCKDB_PATH -c "
SELECT
    count(*) as total_rows,
    hello_infera() as greeting
FROM (VALUES (1), (2), (3)) t(id);"

# Test 5: Check extension is loaded
echo ""
echo "🧪 Test 5: List loaded extensions"
$DUCKDB_PATH -c "SELECT * FROM duckdb_extensions() WHERE extension_name = 'infera';"

# Test 6: Performance test
echo ""
echo "🧪 Test 6: Performance test (1000 calls)"
time $DUCKDB_PATH -c "
SELECT count(*) as total_calls
FROM (
    SELECT hello_infera() as greeting
    FROM range(1000)
);"

echo ""
echo "🎉 All tests completed successfully!"
