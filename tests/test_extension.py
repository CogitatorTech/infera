#!/usr/bin/env python3
"""
Test suite for the Infera DuckDB extension using Python
"""

import subprocess
import sys
import time
from pathlib import Path


def run_duckdb_query(query, duckdb_path="./build/release/duckdb"):
    """Run a DuckDB query and return the result"""
    try:
        result = subprocess.run(
            [duckdb_path, "-c", query],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Query failed: {query}")
        print(f"Error: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"❌ DuckDB executable not found at {duckdb_path}")
        print("Please run 'make release' first to build the extension")
        return None


def test_basic_functionality():
    """Test basic extension functionality"""
    print("🧪 Testing basic functionality...")

    # Test the function exists and returns expected value
    result = run_duckdb_query("SELECT hello_infera() as greeting;")
    if result and "Hello from Infera!" in result:
        print("✅ hello_infera() function works correctly")
        return True
    else:
        print("❌ hello_infera() function failed")
        print(f"Got result: {result}")
        return False


def test_function_properties():
    """Test function properties and metadata"""
    print("🧪 Testing function properties...")

    # Check return type
    result = run_duckdb_query("SELECT typeof(hello_infera()) as return_type;")
    if result and "VARCHAR" in result:
        print("✅ Function returns VARCHAR type as expected")
    else:
        print("❌ Unexpected return type")
        print(f"Got: {result}")
        return False

    # Check function is deterministic (should return same value)
    result = run_duckdb_query("SELECT hello_infera() = hello_infera() as is_deterministic;")
    if result and "true" in result.lower():
        print("✅ Function is deterministic")
    else:
        print("❌ Function appears non-deterministic")
        return False

    return True


def test_performance():
    """Test extension performance"""
    print("🧪 Testing performance...")

    start_time = time.time()
    result = run_duckdb_query("SELECT count(*) FROM (SELECT hello_infera() FROM range(1000));")
    end_time = time.time()

    if result and "1000" in result:
        duration = end_time - start_time
        print(f"✅ Performance test passed: 1000 calls in {duration:.3f} seconds")
        return True
    else:
        print("❌ Performance test failed")
        return False


def test_extension_info():
    """Test extension information"""
    print("🧪 Testing extension information...")

    # Check if extension is loaded
    result = run_duckdb_query("SELECT extension_name FROM duckdb_extensions() WHERE extension_name = 'infera';")
    if result and "infera" in result:
        print("✅ Extension is properly loaded and registered")
        return True
    else:
        print("❌ Extension not found in loaded extensions")
        return False


def main():
    """Run all tests"""
    print("=== Infera Extension Test Suite ===")
    print()

    # Check if DuckDB executable exists
    duckdb_path = Path("./build/release/duckdb")
    if not duckdb_path.exists():
        print("❌ DuckDB executable not found at ./build/release/duckdb")
        print("Please run 'make release' first to build the extension")
        return 1

    print("✅ Found DuckDB executable")
    print()

    # Run tests
    tests = [
        test_basic_functionality,
        test_function_properties,
        test_performance,
        test_extension_info,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()

    # Summary
    print("=== Test Summary ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
