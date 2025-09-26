# Infera DuckDB Extension

A DuckDB extension powered by Rust that demonstrates how to integrate Rust code with DuckDB's extension system.

<img src="../assets/logos/rustacean-flat-noshadow.svg" align="center" width="25%"/>

## Overview

The Infera extension provides a simple example of how to:
- Create Rust functions that can be called from SQL
- Build a DuckDB extension with Rust backing code
- Handle memory management between Rust and C++
- Test extension functionality

## Building the Extension

### Prerequisites

- Rust toolchain (nightly)
- CMake 3.21+
- C++ compiler (GCC/Clang)
- cbindgen for generating C headers

### Build Steps

1. **Build the Rust library first:**
   ```bash
   cargo build --release --features duckdb_extension --manifest-path infera/Cargo.toml
   ```

2. **Generate C header bindings:**
   ```bash
   make rust-binding-headers
   ```

3. **Build the complete extension:**
   ```bash
   make release
   ```

## Testing the Extension

Once built successfully, you can test the extension using several methods:

### Method 1: Direct Command Line Testing

Test basic functionality:
```bash
# Test the hello_infera function
./build/release/duckdb -c "SELECT hello_infera() as greeting;"

# Expected output: "Hello from Infera!"

# Test return type
./build/release/duckdb -c "SELECT typeof(hello_infera()) as return_type;"

# Expected output: "VARCHAR"

# Test multiple calls
./build/release/duckdb -c "SELECT hello_infera() as call1, hello_infera() as call2;"
```

### Method 2: Using SQL Test File

Run the pre-written SQL tests:
```bash
./build/release/duckdb < tests/test_extension.sql
# Or using the Makefile target:
make test-sql
```

### Method 3: Automated Test Scripts

#### Bash Test Script
```bash
./tests/test_extension.sh
# Or using the Makefile target:
make test-extension
```

This runs comprehensive tests including:
- ✅ Basic function calls
- ✅ Return type verification
- ✅ Multiple function calls
- ✅ Performance testing (1000 calls)
- ✅ Extension registration verification

#### Python Test Script
```bash
python3 tests/test_extension.py
# Or using the Makefile target:
make test-python
```

Provides detailed test results with error handling and performance metrics.

### Method 4: Interactive Testing

Start DuckDB interactively:
```bash
./build/release/duckdb
```

Then run SQL commands:
```sql
-- Test basic functionality
SELECT hello_infera() as greeting;

-- Check function properties
SELECT typeof(hello_infera()) as return_type;
SELECT hello_infera() = hello_infera() as is_deterministic;

-- Verify extension is loaded
SELECT extension_name FROM duckdb_extensions() WHERE extension_name = 'infera';

-- Performance test
SELECT count(*) FROM (SELECT hello_infera() FROM range(1000));

-- Exit
.quit
```

### Method 5: Testing Loadable Extension

Test the loadable version of the extension:
```bash
./build/release/duckdb -c "
INSTALL './build/release/extension/infera/infera.duckdb_extension';
LOAD 'infera';
SELECT hello_infera() as greeting;
"
```

## Available Test Commands

The project includes several test targets in the Makefile:

| Command | Description |
|---------|-------------|
| `make test-extension` | Run comprehensive bash test suite |
| `make test-python` | Run Python test suite with detailed reporting |
| `make test-sql` | Run SQL tests from file |
| `make test-quick` | Quick single function test |

## Expected Test Results

When the extension works correctly, you should see:

| Test | Expected Result |
|------|----------------|
| `hello_infera()` | `"Hello from Infera!"` |
| `typeof(hello_infera())` | `"VARCHAR"` |
| Function determinism | `true` (same input = same output) |
| Performance (1000 calls) | Should complete in < 1 second |
| Extension registration | `infera` appears in `duckdb_extensions()` |

## Troubleshooting

### Common Issues

1. **"Function not found" error**
   - Extension might not be loaded properly
   - Verify build completed successfully
   - Check that `./build/release/duckdb` exists

2. **"File not found" error**
   - Build may have failed
   - Run `make release` to rebuild
   - Check for build errors in output

3. **Permission errors**
   - Make executable: `chmod +x ./build/release/duckdb`
   - Ensure test scripts are executable: `chmod +x tests/test_extension.sh tests/test_extension.py`

4. **Linking errors during build**
   - Ensure Rust library is built first: `cargo build --release --features duckdb_extension`
   - Generate headers: `make rust-binding-headers`
   - Clean and rebuild: `rm -rf build/release && make release`

### Build Verification

To verify your build was successful, check for these indicators:
- ✅ Build reaches 100% completion
- ✅ `[ 80%] Built target infera_extension` appears in output
- ✅ `[ 81%] Built target infera_loadable_extension` appears in output
- ✅ No "undefined reference" linking errors
- ✅ `./build/release/duckdb` executable exists

## Development Workflow

### Adding New Rust Functions

1. **Add function to Rust code** (`infera/src/lib.rs`):
   ```rust
   #[no_mangle]
   pub unsafe extern "C" fn my_new_function() -> *mut c_char {
       let s = CString::new("My new result!").unwrap();
       s.into_raw()
   }
   ```

2. **Update C header** (regenerate bindings):
   ```bash
   make rust-binding-headers
   ```

3. **Register in C++ extension** (`bindings/infera_extension.cpp`):
   ```cpp
   // Add new function wrapper and register it
   ScalarFunction my_func("my_new_function", {}, LogicalType::VARCHAR, MyNewFunctionWrapper);
   loader.RegisterFunction(my_func);
   ```

4. **Rebuild and test**:
   ```bash
   make release
   ./build/release/duckdb -c "SELECT my_new_function();"
   ```

## Architecture

The extension consists of:
- **Rust Library** (`infera/`) - Core logic and functions
- **C++ Bindings** (`bindings/`) - DuckDB integration layer
- **CMake Configuration** (`extension_config.cmake`) - Build system integration
- **Test Suite** (`tests/`) - Comprehensive testing framework

## Files Overview

| File | Purpose |
|------|---------|
| `infera/src/lib.rs` | Rust functions exposed to C++ |
| `bindings/include/rust.h` | Generated C header (auto-generated) |
| `bindings/infera_extension.cpp` | DuckDB extension registration |
| `extension_config.cmake` | CMake build configuration |
| `tests/test_extension.sql` | SQL test queries |
| `tests/test_extension.sh` | Bash test script |
| `tests/test_extension.py` | Python test suite |

## Contributing

When contributing:
1. Add tests for new functionality
2. Update this documentation
3. Ensure all tests pass
4. Follow Rust and C++ coding standards
5. Update function exports in both Rust and C++ layers
