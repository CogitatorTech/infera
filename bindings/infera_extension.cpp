#define DUCKDB_EXTENSION_MAIN

#include "infera_extension.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

// Include the generated Rust header
#include "rust.h"

namespace duckdb {

// The actual C++ function that DuckDB will call
static void HelloWorldInfera(DataChunk &args, ExpressionState &state, Vector &result) {
    // Call the Rust function (allocates a C string)
    char *response_from_rust = hello_infera();

    // Copy into DuckDB Value and then free the original allocation to avoid leaking
    // For a zero-arg scalar function DuckDB will produce a single row in the result chunk
    result.SetValue(0, Value(response_from_rust));
    infera_free(response_from_rust);
}

static void LoadInternal(ExtensionLoader &loader) {
    // Register the function with the new ExtensionLoader API (ExtensionUtil removed upstream)
    ScalarFunction hello_func("hello_infera", {}, LogicalType::VARCHAR, HelloWorldInfera);
    loader.RegisterFunction(hello_func);
    loader.SetDescription("Infera demo extension backed by Rust");
}

void InferaExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
}

std::string InferaExtension::Name() {
    return "infera";
}

} // namespace duckdb

// New-style C++ extension entrypoint macro
DUCKDB_CPP_EXTENSION_ENTRY(infera, loader) {
    duckdb::LoadInternal(loader);
}
