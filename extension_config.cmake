# Ensure the bindings/include directory (containing infera_extension.hpp & rust.h) is on the include path
include_directories(${CMAKE_CURRENT_LIST_DIR}/bindings/include)

duckdb_extension_load(infera
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Manually link the pre-built Rust static library into the generated extension targets.
# The Rust library is built beforehand by the Makefile (cargo build --release --features duckdb_extension).
set(INFERA_RUST_LIB ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/libinfera.a)
if (EXISTS ${INFERA_RUST_LIB})
    message(STATUS "[infera] Found Rust library at: ${INFERA_RUST_LIB}")

    # Create an imported target for the Rust library
    add_library(infera_rust STATIC IMPORTED GLOBAL)
    set_target_properties(infera_rust PROPERTIES
        IMPORTED_LOCATION ${INFERA_RUST_LIB}
        INTERFACE_LINK_LIBRARIES "pthread;dl;m"
    )

    # Add the Rust library to global link libraries so it gets linked to everything
    # This is the most reliable way to ensure all targets get the symbols
    link_libraries(${INFERA_RUST_LIB} pthread dl m)

    # Also try to link to specific extension targets if they exist
    if(TARGET infera_extension)
        target_link_libraries(infera_extension infera_rust)
        message(STATUS "[infera] Linked Rust library to infera_extension")
    endif()

    if(TARGET infera_loadable_extension)
        target_link_libraries(infera_loadable_extension infera_rust)
        message(STATUS "[infera] Linked Rust library to infera_loadable_extension")
    endif()

    # Use a generator expression to add the library to all executable targets
    # This should catch the shell target when it's created
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${INFERA_RUST_LIB}>)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-lpthread>)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-ldl>)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-lm>)

else()
    message(WARNING "[infera] Expected Rust static library not found at ${INFERA_RUST_LIB}. Build Rust crate first.")
endif()
