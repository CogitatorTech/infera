include_directories(${CMAKE_CURRENT_LIST_DIR}/infera/bindings/include)

duckdb_extension_load(infera
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Manually link the pre-built Rust static library into the generated extension targets.
# The Rust library is built beforehand by the Makefile (cargo build --release --features duckdb_extension).
set(INFERA_RUST_LIB ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/libinfera.a)

# Collect candidate paths in priority order
set(_INFERA_RUST_CANDIDATES)

# 1. If CARGO_TARGET_DIR env var set (relative like target/<triple> or absolute) add its release path
if(DEFINED ENV{CARGO_TARGET_DIR})
    set(_CTD $ENV{CARGO_TARGET_DIR})
    # Normalize possible relative path (cargo executed from infera crate dir)
    if(NOT IS_ABSOLUTE "${_CTD}")
        set(_CTD_FULL ${CMAKE_CURRENT_LIST_DIR}/infera/${_CTD})
    else()
        set(_CTD_FULL ${_CTD})
    endif()
    list(APPEND _INFERA_RUST_CANDIDATES ${_CTD_FULL}/release/libinfera.a ${_CTD_FULL}/release/infera.lib)
endif()

# 2. Target-triple specific directory if Rust_CARGO_TARGET defined
if(DEFINED Rust_CARGO_TARGET AND NOT "${Rust_CARGO_TARGET}" STREQUAL "")
    list(APPEND _INFERA_RUST_CANDIDATES
        ${CMAKE_CURRENT_LIST_DIR}/infera/target/${Rust_CARGO_TARGET}/release/libinfera.a
        ${CMAKE_CURRENT_LIST_DIR}/infera/target/${Rust_CARGO_TARGET}/release/infera.lib)
endif()

# 3. Default host target release dir (pre-CARGO_TARGET_DIR layout)
list(APPEND _INFERA_RUST_CANDIDATES ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/libinfera.a ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/infera.lib)

# Select first existing candidate
foreach(_cand IN LISTS _INFERA_RUST_CANDIDATES)
    if(EXISTS ${_cand})
        set(INFERA_RUST_LIB ${_cand})
        message(STATUS "[infera] Selected Rust static library: ${INFERA_RUST_LIB}")
        break()
    endif()
endforeach()

# If the expected default path does not exist (common on Windows MSVC or when using target triples),
# try alternate locations and naming conventions.
if(NOT EXISTS ${INFERA_RUST_LIB})
    # Look for MSVC-style static library name (infera.lib) in the release root
    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/infera.lib)
        set(INFERA_RUST_LIB ${CMAKE_CURRENT_LIST_DIR}/infera/target/release/infera.lib)
    else()
        # Glob any target-triple subdirectory build products (first match wins)
        file(GLOB _INFERA_ALT_LIBS
            "${CMAKE_CURRENT_LIST_DIR}/infera/target/*/release/libinfera.a"
            "${CMAKE_CURRENT_LIST_DIR}/infera/target/*/release/infera.lib"
        )
        list(LENGTH _INFERA_ALT_LIBS _ALT_COUNT)
        if(_ALT_COUNT GREATER 0)
            list(GET _INFERA_ALT_LIBS 0 INFERA_RUST_LIB)
            message(STATUS "[infera] Using alternate discovered Rust library: ${INFERA_RUST_LIB}")
        endif()
    endif()
endif()

if (EXISTS ${INFERA_RUST_LIB})
    message(STATUS "[infera] Found Rust library at: ${INFERA_RUST_LIB}")

    # Create an imported target for the Rust library
    add_library(infera_rust STATIC IMPORTED GLOBAL)
    if(UNIX)
        set(_INFERA_RUST_LINK_LIBS "pthread;dl;m")
    else()
        set(_INFERA_RUST_LINK_LIBS "")
    endif()
    set_target_properties(infera_rust PROPERTIES
        IMPORTED_LOCATION ${INFERA_RUST_LIB}
        INTERFACE_LINK_LIBRARIES "${_INFERA_RUST_LINK_LIBS}"
    )

    # Add the Rust library to global link libraries so it gets linked to everything
    if(UNIX)
        link_libraries(${INFERA_RUST_LIB} pthread dl m)
    else()
        link_libraries(${INFERA_RUST_LIB})
        if(WIN32)
            # Explicitly add Windows system libraries required by Rust dependencies (mio, std I/O paths)
            # Nt* symbols come from ntdll; others from userenv/dbghelp; bcrypt often pulled by crypto backends.
            set(_INFERA_WIN_SYSTEM_LIBS ntdll userenv dbghelp bcrypt)
            link_libraries(${_INFERA_WIN_SYSTEM_LIBS})
            target_link_libraries(infera_rust INTERFACE ${_INFERA_WIN_SYSTEM_LIBS})
        endif()
    endif()

    if(TARGET infera_extension)
        target_link_libraries(infera_extension infera_rust)
        message(STATUS "[infera] Linked Rust library to infera_extension")
    endif()

    if(TARGET infera_loadable_extension)
        target_link_libraries(infera_loadable_extension infera_rust)
        message(STATUS "[infera] Linked Rust library to infera_loadable_extension")
    endif()

    if(UNIX)
        add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${INFERA_RUST_LIB}>)
        add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-lpthread>)
        add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-ldl>)
        add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-lm>)
    else()
        add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${INFERA_RUST_LIB}>)
    endif()
else()
    message(WARNING "[infera] Expected Rust static library not found at ${INFERA_RUST_LIB}. Build Rust crate first.")
endif()
