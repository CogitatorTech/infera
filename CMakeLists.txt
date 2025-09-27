cmake_minimum_required(VERSION 3.5)

set(CORROSION_VERBOSE_OUTPUT ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

# --- Fallback platform detection (only if DuckDB build system did not set these) ---
# DuckDB's extension build usually injects OS_NAME / OS_ARCH / DUCKDB_PLATFORM.
# When users run `cmake .` manually these are empty -> we derive sensible defaults.
set(_INFERA_ORIG_OS_NAME "${OS_NAME}")
set(_INFERA_ORIG_OS_ARCH "${OS_ARCH}")
if("${OS_NAME}" STREQUAL "")
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _INFERA_OS_NAME)
  set(OS_NAME "${_INFERA_OS_NAME}")
  set(_infera_fallback_platform_detected TRUE)
else()
  set(_INFERA_OS_NAME "${OS_NAME}")
endif()
if("${OS_ARCH}" STREQUAL "")
  set(_INFERA_OS_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  set(OS_ARCH "${_INFERA_OS_ARCH}")
  set(_infera_fallback_platform_detected TRUE)
else()
  set(_INFERA_OS_ARCH "${OS_ARCH}")
endif()
# Normalise common architecture aliases
if(_INFERA_OS_ARCH STREQUAL "x86_64" OR _INFERA_OS_ARCH STREQUAL "amd64")
  set(_INFERA_OS_ARCH amd64)
elseif(_INFERA_OS_ARCH STREQUAL "aarch64" OR _INFERA_OS_ARCH STREQUAL "arm64")
  set(_INFERA_OS_ARCH arm64)
endif()
set(OS_ARCH "${_INFERA_OS_ARCH}")

# Derive DUCKDB_PLATFORM if missing (rough heuristic mirroring DuckDB naming)
if("${DUCKDB_PLATFORM}" STREQUAL "")
  if(_INFERA_OS_NAME STREQUAL "linux")
    if(_INFERA_OS_ARCH STREQUAL "amd64")
      set(DUCKDB_PLATFORM linux_amd64)
    elseif(_INFERA_OS_ARCH STREQUAL "arm64")
      set(DUCKDB_PLATFORM linux_arm64)
    endif()
  elseif(_INFERA_OS_NAME MATCHES "darwin|osx")
    if(_INFERA_OS_ARCH STREQUAL "arm64")
      set(DUCKDB_PLATFORM osx_arm64)
    elseif(_INFERA_OS_ARCH STREQUAL "amd64")
      set(DUCKDB_PLATFORM osx_amd64)
    else()
      set(DUCKDB_PLATFORM osx_universal)
    endif()
  elseif(WIN32)
    if(_INFERA_OS_ARCH STREQUAL "amd64")
      set(DUCKDB_PLATFORM windows_amd64)
    endif()
  endif()
endif()

if(_infera_fallback_platform_detected)
  message(STATUS "[infera] Fallback platform detection engaged: OS_NAME='${OS_NAME}' OS_ARCH='${OS_ARCH}' DUCKDB_PLATFORM='${DUCKDB_PLATFORM}'")
endif()

set(prefix_to_check "wasm")
string(LENGTH "${prefix_to_check}" prefix_length)
string(SUBSTRING "${DUCKDB_PLATFORM}" 0 ${prefix_length} extracted_platform_prefix)

execute_process(
    COMMAND rustup target list --installed
    OUTPUT_VARIABLE RUST_TARGETS
)

# Propagate arch to rust build for CI (best-effort; corrosion will still build host default if unset)
set(Rust_CARGO_TARGET "")
if(_INFERA_OS_NAME STREQUAL "linux")
    if (_INFERA_OS_ARCH STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    else()
        string(FIND "${RUST_TARGETS}" "musl" MUSL_TARGET_FOUND)
        if(NOT MUSL_TARGET_FOUND EQUAL -1)
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-musl")
        else()
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
        endif()
    endif()
elseif(_INFERA_OS_NAME MATCHES "darwin|osx")
    if (_INFERA_OS_ARCH STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    elseif (_INFERA_OS_ARCH STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    endif()
elseif(WIN32)
   if (_INFERA_OS_ARCH STREQUAL "arm64")
       set(Rust_CARGO_TARGET "aarch64-pc-windows-msvc")
   elseif (_INFERA_OS_ARCH STREQUAL "amd64")
       set(Rust_CARGO_TARGET "x86_64-pc-windows-msvc")
   endif()
endif()

string(FIND "${RUST_TARGETS}" "wasm32-unknown-emscripten" WASM_TARGET_FOUND)
if (NOT WASM_TARGET_FOUND EQUAL -1)
  set(Rust_CARGO_TARGET "wasm32-unknown-emscripten")
endif()

# Final safety net default
if("${Rust_CARGO_TARGET}" STREQUAL "")
  set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
  message(STATUS "[infera] Defaulted Rust_CARGO_TARGET to host x86_64-unknown-linux-gnu")
endif()

message(STATUS "RUST_TARGETS: ${RUST_TARGETS}")
message(STATUS "WASM_TARGET_FOUND: ${WASM_TARGET_FOUND}")
message(STATUS "OS_NAME: ${OS_NAME} (orig='${_INFERA_ORIG_OS_NAME}')")
message(STATUS "OS_ARCH: ${OS_ARCH} (orig='${_INFERA_ORIG_OS_ARCH}')")
message(STATUS "DUCKDB_PLATFORM: ${DUCKDB_PLATFORM}")
message(STATUS "Rust_CARGO_TARGET: ${Rust_CARGO_TARGET}")

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

# Detect if we are being included in an in-tree DuckDB build (source dir name == duckdb)
get_filename_component(_TOP_SOURCE_DIR_NAME "${CMAKE_SOURCE_DIR}" NAME)
if(_TOP_SOURCE_DIR_NAME STREQUAL "duckdb")
  set(INFERA_IN_TREE_DUCKDB_BUILD TRUE)
else()
  set(INFERA_IN_TREE_DUCKDB_BUILD FALSE)
endif()

# Derive the extension source root robustly (directory containing this CMakeLists.txt)
get_filename_component(INFERA_EXTENSION_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(INFERA_CRATE_MANIFEST "${INFERA_EXTENSION_DIR}/infera/Cargo.toml")
if(NOT EXISTS "${INFERA_CRATE_MANIFEST}")
  message(FATAL_ERROR "[infera] Expected Rust crate manifest not found at ${INFERA_CRATE_MANIFEST}. CMAKE_CURRENT_LIST_FILE='${CMAKE_CURRENT_LIST_FILE}'")
endif()

if(NOT INFERA_IN_TREE_DUCKDB_BUILD)
  message(STATUS "[infera] Standalone mode: importing Rust crate via Corrosion (${INFERA_CRATE_MANIFEST})")
  corrosion_import_crate(MANIFEST_PATH "${INFERA_CRATE_MANIFEST}" CRATES "infera")
else()
  message(STATUS "[infera] In-tree DuckDB build detected; skipping Corrosion import (Rust static lib expected to be pre-built and linked via extension_config.cmake)")
endif()

set(TARGET_NAME infera)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)
project(${TARGET_NAME})

include_directories(infera/bindings/include)
set(EXTENSION_SOURCES infera/bindings/infera_extension.cpp)

# Remove previous attempt to include DuckDB extension macros when unavailable.
# For a plain standalone configure (cmake ..) we just build the Rust crate; the
# DuckDB C++ extension requires DuckDB's macro definitions. Users should invoke
# the provided Makefile (e.g. `make release`) or integrate with DuckDB's build
# to compile the extension targets. We detect absence of build_static_extension
# and skip gracefully.
if(NOT COMMAND build_static_extension)
  message(STATUS "[infera] DuckDB extension build macros not present (standalone configure). Skipping C++ extension targets.")
else()
  build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
  build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})
endif()

# Remove stale linking logic guarded by targets that won't exist in standalone mode.
if(COMMAND build_static_extension)
  if(TARGET ${EXTENSION_NAME})
    if(TARGET infera-static)
      target_link_libraries(${EXTENSION_NAME} infera-static)
    elseif(TARGET infera)
      target_link_libraries(${EXTENSION_NAME} infera)
    endif()
  endif()
  if(TARGET ${LOADABLE_EXTENSION_NAME})
    if(TARGET infera)
      target_link_libraries(${LOADABLE_EXTENSION_NAME} infera)
    elseif(TARGET infera-static)
      target_link_libraries(${LOADABLE_EXTENSION_NAME} infera-static)
    endif()
  endif()
endif()

if(TARGET ${EXTENSION_NAME})
  install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
else()
  message(STATUS "[infera] No C++ extension target built; skipping install directive.")
endif()
