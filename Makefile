# Variables for DuckDB Extension
PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RUST_LIB := infera/target/release/libinfera.a
SHELL := /bin/bash

# DuckDB Extension Configuration
DUCKDB_SRCDIR := ./external/duckdb/
EXT_NAME := infera
EXT_CONFIG := ${PROJ_DIR}extension_config.cmake

# Test files location
TESTS_DIR := tests

# Include the official DuckDB extension makefile
include external/extension-ci-tools/makefiles/duckdb_extension.Makefile

# Override the set_duckdb_version target to use the correct path
set_duckdb_version:
	cd external/duckdb && git checkout $(DUCKDB_GIT_VERSION)

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show help messages for all available targets
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

###########################################
# Rust Development Targets
###########################################

.PHONY: rust-format
rust-format: ## Format Rust files
	@echo "Formatting Rust files..."
	@cargo fmt --manifest-path infera/Cargo.toml

.PHONY: rust-test
rust-test: rust-format ## Run Rust tests
	@echo "Running Rust tests..."
	@cargo test --manifest-path infera/Cargo.toml --all-targets -- --nocapture

.PHONY: rust-lint
rust-lint: rust-format ## Run Rust linters
	@echo "Linting Rust files..."
	@cargo clippy --manifest-path infera/Cargo.toml -- -D warnings

.PHONY: rust-fix-lint
rust-fix-lint: ## Fix Rust linter warnings
	@echo "Fixing linter warnings..."
	@cargo clippy --fix --allow-dirty --allow-staged --manifest-path infera/Cargo.toml -- -D warnings

.PHONY: rust-audit
rust-audit: ## Run security audit on Rust dependencies
	@echo "Running security audit..."
	@cargo audit --file infera/Cargo.toml

.PHONY: rust-doc
rust-doc: rust-format ## Generate Rust documentation
	@echo "Generating Rust documentation..."
	@cargo doc --manifest-path infera/Cargo.toml --no-deps --document-private-items

###########################################
# General Development Targets
###########################################

.PHONY: sync-externals
sync-externals: ## Sync external submodules (DuckDB and extension-ci-tools)
	@echo "Syncing external submodules..."
	@git submodule update --init --recursive
	@echo "External submodules synced."

.PHONY: setup-hooks
setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	pre-commit install --hook-type pre-commit
	pre-commit install --hook-type pre-push
	pre-commit install-hooks

.PHONY: test-hooks
test-hooks: ## Test Git hooks on all files
	pre-commit run --all-files

###########################################
# DuckDB Extension Targets
###########################################

.PHONY: rust-lib
rust-lib: rust-format ## Build the Rust static library
	@echo "Building Rust static library..."
	@cargo build --release --features duckdb_extension --manifest-path infera/Cargo.toml

.PHONY: rust-binding-headers
rust-binding-headers: ## Generate Rust binding headers
	@echo "Generating Rust binding headers..."
	@cd infera && cbindgen --config ./cbindgen.toml --crate infera --output ../bindings/include/rust.h

.PHONY: build-ext
build-ext: rust-lib rust-binding-headers ## Build the DuckDB extension (Rust + headers)
	@echo "DuckDB extension components built successfully"

.PHONY: rust-clean
rust-clean: ## Remove Rust generated and temporary files
	@echo "Cleaning Rust artifacts..."
	@cargo clean --manifest-path infera/Cargo.toml
	@rm -rf build/
	@rm -f bindings/include/rust.h

###########################################
# Development Dependencies
###########################################

.PHONY: install-deps
install-deps: ## Install development dependencies
	@echo "Installing development dependencies..."
	@sudo apt-get update
	@sudo apt-get install -y cmake build-essential cbindgen
	@rustup component add rustfmt clippy
	@cargo install cargo-audit

###########################################
# Testing Targets
###########################################

.PHONY: test-extension
test-extension: ## Test the built extension
	@echo "Testing DuckDB extension..."
	@if [ -f "./build/release/duckdb" ]; then \
		echo "Running extension tests..."; \
		./$(TESTS_DIR)/test_extension.sh; \
	else \
		echo "Extension not built. Run 'make release' first."; \
		exit 1; \
	fi

.PHONY: test-quick
test-quick: ## Quick test of the extension function
	@echo "Quick extension test..."
	@if [ -f "./build/release/duckdb" ]; then \
		./build/release/duckdb -c "SELECT hello_infera() as greeting;"; \
	else \
		echo "Extension not built. Run 'make release' first."; \
		exit 1; \
	fi

.PHONY: test-sql
test-sql: ## Run SQL tests from file
	@echo "Running SQL tests..."
	@if [ -f "./build/release/duckdb" ]; then \
		./build/release/duckdb < $(TESTS_DIR)/test_extension.sql; \
	else \
		echo "Extension not built. Run 'make release' first."; \
		exit 1; \
	fi

.PHONY: test-python
test-python: ## Run Python test suite
	@echo "Running Python test suite..."
	@python3 $(TESTS_DIR)/test_extension.py
