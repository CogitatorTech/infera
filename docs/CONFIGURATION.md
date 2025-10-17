## Infera's Configuration Guide

Infera supports configuration via environment variables to customize its behavior without code changes.

### Environment Variables

#### Cache Configuration

##### INFERA_CACHE_DIR

- **Description**: Directory path for caching remote models
- **Type**: String (path)
- **Default**: `$TMPDIR/infera_cache` (system temp directory)
- **Example**:
  ```bash
  export INFERA_CACHE_DIR="/var/cache/infera"
  ```

##### INFERA_CACHE_SIZE_LIMIT

- **Description**: Maximum cache size in bytes
- **Type**: Integer (bytes)
- **Default**: `1073741824` (1GB)
- **Example**:
  ```bash
  ## Set to 5GB
  export INFERA_CACHE_SIZE_LIMIT=5368709120
  
  ## Set to 500MB
  export INFERA_CACHE_SIZE_LIMIT=524288000
  ```

##### INFERA_CACHE_EVICTION

- **Description**: Cache eviction strategy to use when cache is full
- **Type**: String (`LRU`, `LFU`, `FIFO`)
- **Default**: `LRU` (Least Recently Used)
- **Example**:
  ```bash
  export INFERA_CACHE_EVICTION=LRU
  ## Note: Currently only LRU is implemented, LFU and FIFO are planned
  ```

#### HTTP Configuration

##### INFERA_HTTP_TIMEOUT

- **Description**: HTTP request timeout in seconds for downloading remote models
- **Type**: Integer (seconds)
- **Default**: `30`
- **Example**:
  ```bash
  export INFERA_HTTP_TIMEOUT=60
  ```

##### INFERA_HTTP_RETRY_ATTEMPTS

- **Description**: Number of retry attempts for failed downloads
- **Type**: Integer
- **Default**: `3`
- **Example**:
  ```bash
  ## Retry up to 5 times on failure
  export INFERA_HTTP_RETRY_ATTEMPTS=5
  ```

##### INFERA_HTTP_RETRY_DELAY

- **Description**: Initial delay between retry attempts in milliseconds (uses exponential backoff)
- **Type**: Integer (milliseconds)
- **Default**: `1000` (1 second)
- **Example**:
  ```bash
  ## Wait 2 seconds between retries
  export INFERA_HTTP_RETRY_DELAY=2000
  ```

#### Logging Configuration

##### INFERA_VERBOSE

- **Description**: Enable verbose logging (deprecated, use INFERA_LOG_LEVEL instead)
- **Type**: Boolean (`1`, `true`, or `0`, `false`)
- **Default**: `false`
- **Example**:
  ```bash
  export INFERA_VERBOSE=1
  ```

##### INFERA_LOG_LEVEL

- **Description**: Set logging level for detailed output
- **Type**: String (`ERROR`, `WARN`, `INFO`, `DEBUG`)
- **Default**: `WARN`
- **Example**:
  ```bash
  ## Show all messages including debug
  export INFERA_LOG_LEVEL=DEBUG
  
  ## Show only errors
  export INFERA_LOG_LEVEL=ERROR
  
  ## Show informational messages and above
  export INFERA_LOG_LEVEL=INFO
  ```

### Usage Examples

#### Example 1: Custom Cache Directory

```bash
## Set custom cache directory
export INFERA_CACHE_DIR="/mnt/fast-ssd/ml-cache"

## Start DuckDB
./build/release/duckdb

## Check configuration
SELECT infera_get_version();
SELECT infera_get_cache_info();
```

#### Example 2: Larger Cache for Big Models

```bash
## Set cache to 10GB for large models
export INFERA_CACHE_SIZE_LIMIT=10737418240

## Load large models from remote URLs
./build/release/duckdb
```

#### Example 3: Production Configuration

```bash
## Complete production configuration
export INFERA_CACHE_DIR="/var/lib/infera/cache"
export INFERA_CACHE_SIZE_LIMIT=5368709120  ## 5GB
export INFERA_HTTP_TIMEOUT=120             ## 2 minutes
export INFERA_HTTP_RETRY_ATTEMPTS=5        ## Retry up to 5 times
export INFERA_HTTP_RETRY_DELAY=2000        ## 2 second initial delay
export INFERA_LOG_LEVEL=WARN               ## Production logging
export INFERA_CACHE_EVICTION=LRU           ## LRU cache strategy

## Run DuckDB with Infera
./build/release/duckdb
```

#### Example 4: Development/Debug Configuration

```bash
## Development setup with verbose logging
export INFERA_CACHE_DIR="./dev-cache"
export INFERA_LOG_LEVEL=DEBUG              ## Detailed debug logs
export INFERA_HTTP_TIMEOUT=10              ## Shorter timeout for dev
export INFERA_HTTP_RETRY_ATTEMPTS=1        ## Fail fast in development

## Run DuckDB
./build/release/duckdb
```

#### Example 5: Slow Network Configuration

```bash
## Configuration for slow or unreliable networks
export INFERA_HTTP_TIMEOUT=300             ## 5 minute timeout
export INFERA_HTTP_RETRY_ATTEMPTS=10       ## Many retries
export INFERA_HTTP_RETRY_DELAY=5000        ## 5 second initial delay
export INFERA_LOG_LEVEL=INFO               ## Track download progress

./build/release/duckdb
```

### Configuration Verification

You can verify your configuration at runtime:

```sql
-- Check version and cache directory
SELECT infera_get_version();

-- Check cache statistics
SELECT infera_get_cache_info();
```

Example output:

```json
{
    "cache_dir": "/var/cache/infera",
    "total_size_bytes": 204800,
    "file_count": 3,
    "size_limit_bytes": 5368709120
}
```

### Retry Policy Details

When downloading remote models, Infera automatically retries failed downloads with exponential backoff:

1. **Attempt 1**: Download immediately
2. **Attempt 2**: Wait `INFERA_HTTP_RETRY_DELAY` milliseconds (e.g., 1 second)
3. **Attempt 3**: Wait `INFERA_HTTP_RETRY_DELAY * 2` milliseconds (e.g., 2 seconds)
4. **Attempt N**: Wait `INFERA_HTTP_RETRY_DELAY * N` milliseconds

This helps handle temporary network issues, server rate limiting, and transient failures.

### Logging Levels

Logging levels control the verbosity of output to stderr:

- **ERROR**: Only critical errors that prevent operations
- **WARN**: Warnings about potential issues (default)
- **INFO**: Informational messages about operations (cache hits/misses, downloads)
- **DEBUG**: Detailed debugging information (retry attempts, file sizes, etc.)

Example log output with `INFERA_LOG_LEVEL=INFO`:

```
[INFO] Cache miss for URL: https://example.com/model.onnx, downloading...
[INFO] Successfully downloaded: https://example.com/model.onnx
[INFO] Cache hit for URL: https://example.com/model.onnx
```

Example log output with `INFERA_LOG_LEVEL=DEBUG`:

```
[DEBUG] Download attempt 1/3 for https://example.com/model.onnx
[INFO] Successfully downloaded: https://example.com/model.onnx
[DEBUG] Downloaded file size: 15728640 bytes
```

### Cache Eviction Strategies

Currently implemented:

- **LRU (Least Recently Used)**: Evicts files that haven't been accessed in the longest time

Planned for future releases:

- **LFU (Least Frequently Used)**: Evicts files with the lowest access count
- **FIFO (First In First Out)**: Evicts oldest downloaded files first

### Notes

- Environment variables are read once when Infera initializes
- Changes to environment variables require restarting DuckDB
- Invalid values fall back to defaults (no errors thrown)
- Cache directory is created automatically if it doesn't exist
- LRU eviction happens automatically when cache limit is reached
- Logging output goes to stderr and doesn't interfere with SQL query results
- Retry delays use exponential backoff to handle rate limiting gracefully
