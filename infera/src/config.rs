// src/config.rs
// Centralized configuration management for Infera

use once_cell::sync::Lazy;
use std::env;
use std::path::PathBuf;

/// Default cache size limit: 1GB
const DEFAULT_CACHE_SIZE_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

/// Default cache directory name
const DEFAULT_CACHE_DIR_NAME: &str = "infera_cache";

/// Global configuration singleton
pub static CONFIG: Lazy<InferaConfig> = Lazy::new(InferaConfig::from_env);

/// Logging levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
}

impl LogLevel {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "ERROR" => Some(LogLevel::Error),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "INFO" => Some(LogLevel::Info),
            "DEBUG" => Some(LogLevel::Debug),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
        }
    }
}

/// Cache eviction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum CacheEvictionStrategy {
    /// Least Recently Used - evict oldest accessed files first
    LRU,
    /// Least Frequently Used - evict least accessed files first (future)
    LFU,
    /// First In First Out - evict oldest downloaded files first (future)
    FIFO,
}

impl CacheEvictionStrategy {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "LRU" => Some(CacheEvictionStrategy::LRU),
            "LFU" => Some(CacheEvictionStrategy::LFU),
            "FIFO" => Some(CacheEvictionStrategy::FIFO),
            _ => None,
        }
    }
}

/// Configuration options for Infera
#[derive(Debug, Clone)]
pub struct InferaConfig {
    /// Directory path for caching remote models
    pub cache_dir: PathBuf,

    /// Maximum cache size in bytes
    pub cache_size_limit: u64,

    /// Whether to enable verbose logging
    #[allow(dead_code)]
    pub verbose_logging: bool,

    /// HTTP request timeout in seconds
    pub http_timeout_secs: u64,

    /// Number of retry attempts for failed downloads
    pub http_retry_attempts: u32,

    /// Delay between retry attempts in milliseconds
    pub http_retry_delay_ms: u64,

    /// Cache eviction strategy
    #[allow(dead_code)]
    pub cache_eviction_strategy: CacheEvictionStrategy,

    /// Logging level
    pub log_level: LogLevel,
}

impl InferaConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            cache_dir: Self::get_cache_dir_from_env(),
            cache_size_limit: Self::get_cache_size_limit_from_env(),
            verbose_logging: Self::get_verbose_logging_from_env(),
            http_timeout_secs: Self::get_http_timeout_from_env(),
            http_retry_attempts: Self::get_http_retry_attempts_from_env(),
            http_retry_delay_ms: Self::get_http_retry_delay_from_env(),
            cache_eviction_strategy: Self::get_cache_eviction_strategy_from_env(),
            log_level: Self::get_log_level_from_env(),
        }
    }

    /// Get cache directory from INFERA_CACHE_DIR or default
    fn get_cache_dir_from_env() -> PathBuf {
        env::var("INFERA_CACHE_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| env::temp_dir().join(DEFAULT_CACHE_DIR_NAME))
    }

    /// Get cache size limit from INFERA_CACHE_SIZE_LIMIT or default (1GB)
    fn get_cache_size_limit_from_env() -> u64 {
        env::var("INFERA_CACHE_SIZE_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_CACHE_SIZE_LIMIT_BYTES)
    }

    /// Get verbose logging setting from INFERA_VERBOSE or default (false)
    fn get_verbose_logging_from_env() -> bool {
        env::var("INFERA_VERBOSE")
            .ok()
            .map(|s| s == "1" || s.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Get HTTP timeout from INFERA_HTTP_TIMEOUT or default (30 seconds)
    fn get_http_timeout_from_env() -> u64 {
        env::var("INFERA_HTTP_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30)
    }

    /// Get HTTP retry attempts from INFERA_HTTP_RETRY_ATTEMPTS or default (3)
    fn get_http_retry_attempts_from_env() -> u32 {
        env::var("INFERA_HTTP_RETRY_ATTEMPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
    }

    /// Get HTTP retry delay from INFERA_HTTP_RETRY_DELAY or default (1000ms)
    fn get_http_retry_delay_from_env() -> u64 {
        env::var("INFERA_HTTP_RETRY_DELAY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000)
    }

    /// Get cache eviction strategy from INFERA_CACHE_EVICTION or default (LRU)
    fn get_cache_eviction_strategy_from_env() -> CacheEvictionStrategy {
        env::var("INFERA_CACHE_EVICTION")
            .ok()
            .and_then(|s| CacheEvictionStrategy::from_str(&s))
            .unwrap_or(CacheEvictionStrategy::LRU)
    }

    /// Get log level from INFERA_LOG_LEVEL or default (WARN)
    fn get_log_level_from_env() -> LogLevel {
        env::var("INFERA_LOG_LEVEL")
            .ok()
            .and_then(|s| LogLevel::from_str(&s))
            .unwrap_or(LogLevel::Warn)
    }

    /// Check if a log message should be printed based on current log level
    pub fn should_log(&self, level: LogLevel) -> bool {
        level <= self.log_level
    }
}

impl Default for InferaConfig {
    fn default() -> Self {
        Self {
            cache_dir: env::temp_dir().join(DEFAULT_CACHE_DIR_NAME),
            cache_size_limit: DEFAULT_CACHE_SIZE_LIMIT_BYTES,
            verbose_logging: false,
            http_timeout_secs: 30,
            http_retry_attempts: 3,
            http_retry_delay_ms: 1000,
            cache_eviction_strategy: CacheEvictionStrategy::LRU,
            log_level: LogLevel::Warn,
        }
    }
}

/// Log a message if the log level is enabled
#[macro_export]
macro_rules! log {
    ($level:expr, $($arg:tt)*) => {
        if $crate::config::CONFIG.should_log($level) {
            eprintln!("[{}] {}", $level.as_str(), format!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = InferaConfig::default();
        assert_eq!(config.cache_size_limit, DEFAULT_CACHE_SIZE_LIMIT_BYTES);
        assert!(!config.verbose_logging);
        assert_eq!(config.http_timeout_secs, 30);
        assert_eq!(config.http_retry_attempts, 3);
        assert_eq!(config.http_retry_delay_ms, 1000);
        assert_eq!(config.cache_eviction_strategy, CacheEvictionStrategy::LRU);
        assert_eq!(config.log_level, LogLevel::Warn);
    }

    #[test]
    fn test_cache_dir_ends_with_infera_cache() {
        let config = InferaConfig::default();
        assert!(config
            .cache_dir
            .to_str()
            .unwrap()
            .ends_with(DEFAULT_CACHE_DIR_NAME));
    }

    #[test]
    fn test_log_level_parsing() {
        assert_eq!(LogLevel::from_str("ERROR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_cache_eviction_strategy_parsing() {
        assert_eq!(
            CacheEvictionStrategy::from_str("LRU"),
            Some(CacheEvictionStrategy::LRU)
        );
        assert_eq!(
            CacheEvictionStrategy::from_str("lfu"),
            Some(CacheEvictionStrategy::LFU)
        );
        assert_eq!(
            CacheEvictionStrategy::from_str("FIFO"),
            Some(CacheEvictionStrategy::FIFO)
        );
        assert_eq!(CacheEvictionStrategy::from_str("invalid"), None);
    }

    #[test]
    fn test_should_log() {
        let mut config = InferaConfig::default();
        config.log_level = LogLevel::Info;

        assert!(config.should_log(LogLevel::Error));
        assert!(config.should_log(LogLevel::Warn));
        assert!(config.should_log(LogLevel::Info));
        assert!(!config.should_log(LogLevel::Debug));
    }
}
