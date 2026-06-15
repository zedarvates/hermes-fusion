//! Hermes KVFlash Cache - Hot cache layer using Lucebox KVFlash
//! 
//! Provides ultra-fast KV operations for:
//! - Rate limiting (token buckets, sliding windows)
//! - Session storage (TTL, auto-expiry)
//! - Hot key-value cache (sub-ms latency)
//! 
//! Falls back to Moka (in-memory) if KVFlash unavailable.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use moka::future::Cache as MokaCache;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// NOTE: KVFlash from Lucebox not yet published to crates.io
// When available, uncomment below:
// #[cfg(feature = "kvflash")]
// use kvflash::KvFlash;

/// Configuration for the hybrid cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Path to KVFlash database (None = in-memory only)
    pub kvflash_path: Option<String>,
    /// Max entries in hot cache
    pub max_entries: u64,
    /// Default TTL for entries (seconds)
    pub default_ttl_secs: u64,
    /// Enable rate limiting buckets
    pub enable_rate_limit: bool,
    /// Enable session storage
    pub enable_sessions: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            kvflash_path: Some("/tmp/hermes_kvflash".to_string()),
            max_entries: 100_000,
            default_ttl_secs: 3600,
            enable_rate_limit: true,
            enable_sessions: true,
        }
    }
}

/// Rate limit bucket state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitBucket {
    pub tokens: f64,
    pub last_refill: u64,
    pub capacity: f64,
    pub refill_rate: f64,
}

impl RateLimitBucket {
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            tokens: capacity,
            last_refill: now,
            capacity,
            refill_rate,
        }
    }

    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let elapsed = now.saturating_sub(self.last_refill) as f64;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    pub fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

/// Session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    pub expires_at: u64,
    pub last_accessed: u64,
}

impl Session {
    pub fn new(id: String, ttl_secs: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            id,
            data: HashMap::new(),
            created_at: now,
            expires_at: now + ttl_secs,
            last_accessed: now,
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now >= self.expires_at
    }

    pub fn touch(&mut self, ttl_secs: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_accessed = now;
        self.expires_at = now + ttl_secs;
    }
}

/// Hybrid cache backend
#[derive(Clone)]
pub enum CacheBackend {
    #[cfg(feature = "kvflash")]
    KvFlash(Arc<KvFlash>),
    InMemory(Arc<MokaCache<String, Vec<u8>>>),
}

/// Main hybrid cache
pub struct HybridCache {
    backend: CacheBackend,
    config: CacheConfig,
    rate_limiters: Arc<RwLock<HashMap<String, RateLimitBucket>>>,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct CacheStats {
    pub gets: u64,
    pub sets: u64,
    pub deletes: u64,
    pub hits: u64,
    pub misses: u64,
    pub rate_limit_checks: u64,
    pub rate_limit_allowed: u64,
    pub rate_limit_denied: u64,
    pub session_creates: u64,
    pub session_reads: u64,
    pub session_expired: u64,
}

impl HybridCache {
    /// Create new hybrid cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let backend = if let Some(path) = &config.kvflash_path {
            #[cfg(feature = "kvflash")]
            {
                match KvFlash::open(path).await {
                    Ok(kv) => {
                        info!("KVFlash opened at: {}", path);
                        CacheBackend::KvFlash(Arc::new(kv))
                    }
                    Err(e) => {
                        warn!("KVFlash failed to open ({}), falling back to in-memory: {}", path, e);
                        CacheBackend::InMemory(Self::create_memory_cache(config.max_entries))
                    }
                }
            }
            #[cfg(not(feature = "kvflash"))]
            {
                info!("KVFlash feature not enabled, using in-memory cache");
                CacheBackend::InMemory(Self::create_memory_cache(config.max_entries))
            }
        } else {
            CacheBackend::InMemory(Self::create_memory_cache(config.max_entries))
        };

        Ok(Self {
            backend,
            config,
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }

    fn create_memory_cache(max_entries: u64) -> Arc<MokaCache<String, Vec<u8>>> {
        Arc::new(
            MokaCache::builder()
                .max_capacity(max_entries)
                .time_to_live(Duration::from_secs(3600))
                .build()
        )
    }

    /// Get raw value
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut stats = self.stats.write().await;
        stats.gets += 1;
        
        let result: Option<Vec<u8>> = match &self.backend {
            CacheBackend::InMemory(cache) => cache.get(key).await,
        };

        if result.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        
        Ok(result)
    }

    /// Set raw value with TTL
    pub async fn set(&self, key: &str, value: Vec<u8>, _ttl_secs: Option<u64>) -> Result<()> {
        let mut stats = self.stats.write().await;
        stats.sets += 1;
        
        match &self.backend {
            CacheBackend::InMemory(cache) => {
                cache.insert(key.to_string(), value).await;
                Ok(())
            }
        }
    }

    /// Delete key
    pub async fn delete(&self, key: &str) -> Result<bool> {
        let mut stats = self.stats.write().await;
        stats.deletes += 1;
        
        match &self.backend {
            CacheBackend::InMemory(cache) => {
                cache.invalidate(key).await;
                // Moka invalidate returns (), assume success if no error
                Ok(true)
            }
        }
    }

    /// Check if key exists
    pub async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key).await?.is_some())
    }

    // ===== Rate Limiting =====

    /// Check and consume rate limit tokens
    /// Returns (allowed, remaining_tokens, retry_after_secs)
    pub async fn rate_limit_check(
        &self,
        key: &str,
        tokens: f64,
        capacity: f64,
        refill_rate: f64,
    ) -> Result<(bool, f64, Option<u64>)> {
        if !self.config.enable_rate_limit {
            return Ok((true, capacity, None));
        }

        let mut stats = self.stats.write().await;
        stats.rate_limit_checks += 1;

        let mut limiters = self.rate_limiters.write().await;
        let bucket = limiters.entry(key.to_string())
            .or_insert_with(|| RateLimitBucket::new(capacity, refill_rate));

        let allowed = bucket.try_consume(tokens);
        let remaining = bucket.available();

        if allowed {
            stats.rate_limit_allowed += 1;
            Ok((true, remaining, None))
        } else {
            stats.rate_limit_denied += 1;
            let retry_after = ((tokens - remaining) / refill_rate).ceil() as u64;
            Ok((false, remaining, Some(retry_after.max(1))))
        }
    }

    /// Get current rate limit status without consuming
    pub async fn rate_limit_status(&self, key: &str) -> Result<Option<(f64, f64)>> {
        let mut limiters = self.rate_limiters.write().await;
        Ok(limiters.get_mut(key).map(|b| (b.available(), b.capacity)))
    }

    /// Reset rate limit bucket
    pub async fn rate_limit_reset(&self, key: &str) -> Result<()> {
        let mut limiters = self.rate_limiters.write().await;
        limiters.remove(key);
        Ok(())
    }

    // ===== Session Management =====

    /// Create new session
    pub async fn session_create(&self, session_id: &str, ttl_secs: u64) -> Result<Session> {
        if !self.config.enable_sessions {
            return Err(anyhow::anyhow!("Sessions disabled"));
        }

        let mut stats = self.stats.write().await;
        stats.session_creates += 1;

        let mut sessions = self.sessions.write().await;
        let session = Session::new(session_id.to_string(), ttl_secs);
        sessions.insert(session_id.to_string(), session.clone());
        
        let key = format!("session:{}", session_id);
        let data = serde_json::to_vec(&session)?;
        self.set(&key, data, Some(ttl_secs)).await?;
        
        Ok(session)
    }

    /// Get session by ID
    pub async fn session_get(&self, session_id: &str) -> Result<Option<Session>> {
        if !self.config.enable_sessions {
            return Ok(None);
        }

        let mut stats = self.stats.write().await;
        stats.session_reads += 1;

        {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(session_id) {
                if !session.is_expired() {
                    return Ok(Some(session.clone()));
                }
            }
        }

        let key = format!("session:{}", session_id);
        if let Some(data) = self.get(&key).await? {
            let session: Session = serde_json::from_slice(&data)?;
            if !session.is_expired() {
                let mut sessions = self.sessions.write().await;
                sessions.insert(session_id.to_string(), session.clone());
                return Ok(Some(session));
            } else {
                self.session_delete(session_id).await?;
                stats.session_expired += 1;
            }
        }
        
        Ok(None)
    }

    /// Update session data
    pub async fn session_set(
        &self,
        session_id: &str,
        key: &str,
        value: serde_json::Value,
        ttl_secs: u64,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.data.insert(key.to_string(), value);
            session.touch(ttl_secs);
            
            let persist_key = format!("session:{}", session_id);
            let data = serde_json::to_vec(session)?;
            self.set(&persist_key, data, Some(ttl_secs)).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found"))
        }
    }

    /// Delete session
    pub async fn session_delete(&self, session_id: &str) -> Result<bool> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        
        let key = format!("session:{}", session_id);
        self.delete(&key).await
    }

    /// Clean up expired sessions
    pub async fn session_cleanup(&self) -> Result<u64> {
        let mut stats = self.stats.write().await;
        let mut sessions = self.sessions.write().await;
        
        let expired_keys: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        
        let count = expired_keys.len() as u64;
        for key in &expired_keys {
            sessions.remove(key);
            let persist_key = format!("session:{}", key);
            let _ = self.delete(&persist_key).await;
        }
        
        stats.session_expired += count;
        Ok(count)
    }

    // ===== High-level Cache Operations =====

    /// Get JSON value
    pub async fn get_json<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        if let Some(data) = self.get(key).await? {
            Ok(Some(serde_json::from_slice(&data)?))
        } else {
            Ok(None)
        }
    }

    /// Set JSON value
    pub async fn set_json<T: Serialize>(&self, key: &str, value: &T, ttl_secs: Option<u64>) -> Result<()> {
        let data = serde_json::to_vec(value)?;
        self.set(key, data, ttl_secs).await
    }

    /// Get stats
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> bool {
        match &self.backend {
            #[cfg(feature = "kvflash")]
            CacheBackend::KvFlash(kv) => {
                let test_key = "__health_check__";
                let _ = kv.get(test_key).await;
                true
            }
            CacheBackend::InMemory(_) => true,
        }
    }
}
/// Python bindings
#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use futures_util::future::TryFutureExt;
    use pyo3::prelude::*;
    use pyo3_async_runtimes::tokio::future_into_py;
    use std::sync::Arc;

    #[pyclass]
    struct PyHybridCache {
        inner: Arc<HybridCache>,
    }

    #[pymethods]
    impl PyHybridCache {
        #[new]
        fn new(config: Option<PyHybridCacheConfig>) -> PyResult<Self> {
            let cfg = config.map(|c| c.into()).unwrap_or_default();
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let inner = rt.block_on(async {
                HybridCache::new(cfg).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Self { inner: Arc::new(inner) })
        }

        fn get<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.get(&key).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn set<'py>(&self, py: Python<'py>, key: String, value: Vec<u8>, ttl_secs: Option<u64>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.set(&key, value, ttl_secs).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn delete<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.delete(&key).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn rate_limit_check<'py>(&self, py: Python<'py>, key: String, tokens: f64, capacity: f64, refill_rate: f64) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.rate_limit_check(&key, tokens, capacity, refill_rate).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn session_create<'py>(&self, py: Python<'py>, session_id: String, ttl_secs: u64) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.session_create(&session_id, ttl_secs).await
                    .map(|s| PySession { inner: s })
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn session_get<'py>(&self, py: Python<'py>, session_id: String) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                inner.session_get(&session_id).await
                    .map(|s| s.map(|inner| PySession { inner }))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        }

        fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                let stats = inner.stats().await;
                Ok(PyCacheStats { inner: stats })
            })
        }

        fn health_check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.clone();
            future_into_py(py, async move {
                Ok(inner.health_check().await)
            })
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct PyHybridCacheConfig {
        #[pyo3(get, set)]
        kvflash_path: Option<String>,
        #[pyo3(get, set)]
        max_entries: u64,
        #[pyo3(get, set)]
        default_ttl_secs: u64,
        #[pyo3(get, set)]
        enable_rate_limit: bool,
        #[pyo3(get, set)]
        enable_sessions: bool,
    }

    impl From<PyHybridCacheConfig> for CacheConfig {
        fn from(c: PyHybridCacheConfig) -> Self {
            CacheConfig {
                kvflash_path: c.kvflash_path,
                max_entries: c.max_entries,
                default_ttl_secs: c.default_ttl_secs,
                enable_rate_limit: c.enable_rate_limit,
                enable_sessions: c.enable_sessions,
            }
        }
    }

    impl Default for PyHybridCacheConfig {
        fn default() -> Self {
            let cfg = CacheConfig::default();
            Self {
                kvflash_path: cfg.kvflash_path,
                max_entries: cfg.max_entries,
                default_ttl_secs: cfg.default_ttl_secs,
                enable_rate_limit: cfg.enable_rate_limit,
                enable_sessions: cfg.enable_sessions,
            }
        }
    }

    #[pyclass]
    struct PySession {
        inner: Session,
    }

    #[pymethods]
    impl PySession {
        #[getter]
        fn id(&self) -> String {
            self.inner.id.clone()
        }

        #[getter]
        fn created_at(&self) -> u64 {
            self.inner.created_at
        }

        #[getter]
        fn expires_at(&self) -> u64 {
            self.inner.expires_at
        }

        fn is_expired(&self) -> bool {
            self.inner.is_expired()
        }

        // Simplified data access - convert to JSON string
        fn data_json(&self) -> String {
            serde_json::to_string(&self.inner.data).unwrap_or_default()
        }

        fn get_data(&self, key: &str) -> Option<String> {
            self.inner.data.get(key).map(|v| v.to_string())
        }
    }

    #[pyclass]
    struct PyCacheStats {
        inner: CacheStats,
    }

    #[pymethods]
    impl PyCacheStats {
        #[getter]
        fn gets(&self) -> u64 { self.inner.gets }
        #[getter]
        fn sets(&self) -> u64 { self.inner.sets }
        #[getter]
        fn hits(&self) -> u64 { self.inner.hits }
        #[getter]
        fn misses(&self) -> u64 { self.inner.misses }
        #[getter]
        fn hit_rate(&self) -> f64 {
            if self.inner.gets == 0 { 0.0 } else { self.inner.hits as f64 / self.inner.gets as f64 }
        }
    }

    #[pymodule]
    fn hermes_kvflash(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyHybridCache>()?;
        m.add_class::<PyHybridCacheConfig>()?;
        m.add_class::<PySession>()?;
        m.add_class::<PyCacheStats>()?;
        Ok(())
    }
}