//! WasmRuntime: compile, instantiate, and call WASM modules.

use wasmtime::{Config, Engine, Module};

use crate::error::WasmError;

/// Default life budget per call (~1 minute of WASM runtime).
pub const DEFAULT_LIFE_BUDGET: u64 = 120_000_000_000_000;

/// Epoch tick interval in milliseconds.
const EPOCH_TICK_MS: u64 = 10;

/// Epoch ticks per second (100 ticks at 10ms each).
pub const EPOCH_TICKS_PER_SECOND: u64 = 1000 / EPOCH_TICK_MS;

/// Maximum linear memory (64MB = 1024 pages of 64KB each).
const MAX_MEMORY_PAGES: u64 = 1024;

/// Configuration for the WASM runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Life budget per call (instruction count). Default: ~1 minute.
    pub default_life_budget: u64,
    /// Wall-time limit per call in seconds. Default: 30.
    pub wall_time_limit_secs: u64,
    /// Max linear memory in 64KB pages. Default: 1024 (64MB).
    pub max_memory_pages: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            default_life_budget: DEFAULT_LIFE_BUDGET,
            wall_time_limit_secs: 30,
            max_memory_pages: MAX_MEMORY_PAGES,
        }
    }
}

/// The WASM runtime engine. Shared across all module instances.
pub struct WasmRuntime {
    engine: Engine,
    config: RuntimeConfig,
    /// Handle for the epoch ticker task. Aborted on drop.
    epoch_ticker: tokio::task::JoinHandle<()>,
}

impl Drop for WasmRuntime {
    fn drop(&mut self) {
        self.epoch_ticker.abort();
    }
}

impl WasmRuntime {
    /// Create a new WASM runtime with the given config.
    pub fn new(config: RuntimeConfig) -> Result<Self, WasmError> {
        let mut engine_config = Config::new();
        // Enable instruction metering (life budget per call).
        engine_config
            .consume_fuel(true)
            .epoch_interruption(true)
            .wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable);

        let engine = Engine::new(&engine_config)
            .map_err(|e| WasmError::CompilationError(format!("engine init: {}", e)))?;

        // Spawn epoch ticker — increments the engine epoch every 10ms.
        let engine_clone = engine.clone();
        let epoch_ticker = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(EPOCH_TICK_MS));
            loop {
                interval.tick().await;
                engine_clone.increment_epoch();
            }
        });

        Ok(WasmRuntime {
            engine,
            config,
            epoch_ticker,
        })
    }

    /// Compile WASM bytes into a Module.
    pub fn compile(&self, wasm_bytes: &[u8]) -> Result<Module, WasmError> {
        Module::new(&self.engine, wasm_bytes)
            .map_err(|e| WasmError::CompilationError(e.to_string()))
    }

    /// Get a reference to the engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the runtime config.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Compute the blake3 hash of WASM bytes for content-addressed storage.
    pub fn hash_module(wasm_bytes: &[u8]) -> [u8; 32] {
        *blake3::hash(wasm_bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let rt = WasmRuntime::new(RuntimeConfig::default()).unwrap();
        assert_eq!(rt.config().default_life_budget, DEFAULT_LIFE_BUDGET);
    }

    #[tokio::test]
    async fn test_hash_module() {
        let bytes = b"test wasm bytes";
        let hash = WasmRuntime::hash_module(bytes);
        assert_eq!(hash.len(), 32);
        // Deterministic
        assert_eq!(hash, WasmRuntime::hash_module(bytes));
    }
}
