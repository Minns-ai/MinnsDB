/// Error categories for instance lifecycle management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// User error (permission denied, bad args): instance is reusable.
    User,
    /// Recoverable error (life exceeded, timeout): instance reusable after reset.
    Recoverable,
    /// Fatal trap (WASM trap, stack overflow, OOB): instance must be discarded.
    Trap,
}

#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("compilation error: {0}")]
    CompilationError(String),
    #[error("instantiation error: {0}")]
    InstantiationError(String),
    #[error("execution error: {0}")]
    ExecutionError(String),
    #[error("life exceeded: module ran out of instruction budget")]
    LifeExceeded,
    #[error("timeout: module exceeded wall-time limit")]
    Timeout,
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("module not found: {0}")]
    ModuleNotFound(String),
    #[error("module already exists: {0}")]
    ModuleAlreadyExists(String),
    #[error("invalid module: {0}")]
    InvalidModule(String),
    #[error("ABI error: {0}")]
    AbiError(String),
    #[error("host function error: {0}")]
    HostError(String),
    #[error("persistence error: {0}")]
    PersistenceError(String),
    #[error("schedule error: {0}")]
    ScheduleError(String),
}

impl WasmError {
    /// Classify this error for instance lifecycle decisions.
    pub fn category(&self) -> ErrorCategory {
        match self {
            WasmError::PermissionDenied(_)
            | WasmError::ModuleNotFound(_)
            | WasmError::ModuleAlreadyExists(_)
            | WasmError::InvalidModule(_)
            | WasmError::AbiError(_)
            | WasmError::HostError(_)
            | WasmError::PersistenceError(_)
            | WasmError::ScheduleError(_) => ErrorCategory::User,

            WasmError::LifeExceeded | WasmError::Timeout => ErrorCategory::Recoverable,

            WasmError::CompilationError(_)
            | WasmError::InstantiationError(_)
            | WasmError::ExecutionError(_) => ErrorCategory::Trap,
        }
    }
}

impl From<wasmtime::Error> for WasmError {
    fn from(e: wasmtime::Error) -> Self {
        let msg = e.to_string();
        // Detect life budget exhaustion
        if msg.contains("fuel") || msg.contains("all fuel consumed") {
            return WasmError::LifeExceeded;
        }
        // Detect epoch-based timeout
        if msg.contains("epoch") {
            return WasmError::Timeout;
        }
        // Detect WASM traps via downcast
        if e.downcast_ref::<wasmtime::Trap>().is_some() {
            return WasmError::ExecutionError(format!("trap: {}", msg));
        }
        WasmError::ExecutionError(msg)
    }
}
