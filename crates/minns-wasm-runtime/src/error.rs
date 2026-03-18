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

impl From<wasmtime::Error> for WasmError {
    fn from(e: wasmtime::Error) -> Self {
        let msg = e.to_string();
        // Detect life budget exhaustion from wasmtime error messages
        if msg.contains("fuel") || msg.contains("all fuel consumed") {
            WasmError::LifeExceeded
        } else if msg.contains("epoch") {
            WasmError::Timeout
        } else {
            WasmError::ExecutionError(msg)
        }
    }
}
