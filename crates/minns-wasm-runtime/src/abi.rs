//! ABI types and MessagePack memory exchange helpers for WASM↔host communication.

use serde::{Deserialize, Serialize};
use wasmtime::Memory;

use crate::error::WasmError;

/// Maximum size for data crossing the WASM boundary (1MB).
pub const MAX_EXCHANGE_SIZE: usize = 1024 * 1024;

/// Module descriptor returned by __minns_describe__.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDescriptor {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub functions: Vec<FunctionDef>,
    #[serde(default)]
    pub triggers: Vec<TriggerDef>,
    #[serde(default)]
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerDef {
    #[serde(rename = "type")]
    pub trigger_type: String,
    #[serde(default)]
    pub table: Option<String>,
    #[serde(default)]
    pub edge_type: Option<String>,
    #[serde(default)]
    pub node_type: Option<String>,
    #[serde(default)]
    pub cron: Option<String>,
    pub function: String,
}

/// Result of a module function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallResult {
    pub success: bool,
    #[serde(default)]
    pub data: Option<Vec<u8>>, // raw MessagePack bytes for the result payload
    #[serde(default)]
    pub error: Option<String>,
}

/// Write MessagePack bytes into WASM linear memory.
/// Returns the pointer and length written.
pub fn write_to_wasm<T: wasmtime::AsContextMut>(
    memory: &Memory,
    store: &mut T,
    alloc_fn: &wasmtime::TypedFunc<i32, i32>,
    data: &[u8],
) -> Result<(i32, i32), WasmError> {
    if data.len() > MAX_EXCHANGE_SIZE {
        return Err(WasmError::AbiError(format!(
            "data exceeds max exchange size: {} > {}",
            data.len(),
            MAX_EXCHANGE_SIZE
        )));
    }

    let len = data.len() as i32;
    let ptr = alloc_fn
        .call(&mut *store, len)
        .map_err(|e| WasmError::AbiError(format!("alloc failed: {}", e)))?;

    if ptr < 0 {
        return Err(WasmError::AbiError(
            "alloc returned negative pointer".into(),
        ));
    }

    memory
        .write(&mut *store, ptr as usize, data)
        .map_err(|e| WasmError::AbiError(format!("memory write failed: {}", e)))?;

    Ok((ptr, len))
}

/// Read bytes from WASM linear memory at the given pointer and length.
pub fn read_from_wasm<T: wasmtime::AsContext>(
    memory: &Memory,
    store: &T,
    ptr: i32,
    len: i32,
) -> Result<Vec<u8>, WasmError> {
    if ptr < 0 || len < 0 {
        return Err(WasmError::AbiError(format!(
            "invalid pointer/length: ptr={}, len={}",
            ptr, len
        )));
    }
    if len as usize > MAX_EXCHANGE_SIZE {
        return Err(WasmError::AbiError(format!(
            "read exceeds max exchange size: {} > {}",
            len, MAX_EXCHANGE_SIZE
        )));
    }

    let mut buf = vec![0u8; len as usize];
    memory
        .read(store, ptr as usize, &mut buf)
        .map_err(|e| WasmError::AbiError(format!("memory read failed: {}", e)))?;

    Ok(buf)
}

/// Serialise a value to MessagePack bytes.
pub fn to_msgpack<T: Serialize>(value: &T) -> Result<Vec<u8>, WasmError> {
    rmp_serde::to_vec(value).map_err(|e| WasmError::AbiError(format!("msgpack encode: {}", e)))
}

/// Deserialise MessagePack bytes to a value.
pub fn from_msgpack<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, WasmError> {
    rmp_serde::from_slice(data).map_err(|e| WasmError::AbiError(format!("msgpack decode: {}", e)))
}
