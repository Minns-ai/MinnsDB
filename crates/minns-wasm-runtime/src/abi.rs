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

/// Pool of host-side buffers for zero-copy data exchange.
/// Instead of copying large results into WASM linear memory, host functions
/// can write to a buffer and return a buffer ID. The module then reads
/// incrementally via buffer_read.
#[derive(Debug, Default)]
pub struct BufferPool {
    buffers: Vec<Vec<u8>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new buffer with the given data. Returns buffer ID.
    pub fn alloc(&mut self, data: Vec<u8>) -> u32 {
        let id = self.buffers.len() as u32;
        self.buffers.push(data);
        id
    }

    /// Get the length of a buffer.
    pub fn len(&self, id: u32) -> Option<usize> {
        self.buffers.get(id as usize).map(|b| b.len())
    }

    /// Read bytes from a buffer at an offset.
    pub fn read(&self, id: u32, offset: usize, len: usize) -> Option<&[u8]> {
        let buf = self.buffers.get(id as usize)?;
        if offset + len > buf.len() {
            return None;
        }
        Some(&buf[offset..offset + len])
    }

    /// Write bytes into a buffer (append or overwrite).
    pub fn write(&mut self, id: u32, data: &[u8]) -> bool {
        if let Some(buf) = self.buffers.get_mut(id as usize) {
            buf.extend_from_slice(data);
            true
        } else {
            false
        }
    }

    /// Clear all buffers (between calls).
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Number of active buffers.
    pub fn count(&self) -> usize {
        self.buffers.len()
    }
}
