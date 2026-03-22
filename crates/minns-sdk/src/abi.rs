//! ABI glue: the exports that the MinnsDB runtime expects from every module.
//!
//! This module provides helpers so you don't have to write raw `#[no_mangle]`
//! extern functions yourself. Use `register_descriptor` to set up your module's
//! metadata, and `set_result` to return data from function calls.
//!
//! # Example module skeleton
//!
//! ```rust,ignore
//! use minns_sdk::prelude::*;
//!
//! static DESCRIPTOR: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
//!
//! #[no_mangle]
//! pub extern "C" fn __minns_describe__() -> i32 {
//!     let desc = ModuleDescriptor {
//!         name: "my_module".into(),
//!         version: "0.1.0".into(),
//!         functions: vec![FunctionDef { name: "hello".into(), description: "says hello".into() }],
//!         triggers: vec![],
//!         permissions: vec![],
//!     };
//!     register_descriptor(&desc)
//! }
//!
//! #[no_mangle]
//! pub extern "C" fn __minns_describe_len__() -> i32 { descriptor_len() }
//!
//! #[no_mangle]
//! pub extern "C" fn __minns_call__(func_id: i32, args_ptr: i32, args_len: i32) -> i32 {
//!     match func_id {
//!         0 => { set_result(&"world"); 0 }
//!         _ => -1,
//!     }
//! }
//! ```

use crate::types::ModuleDescriptor;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Allocator export (required by the runtime for host→guest data transfer)
// ---------------------------------------------------------------------------

/// Allocate `size` bytes of WASM linear memory. The runtime calls this to
/// write arguments into the module's memory before invoking functions.
#[no_mangle]
pub extern "C" fn alloc(size: i32) -> i32 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as i32 }
}

// ---------------------------------------------------------------------------
// Descriptor storage (thread-safe via Mutex — WASM is single-threaded anyway)
// ---------------------------------------------------------------------------

static DESCRIPTOR_BYTES: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Serialize and store the module descriptor. Call this from your
/// `__minns_describe__` export. Returns the pointer to the MessagePack bytes.
pub fn register_descriptor(desc: &ModuleDescriptor) -> i32 {
    let bytes = rmp_serde::to_vec(desc).expect("failed to serialize descriptor");
    let mut guard = DESCRIPTOR_BYTES.lock().unwrap();
    *guard = bytes;
    guard.as_ptr() as i32
}

/// Returns the byte length of the registered descriptor.
/// Call this from your `__minns_describe_len__` export.
pub fn descriptor_len() -> i32 {
    let guard = DESCRIPTOR_BYTES.lock().unwrap();
    guard.len() as i32
}

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

/// Serialize a value to MessagePack and push it to the host as the function result.
/// Call this at the end of your `__minns_call__` handler.
pub fn set_result<T: serde::Serialize>(value: &T) {
    let bytes = rmp_serde::to_vec(value).expect("failed to serialize result");
    set_result_bytes(&bytes);
}

/// Push raw bytes as the function result.
pub fn set_result_bytes(bytes: &[u8]) {
    extern "C" {
        fn result_write(ptr: i32, len: i32);
    }
    unsafe { result_write(bytes.as_ptr() as i32, bytes.len() as i32) }
}

// ---------------------------------------------------------------------------
// Argument deserialization
// ---------------------------------------------------------------------------

/// Deserialize MessagePack function arguments from the pointer/length
/// passed to `__minns_call__`.
pub fn read_args<T: serde::de::DeserializeOwned>(ptr: i32, len: i32) -> Result<T, String> {
    if ptr <= 0 || len <= 0 {
        return Err("invalid argument pointer/length".into());
    }
    let slice = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };
    rmp_serde::from_slice(slice).map_err(|e| format!("failed to decode args: {}", e))
}

/// Read raw argument bytes without deserializing.
pub fn read_args_raw(ptr: i32, len: i32) -> Vec<u8> {
    if ptr <= 0 || len <= 0 {
        return Vec::new();
    }
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec() }
}
