//! Host function bindings — safe Rust wrappers around the raw `extern "C"` imports.
//!
//! These are the functions the MinnsDB runtime makes available to every module.
//! Data exchange pattern:
//!   1. Module calls host function → returns status code (0 = ok, -1 = error)
//!   2. Module calls `result_len()` → byte length of result
//!   3. Module calls `alloc(len)` → WASM memory pointer
//!   4. Module calls `result_read(ptr, len)` → host copies result into WASM memory
//!   5. Module deserializes the MessagePack bytes

use crate::types::*;

// ---------------------------------------------------------------------------
// Raw FFI imports (env namespace, provided by MinnsDB runtime)
// ---------------------------------------------------------------------------

extern "C" {
    fn minns_log(level: i32, msg_ptr: i32, msg_len: i32);
    fn result_len() -> i32;
    fn result_read(dst_ptr: i32, max_len: i32) -> i32;
    fn module_id() -> i64;
    fn group_id() -> i64;
    fn table_get(table_ptr: i32, table_len: i32, row_id: i64, result_ptr: i32) -> i32;
    fn table_insert(table_ptr: i32, table_len: i32, row_ptr: i32, row_len: i32) -> i32;
    fn table_delete(table_ptr: i32, table_len: i32, row_id: i64) -> i32;
    fn table_query(query_ptr: i32, query_len: i32, result_ptr: i32) -> i32;
    fn graph_query(query_ptr: i32, query_len: i32, result_ptr: i32) -> i32;
    fn http_fetch(req_ptr: i32, req_len: i32, result_ptr: i32) -> i32;
}

// ---------------------------------------------------------------------------
// Internal: read the host's last_result into WASM memory
// ---------------------------------------------------------------------------

fn read_host_result() -> Vec<u8> {
    let len = unsafe { result_len() };
    if len <= 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; len as usize];
    let copied = unsafe { result_read(buf.as_mut_ptr() as i32, len) };
    if copied < 0 {
        return Vec::new();
    }
    buf.truncate(copied as usize);
    buf
}

fn decode_result_or_err<T: serde::de::DeserializeOwned>(code: i32) -> Result<T, String> {
    let buf = read_host_result();
    if code < 0 {
        let err: String =
            rmp_serde::from_slice(&buf).unwrap_or_else(|_| "unknown host error".into());
        return Err(err);
    }
    rmp_serde::from_slice(&buf).map_err(|e| format!("decode error: {}", e))
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

pub const LOG_TRACE: i32 = 0;
pub const LOG_DEBUG: i32 = 1;
pub const LOG_INFO: i32 = 2;
pub const LOG_WARN: i32 = 3;
pub const LOG_ERROR: i32 = 4;

/// Write a log message at the given level.
pub fn log_msg(level: i32, msg: &str) {
    unsafe { minns_log(level, msg.as_ptr() as i32, msg.len() as i32) }
}

pub fn info(msg: &str) {
    log_msg(LOG_INFO, msg);
}
pub fn warn(msg: &str) {
    log_msg(LOG_WARN, msg);
}
pub fn error(msg: &str) {
    log_msg(LOG_ERROR, msg);
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

pub fn get_module_id() -> u64 {
    unsafe { module_id() as u64 }
}

pub fn get_group_id() -> u64 {
    unsafe { group_id() as u64 }
}

// ---------------------------------------------------------------------------
// Table operations
// ---------------------------------------------------------------------------

/// Get a row by ID from a table.
pub fn table_get_row<T: serde::de::DeserializeOwned>(
    table_name: &str,
    row_id: u64,
) -> Result<T, String> {
    let code = unsafe {
        table_get(
            table_name.as_ptr() as i32,
            table_name.len() as i32,
            row_id as i64,
            0,
        )
    };
    decode_result_or_err(code)
}

/// Insert a row into a table. Returns (row_id, version_id).
pub fn table_insert_row<T: serde::Serialize>(
    table_name: &str,
    values: &T,
) -> Result<InsertResult, String> {
    let row_bytes = rmp_serde::to_vec(values).map_err(|e| format!("encode error: {}", e))?;
    let code = unsafe {
        table_insert(
            table_name.as_ptr() as i32,
            table_name.len() as i32,
            row_bytes.as_ptr() as i32,
            row_bytes.len() as i32,
        )
    };
    decode_result_or_err(code)
}

/// Delete a row by ID. Returns the new version ID.
pub fn table_delete_row(table_name: &str, row_id: u64) -> Result<u64, String> {
    let code = unsafe {
        table_delete(
            table_name.as_ptr() as i32,
            table_name.len() as i32,
            row_id as i64,
        )
    };
    decode_result_or_err(code)
}

/// Execute a MinnsQL table query (FROM/WHERE/RETURN).
pub fn table_query_exec(query: &str) -> Result<QueryResult, String> {
    let code = unsafe { table_query(query.as_ptr() as i32, query.len() as i32, 0) };
    decode_result_or_err(code)
}

// ---------------------------------------------------------------------------
// Graph operations
// ---------------------------------------------------------------------------

/// Execute a MinnsQL graph query (MATCH/RETURN).
pub fn graph_query_exec(query: &str) -> Result<QueryResult, String> {
    let code = unsafe { graph_query(query.as_ptr() as i32, query.len() as i32, 0) };
    decode_result_or_err(code)
}

// ---------------------------------------------------------------------------
// HTTP (sandboxed)
// ---------------------------------------------------------------------------

/// Make an HTTP request. Only allowed domains per module permissions.
pub fn http_fetch_exec(request: &HttpRequest) -> Result<HttpResponse, String> {
    let req_bytes = rmp_serde::to_vec(request).map_err(|e| format!("encode error: {}", e))?;
    let code = unsafe { http_fetch(req_bytes.as_ptr() as i32, req_bytes.len() as i32, 0) };
    decode_result_or_err(code)
}
