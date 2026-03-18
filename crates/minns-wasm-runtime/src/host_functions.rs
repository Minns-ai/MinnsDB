//! Host function implementations exposed to WASM modules.
//!
//! All data crosses the boundary as MessagePack bytes.
//! Each function checks permissions and records usage.

use std::sync::Arc;

use tokio::sync::RwLock;
use wasmtime::{Caller, Linker};

use agent_db_tables::catalog::TableCatalog;
use agent_db_tables::types::CellValue;

use serde::Serialize;

use crate::abi;
use crate::error::WasmError;
use crate::permissions::PermissionSet;
use crate::usage::ModuleUsageCounters;

/// MessagePack-serialisable query result.
#[derive(Serialize)]
struct QueryResultMsg {
    columns: Vec<String>,
    rows: Vec<Vec<MsgpackValue>>,
}

/// A Value converted to a msgpack-friendly enum.
#[derive(Serialize)]
#[serde(untagged)]
enum MsgpackValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<MsgpackValue>),
    Map(std::collections::HashMap<String, MsgpackValue>),
}

fn value_to_msgpack_value(v: agent_db_graph::query_lang::Value) -> MsgpackValue {
    match v {
        agent_db_graph::query_lang::Value::Null => MsgpackValue::Null,
        agent_db_graph::query_lang::Value::Bool(b) => MsgpackValue::Bool(b),
        agent_db_graph::query_lang::Value::Int(i) => MsgpackValue::Int(i),
        agent_db_graph::query_lang::Value::Float(f) => MsgpackValue::Float(f),
        agent_db_graph::query_lang::Value::String(s) => MsgpackValue::String(s),
        agent_db_graph::query_lang::Value::List(l) => {
            MsgpackValue::List(l.into_iter().map(value_to_msgpack_value).collect())
        },
        agent_db_graph::query_lang::Value::Map(m) => MsgpackValue::Map(
            m.into_iter()
                .map(|(k, v)| (k, value_to_msgpack_value(v)))
                .collect(),
        ),
    }
}

/// Environment passed to every host function call via wasmtime Store data.
pub struct HostEnv {
    /// Module's permission set.
    pub permissions: PermissionSet,
    /// Usage counters (shared across calls for the same module).
    pub usage: Arc<ModuleUsageCounters>,
    /// Module's group_id for table operations.
    pub group_id: u64,
    /// Module identifier.
    pub module_id: u64,
    /// Caller identity (set per call).
    pub caller_id: String,
    /// Shared table catalog.
    pub table_catalog: Arc<RwLock<TableCatalog>>,
    /// Last result buffer (host writes here, module reads via result_len + alloc).
    pub last_result: Vec<u8>,
    /// HTTP client for sandboxed fetch (shared, has timeout configured).
    pub http_client: Option<reqwest::Client>,
    /// Store limits for memory enforcement.
    pub limiter: wasmtime::StoreLimits,
}

impl HostEnv {
    pub fn new(
        permissions: PermissionSet,
        usage: Arc<ModuleUsageCounters>,
        group_id: u64,
        module_id: u64,
        table_catalog: Arc<RwLock<TableCatalog>>,
    ) -> Self {
        HostEnv {
            permissions,
            usage,
            group_id,
            module_id,
            caller_id: String::new(),
            table_catalog,
            last_result: Vec::new(),
            http_client: Some(
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(5))
                    .redirect(reqwest::redirect::Policy::none()) // No redirects — prevents SSRF
                    .build()
                    .unwrap_or_default(),
            ),
            limiter: wasmtime::StoreLimitsBuilder::new()
                .memory_size(64 * 1024 * 1024) // 64MB max WASM memory
                .build(),
        }
    }
}

/// Register all host functions with the wasmtime Linker.
pub fn register_host_functions(linker: &mut Linker<HostEnv>) -> Result<(), WasmError> {
    // -- Logging --
    linker
        .func_wrap(
            "minns",
            "log",
            |mut caller: Caller<'_, HostEnv>, level: i32, msg_ptr: i32, msg_len: i32| {
                let memory = caller.get_export("memory").and_then(|e| e.into_memory());
                if let Some(mem) = memory {
                    if let Ok(bytes) = abi::read_from_wasm(&mem, &caller, msg_ptr, msg_len) {
                        let msg = String::from_utf8_lossy(&bytes);
                        match level {
                            0 => tracing::trace!("[wasm] {}", msg),
                            1 => tracing::debug!("[wasm] {}", msg),
                            2 => tracing::info!("[wasm] {}", msg),
                            3 => tracing::warn!("[wasm] {}", msg),
                            _ => tracing::error!("[wasm] {}", msg),
                        }
                    }
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link log: {}", e)))?;

    // -- Result length (module calls this to know how much to read) --
    linker
        .func_wrap(
            "minns",
            "result_len",
            |caller: Caller<'_, HostEnv>| -> i32 { caller.data().last_result.len() as i32 },
        )
        .map_err(|e| WasmError::HostError(format!("link result_len: {}", e)))?;

    // -- Module identity --
    linker
        .func_wrap("minns", "module_id", |caller: Caller<'_, HostEnv>| -> i64 {
            caller.data().module_id as i64
        })
        .map_err(|e| WasmError::HostError(format!("link module_id: {}", e)))?;

    linker
        .func_wrap("minns", "group_id", |caller: Caller<'_, HostEnv>| -> i64 {
            caller.data().group_id as i64
        })
        .map_err(|e| WasmError::HostError(format!("link group_id: {}", e)))?;

    // -- Table: get row by ID --
    linker
        .func_wrap(
            "minns",
            "table_get",
            |mut caller: Caller<'_, HostEnv>,
             table_ptr: i32,
             table_len: i32,
             row_id: i64,
             _result_ptr: i32|
             -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    let table_name_bytes =
                        abi::read_from_wasm(&memory, &caller, table_ptr, table_len)?;
                    let table_name = String::from_utf8_lossy(&table_name_bytes).to_string();

                    caller.data().permissions.check_table_read(&table_name)?;

                    let group_id = caller.data().group_id;
                    let catalog = caller
                        .data()
                        .table_catalog
                        .try_read()
                        .map_err(|_| WasmError::HostError("catalog locked".into()))?;
                    let table = catalog.get_table(&table_name).ok_or_else(|| {
                        WasmError::HostError(format!("table not found: {}", table_name))
                    })?;

                    let row = table
                        .get_active(group_id, row_id as u64)
                        .ok_or_else(|| WasmError::HostError(format!("row {} not found", row_id)))?;

                    caller.data().usage.record_rows_read(1);
                    abi::to_msgpack(&row.values)
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link table_get: {}", e)))?;

    // -- Table: insert --
    linker
        .func_wrap(
            "minns",
            "table_insert",
            |mut caller: Caller<'_, HostEnv>,
             table_ptr: i32,
             table_len: i32,
             row_ptr: i32,
             row_len: i32|
             -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    let table_name_bytes =
                        abi::read_from_wasm(&memory, &caller, table_ptr, table_len)?;
                    let table_name = String::from_utf8_lossy(&table_name_bytes).to_string();

                    caller.data().permissions.check_table_write(&table_name)?;

                    let row_bytes = abi::read_from_wasm(&memory, &caller, row_ptr, row_len)?;
                    let values: Vec<CellValue> = abi::from_msgpack(&row_bytes)?;

                    let group_id = caller.data().group_id;
                    let mut catalog = caller
                        .data()
                        .table_catalog
                        .try_write()
                        .map_err(|_| WasmError::HostError("catalog locked for write".into()))?;
                    let table = catalog.get_table_mut(&table_name).ok_or_else(|| {
                        WasmError::HostError(format!("table not found: {}", table_name))
                    })?;

                    let (row_id, version_id) = table
                        .insert(group_id, values)
                        .map_err(|e| WasmError::HostError(e.to_string()))?;

                    caller.data().usage.record_rows_written(1);

                    #[derive(serde::Serialize)]
                    struct InsertResult {
                        row_id: u64,
                        version_id: u64,
                    }
                    abi::to_msgpack(&InsertResult { row_id, version_id })
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link table_insert: {}", e)))?;

    // -- Table: delete --
    linker
        .func_wrap(
            "minns",
            "table_delete",
            |mut caller: Caller<'_, HostEnv>, table_ptr: i32, table_len: i32, row_id: i64| -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    let table_name_bytes =
                        abi::read_from_wasm(&memory, &caller, table_ptr, table_len)?;
                    let table_name = String::from_utf8_lossy(&table_name_bytes).to_string();

                    caller.data().permissions.check_table_write(&table_name)?;

                    let group_id = caller.data().group_id;
                    let mut catalog = caller
                        .data()
                        .table_catalog
                        .try_write()
                        .map_err(|_| WasmError::HostError("catalog locked for write".into()))?;
                    let table = catalog.get_table_mut(&table_name).ok_or_else(|| {
                        WasmError::HostError(format!("table not found: {}", table_name))
                    })?;

                    let vid = table
                        .delete(group_id, row_id as u64)
                        .map_err(|e| WasmError::HostError(e.to_string()))?;

                    caller.data().usage.record_rows_written(1);
                    abi::to_msgpack(&vid)
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link table_delete: {}", e)))?;

    // -- Table: query (MinnsQL FROM query) --
    linker
        .func_wrap(
            "minns",
            "table_query",
            |mut caller: Caller<'_, HostEnv>,
             query_ptr: i32,
             query_len: i32,
             _result_ptr: i32|
             -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    let query_bytes = abi::read_from_wasm(&memory, &caller, query_ptr, query_len)?;
                    let query_str = String::from_utf8_lossy(&query_bytes).to_string();

                    // Extract table name from FROM clause for permission check.
                    // Simple extraction: look for "FROM <table_name>" pattern.
                    let query_upper = query_str.to_uppercase();
                    if let Some(pos) = query_upper.find("FROM ") {
                        let rest = query_str[pos + 5..].trim();
                        let table_name = rest.split_whitespace().next().unwrap_or("");
                        if !table_name.is_empty() {
                            caller.data().permissions.check_table_read(table_name)?;
                        }
                    }

                    let group_id = caller.data().group_id;
                    let catalog = caller
                        .data()
                        .table_catalog
                        .try_read()
                        .map_err(|_| WasmError::HostError("catalog locked".into()))?;

                    let output = agent_db_graph::query_lang::execute_table_query(
                        &query_str, &catalog, group_id,
                    )
                    .map_err(|e| WasmError::HostError(e.to_string()))?;

                    let rows_count = output.rows.len() as u64;
                    caller.data().usage.record_rows_read(rows_count);

                    // Convert QueryOutput to a msgpack-serialisable form
                    let result = QueryResultMsg {
                        columns: output.columns,
                        rows: output
                            .rows
                            .into_iter()
                            .map(|row| row.into_iter().map(value_to_msgpack_value).collect())
                            .collect(),
                    };
                    abi::to_msgpack(&result)
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link table_query: {}", e)))?;

    // -- Graph: query (MinnsQL MATCH query) --
    linker
        .func_wrap(
            "minns",
            "graph_query",
            |mut caller: Caller<'_, HostEnv>,
             query_ptr: i32,
             query_len: i32,
             _result_ptr: i32|
             -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    caller.data().permissions.check_graph_query()?;

                    let query_bytes = abi::read_from_wasm(&memory, &caller, query_ptr, query_len)?;
                    let _query_str = String::from_utf8_lossy(&query_bytes).to_string();

                    // Graph query execution requires access to the GraphEngine,
                    // which is not available in the HostEnv yet.
                    Err(WasmError::HostError(
                        "graph_query not yet available — use table_query for table operations"
                            .into(),
                    ))
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link graph_query: {}", e)))?;

    // -- HTTP: fetch (sandboxed) --
    linker
        .func_wrap(
            "minns",
            "http_fetch",
            |mut caller: Caller<'_, HostEnv>,
             req_ptr: i32,
             req_len: i32,
             _result_ptr: i32|
             -> i32 {
                let result = (|| -> Result<Vec<u8>, WasmError> {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

                    let req_bytes = abi::read_from_wasm(&memory, &caller, req_ptr, req_len)?;

                    #[derive(serde::Deserialize)]
                    struct HttpRequest {
                        url: String,
                        #[serde(default = "default_method")]
                        method: String,
                        #[serde(default)]
                        headers: std::collections::HashMap<String, String>,
                        #[serde(default)]
                        body: Option<Vec<u8>>,
                    }
                    fn default_method() -> String {
                        "GET".into()
                    }

                    let req: HttpRequest = abi::from_msgpack(&req_bytes)?;

                    // Validate URL scheme — only http/https allowed
                    if !req.url.starts_with("http://") && !req.url.starts_with("https://") {
                        return Err(WasmError::PermissionDenied(
                            "only http:// and https:// URLs are allowed".into(),
                        ));
                    }

                    // Extract domain for permission check
                    let domain = req
                        .url
                        .split("//")
                        .nth(1)
                        .and_then(|s| s.split('/').next())
                        .and_then(|s| s.split('@').last()) // strip userinfo
                        .and_then(|s| s.split(':').next()) // strip port
                        .unwrap_or("");

                    // Block internal/private network addresses
                    let blocked = [
                        "localhost",
                        "127.0.0.1",
                        "0.0.0.0",
                        "[::1]",
                        "169.254.169.254",
                        "metadata.google.internal",
                    ];
                    if blocked.contains(&domain)
                        || domain.starts_with("10.")
                        || domain.starts_with("172.16.")
                        || domain.starts_with("192.168.")
                    {
                        return Err(WasmError::PermissionDenied(format!(
                            "blocked internal network address: {}",
                            domain
                        )));
                    }

                    caller.data().permissions.check_http_fetch(domain)?;

                    let client = caller
                        .data()
                        .http_client
                        .as_ref()
                        .ok_or_else(|| WasmError::HostError("HTTP client not available".into()))?
                        .clone();

                    // Execute HTTP request synchronously (blocking on tokio runtime)
                    let response = tokio::task::block_in_place(|| {
                        let rt = tokio::runtime::Handle::current();
                        rt.block_on(async {
                            let mut builder = match req.method.to_uppercase().as_str() {
                                "POST" => client.post(&req.url),
                                "PUT" => client.put(&req.url),
                                "DELETE" => client.delete(&req.url),
                                "PATCH" => client.patch(&req.url),
                                _ => client.get(&req.url),
                            };
                            for (k, v) in &req.headers {
                                builder = builder.header(k.as_str(), v.as_str());
                            }
                            if let Some(body) = req.body {
                                builder = builder.body(body);
                            }
                            builder.send().await
                        })
                    })
                    .map_err(|e| WasmError::HostError(format!("HTTP request failed: {}", e)))?;

                    let status = response.status().as_u16();

                    // Pre-flight size check via Content-Length header
                    if let Some(cl) = response.content_length() {
                        if cl > 1024 * 1024 {
                            return Err(WasmError::HostError(format!(
                                "HTTP response Content-Length {} exceeds 1MB limit",
                                cl
                            )));
                        }
                    }

                    let response_bytes = tokio::task::block_in_place(|| {
                        let rt = tokio::runtime::Handle::current();
                        rt.block_on(response.bytes())
                    })
                    .map_err(|e| WasmError::HostError(format!("HTTP read body failed: {}", e)))?;

                    if response_bytes.len() > 1024 * 1024 {
                        return Err(WasmError::HostError(
                            "HTTP response exceeds 1MB limit".into(),
                        ));
                    }

                    caller
                        .data()
                        .usage
                        .record_http_request(response_bytes.len() as u64);

                    #[derive(serde::Serialize)]
                    struct HttpResponse {
                        status: u16,
                        body: Vec<u8>,
                    }

                    abi::to_msgpack(&HttpResponse {
                        status,
                        body: response_bytes.to_vec(),
                    })
                })();

                match result {
                    Ok(bytes) => {
                        caller.data_mut().last_result = bytes;
                        0
                    },
                    Err(e) => {
                        let err_bytes = abi::to_msgpack(&e.to_string()).unwrap_or_default();
                        caller.data_mut().last_result = err_bytes;
                        -1
                    },
                }
            },
        )
        .map_err(|e| WasmError::HostError(format!("link http_fetch: {}", e)))?;

    Ok(())
}
