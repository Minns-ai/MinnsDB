//! REST API handlers for WASM agent modules.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Deserialize)]
pub struct UploadModuleRequest {
    pub name: String,
    /// Base64-encoded WASM bytes.
    pub wasm_base64: String,
    pub permissions: Vec<String>,
    #[serde(default)]
    pub group_id: Option<u64>,
}

#[derive(Serialize)]
pub struct ModuleInfoResponse {
    pub name: String,
    pub module_id: u64,
    pub enabled: bool,
    pub permissions: Vec<String>,
    pub functions: Vec<String>,
    pub triggers: usize,
}

#[derive(Deserialize)]
pub struct CallFunctionRequest {
    /// MessagePack args as base64 (for HTTP transport).
    #[serde(default)]
    pub args_base64: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateScheduleRequest {
    pub cron: String,
    pub function: String,
}

fn wasm_err(e: minns_wasm_runtime::error::WasmError) -> (StatusCode, Json<serde_json::Value>) {
    let status = match &e {
        minns_wasm_runtime::error::WasmError::ModuleNotFound(_) => StatusCode::NOT_FOUND,
        minns_wasm_runtime::error::WasmError::ModuleAlreadyExists(_) => StatusCode::CONFLICT,
        minns_wasm_runtime::error::WasmError::PermissionDenied(_) => StatusCode::FORBIDDEN,
        minns_wasm_runtime::error::WasmError::LifeExceeded => StatusCode::REQUEST_TIMEOUT,
        minns_wasm_runtime::error::WasmError::Timeout => StatusCode::REQUEST_TIMEOUT,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, Json(serde_json::json!({ "error": e.to_string() })))
}

/// Maximum WASM module size (5MB).
const MAX_MODULE_SIZE: usize = 5 * 1024 * 1024;
/// Maximum number of modules.
const MAX_MODULES: usize = 100;

/// POST /api/modules — upload a WASM module
pub async fn upload_module(
    State(state): State<AppState>,
    Json(req): Json<UploadModuleRequest>,
) -> impl IntoResponse {
    use base64::Engine as _;
    let wasm_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.wasm_base64) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("invalid base64: {}", e) })),
            )
        },
    };

    if wasm_bytes.len() > MAX_MODULE_SIZE {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({ "error": format!("module exceeds {}MB limit", MAX_MODULE_SIZE / 1024 / 1024) }),
            ),
        );
    }

    let group_id = req.group_id.unwrap_or(0);

    // Check module count limit
    {
        let registry = state.module_registry.read().await;
        if registry.module_count() >= MAX_MODULES {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    serde_json::json!({ "error": format!("module limit reached ({})", MAX_MODULES) }),
                ),
            );
        }
    }
    let mut registry = state.module_registry.write().await;
    match registry.upload(
        &state.wasm_runtime,
        req.name.clone(),
        wasm_bytes,
        req.permissions,
        group_id,
        state.table_catalog.clone(),
    ) {
        Ok(instance) => {
            let info = ModuleInfoResponse {
                name: instance.descriptor.name.clone(),
                module_id: instance.module_id,
                enabled: instance.enabled,
                permissions: instance.permissions.grants().to_vec(),
                functions: instance
                    .descriptor
                    .functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect(),
                triggers: instance.descriptor.triggers.len(),
            };
            (
                StatusCode::CREATED,
                Json(
                    serde_json::to_value(&info)
                        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()})),
                ),
            )
        },
        Err(e) => wasm_err(e),
    }
}

/// GET /api/modules — list modules
pub async fn list_modules(State(state): State<AppState>) -> impl IntoResponse {
    let registry = state.module_registry.read().await;
    let modules: Vec<serde_json::Value> = registry
        .list_records()
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "module_id": r.module_id,
                "enabled": r.enabled,
                "group_id": r.group_id,
                "uploaded_at": r.uploaded_at,
            })
        })
        .collect();
    Json(serde_json::json!(modules))
}

/// GET /api/modules/:name — module info
pub async fn get_module(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let registry = state.module_registry.read().await;
    match registry.get(&name) {
        Some(instance) => {
            let info = ModuleInfoResponse {
                name: instance.descriptor.name.clone(),
                module_id: instance.module_id,
                enabled: instance.enabled,
                permissions: instance.permissions.grants().to_vec(),
                functions: instance
                    .descriptor
                    .functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect(),
                triggers: instance.descriptor.triggers.len(),
            };
            (
                StatusCode::OK,
                Json(
                    serde_json::to_value(&info)
                        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()})),
                ),
            )
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("module not found: {}", name) })),
        ),
    }
}

/// DELETE /api/modules/:name — unload module and clean up schedules
pub async fn delete_module(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut registry = state.module_registry.write().await;
    match registry.unload(&name) {
        Ok(_) => {
            // Clean up schedules for this module
            let mut runner = state.schedule_runner.write().await;
            let schedule_ids: Vec<u64> = runner
                .list_for_module(&name)
                .iter()
                .map(|s| s.schedule_id)
                .collect();
            for id in schedule_ids {
                runner.remove(id);
            }
            (StatusCode::OK, Json(serde_json::json!({ "deleted": true })))
        },
        Err(e) => wasm_err(e),
    }
}

/// POST /api/modules/:name/call/:function — call a module function
pub async fn call_function(
    State(state): State<AppState>,
    Path((name, func_name)): Path<(String, String)>,
    Json(req): Json<CallFunctionRequest>,
) -> impl IntoResponse {
    let registry = state.module_registry.read().await;
    let instance = match registry.get(&name) {
        Some(i) => i,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("module not found: {}", name) })),
            )
        },
    };

    let args = if let Some(b64) = &req.args_base64 {
        use base64::Engine as _;
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("invalid args base64: {}", e) })),
                )
            },
        }
    } else {
        vec![]
    };

    match instance.call_function(&state.wasm_runtime, &func_name, &args) {
        Ok(result_bytes) => {
            use base64::Engine as _;
            let result_b64 = base64::engine::general_purpose::STANDARD.encode(&result_bytes);
            (
                StatusCode::OK,
                Json(serde_json::json!({ "result_base64": result_b64 })),
            )
        },
        Err(e) => wasm_err(e),
    }
}

/// PUT /api/modules/:name/enable
pub async fn enable_module(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut registry = state.module_registry.write().await;
    match registry.enable(&name) {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "enabled": true }))),
        Err(e) => wasm_err(e),
    }
}

/// PUT /api/modules/:name/disable
pub async fn disable_module(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut registry = state.module_registry.write().await;
    match registry.disable(&name) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "enabled": false })),
        ),
        Err(e) => wasm_err(e),
    }
}

/// GET /api/modules/:name/usage
pub async fn get_usage(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let registry = state.module_registry.read().await;
    match registry.get_usage(&name) {
        Ok(usage) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(&usage)
                    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()})),
            ),
        ),
        Err(e) => wasm_err(e),
    }
}

/// POST /api/modules/:name/usage/reset
pub async fn reset_usage(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut registry = state.module_registry.write().await;
    match registry.reset_usage(&name) {
        Ok(previous) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "previous_period": serde_json::to_value(&previous).unwrap_or_else(|e| serde_json::json!({"error": e.to_string()})),
                "reset": true,
            })),
        ),
        Err(e) => wasm_err(e),
    }
}

/// GET /api/modules/:name/schedules
pub async fn list_schedules(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let runner = state.schedule_runner.read().await;
    let schedules: Vec<serde_json::Value> = runner
        .list_for_module(&name)
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "schedule_id": s.schedule_id,
                "cron": s.cron,
                "function": s.function,
                "enabled": s.enabled,
                "next_run": s.next_run,
                "last_run": s.last_run,
            })
        })
        .collect();
    Json(serde_json::json!(schedules))
}

/// POST /api/modules/:name/schedules
pub async fn create_schedule(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<CreateScheduleRequest>,
) -> impl IntoResponse {
    // Validate module exists
    {
        let registry = state.module_registry.read().await;
        if registry.get(&name).is_none() {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("module not found: {}", name) })),
            );
        }
    }
    let mut runner = state.schedule_runner.write().await;
    match runner.add(name, req.cron, req.function) {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({ "schedule_id": id })),
        ),
        Err(e) => wasm_err(e),
    }
}

/// DELETE /api/modules/:name/schedules/:id
pub async fn delete_schedule(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let mut runner = state.schedule_runner.write().await;
    // Validate the schedule belongs to this module
    let belongs = runner
        .list_for_module(&name)
        .iter()
        .any(|s| s.schedule_id == id);
    if !belongs {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "schedule not found for this module" })),
        );
    }
    if runner.remove(id) {
        (StatusCode::OK, Json(serde_json::json!({ "deleted": true })))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "schedule not found" })),
        )
    }
}
