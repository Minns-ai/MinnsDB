//! REST API handlers for temporal tables.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use agent_db_tables::schema::{ColumnDef, Constraint};
use agent_db_tables::types::CellValue;

use crate::state::AppState;

// -- Request/Response types --

#[derive(Deserialize)]
pub struct CreateTableRequest {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
}

#[derive(Serialize)]
pub struct CreateTableResponse {
    pub table_id: u64,
    pub name: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct InsertRowRequest {
    pub group_id: Option<u64>,
    pub values: Vec<CellValue>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct InsertBatchRequest {
    pub group_id: Option<u64>,
    pub rows: Vec<Vec<CellValue>>,
}

#[derive(Deserialize)]
pub struct UpdateRowRequest {
    pub group_id: Option<u64>,
    pub values: Vec<CellValue>,
}

#[derive(Serialize)]
pub struct RowResponse {
    pub row_id: u64,
    pub version_id: u64,
}

#[derive(Serialize)]
pub struct UpdateResponse {
    pub old_version_id: u64,
    pub new_version_id: u64,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub version_id: u64,
}

#[derive(Deserialize)]
pub struct ScanQuery {
    pub when: Option<String>, // "active" | "all"
    pub as_of: Option<u64>,
    pub group_id: Option<u64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Serialize)]
pub struct ScanResponse {
    pub rows: Vec<RowData>,
    pub count: usize,
}

#[derive(Serialize)]
pub struct RowData {
    pub row_id: u64,
    pub version_id: u64,
    pub group_id: u64,
    pub valid_from: u64,
    pub valid_until: Option<u64>,
    pub values: Vec<CellValue>,
}

#[derive(Serialize)]
pub struct TableStatsResponse {
    pub name: String,
    pub active_rows: usize,
    pub total_versions: usize,
    pub pages: usize,
    pub generation: u64,
}

#[derive(Serialize)]
pub struct SchemaResponse {
    pub table_id: u64,
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<Constraint>,
}

#[derive(Serialize)]
pub struct CompactResponse {
    pub versions_removed: usize,
    pub pages_compacted: usize,
}

fn gid(q: Option<u64>) -> u64 {
    q.unwrap_or(0)
}

fn decoded_to_row_data(row: agent_db_tables::row_codec::DecodedRow) -> RowData {
    RowData {
        row_id: row.row_id,
        version_id: row.version_id,
        group_id: row.group_id,
        valid_from: row.valid_from,
        valid_until: row.valid_until,
        values: row.values,
    }
}

fn table_err(e: agent_db_tables::error::TableError) -> (StatusCode, Json<serde_json::Value>) {
    let status = match &e {
        agent_db_tables::error::TableError::TableNotFound(_) => StatusCode::NOT_FOUND,
        agent_db_tables::error::TableError::RowNotFound(_) => StatusCode::NOT_FOUND,
        agent_db_tables::error::TableError::TableAlreadyExists(_) => StatusCode::CONFLICT,
        agent_db_tables::error::TableError::RowAlreadyDeleted(_) => StatusCode::CONFLICT,
        agent_db_tables::error::TableError::UniqueConstraintViolation(_)
        | agent_db_tables::error::TableError::PrimaryKeyViolation => StatusCode::CONFLICT,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, Json(serde_json::json!({ "error": e.to_string() })))
}

// -- Handlers --

pub async fn create_table(
    State(state): State<AppState>,
    Json(req): Json<CreateTableRequest>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    match catalog.create_table(req.name.clone(), req.columns, req.constraints) {
        Ok(table_id) => (
            StatusCode::CREATED,
            Json(
                serde_json::to_value(CreateTableResponse {
                    table_id,
                    name: req.name,
                })
                .unwrap(),
            ),
        ),
        Err(e) => table_err(e),
    }
}

pub async fn drop_table(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    match catalog.drop_table(&name) {
        Ok(id) => (
            StatusCode::OK,
            Json(serde_json::json!({ "table_id": id, "dropped": true })),
        ),
        Err(e) => table_err(e),
    }
}

pub async fn list_tables(State(state): State<AppState>) -> impl IntoResponse {
    let catalog = state.table_catalog.read().await;
    let tables: Vec<SchemaResponse> = catalog
        .list_tables()
        .into_iter()
        .map(|s| SchemaResponse {
            table_id: s.table_id,
            name: s.name.clone(),
            columns: s.columns.clone(),
            constraints: s.constraints.clone(),
        })
        .collect();
    Json(
        serde_json::to_value(tables)
            .unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})),
    )
}

pub async fn get_schema(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let catalog = state.table_catalog.read().await;
    match catalog.get_table(&name) {
        Some(table) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(SchemaResponse {
                    table_id: table.schema.table_id,
                    name: table.schema.name.clone(),
                    columns: table.schema.columns.clone(),
                    constraints: table.schema.constraints.clone(),
                })
                .unwrap(),
            ),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("table not found: {}", name) })),
        ),
    }
}

pub async fn insert_rows(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    let table = match catalog.get_table_mut(&name) {
        Some(t) => t,
        None => {
            return table_err(agent_db_tables::error::TableError::TableNotFound(name));
        },
    };

    // Support both single row and batch
    if let Some(obj) = req.as_object() {
        // Single row: { group_id?, values: [...] }
        let group_id = obj.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let values: Vec<CellValue> = match obj.get("values") {
            Some(v) => match serde_json::from_value(v.clone()) {
                Ok(vals) => vals,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "error": e.to_string() })),
                    )
                },
            },
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "missing 'values' field" })),
                )
            },
        };
        match table.insert(group_id, values) {
            Ok((rid, vid)) => (
                StatusCode::CREATED,
                Json(
                    serde_json::to_value(RowResponse {
                        row_id: rid,
                        version_id: vid,
                    })
                    .unwrap(),
                ),
            ),
            Err(e) => table_err(e),
        }
    } else if let Some(arr) = req.as_array() {
        // Batch: [{ group_id?, values: [...] }, ...]
        // For simplicity, batch uses group_id from first element
        let mut results = Vec::new();
        for item in arr {
            let group_id = item.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let values: Vec<CellValue> = match item.get("values") {
                Some(v) => match serde_json::from_value(v.clone()) {
                    Ok(vals) => vals,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "error": e.to_string() })),
                        )
                    },
                },
                None => continue,
            };
            match table.insert(group_id, values) {
                Ok((rid, vid)) => results.push(RowResponse {
                    row_id: rid,
                    version_id: vid,
                }),
                Err(e) => return table_err(e),
            }
        }
        (
            StatusCode::CREATED,
            Json(
                serde_json::to_value(results)
                    .unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})),
            ),
        )
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "expected object or array" })),
        )
    }
}

pub async fn update_row(
    State(state): State<AppState>,
    Path((name, row_id)): Path<(String, u64)>,
    Json(req): Json<UpdateRowRequest>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    let table = match catalog.get_table_mut(&name) {
        Some(t) => t,
        None => return table_err(agent_db_tables::error::TableError::TableNotFound(name)),
    };
    match table.update(gid(req.group_id), row_id, req.values) {
        Ok((old_vid, new_vid)) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(UpdateResponse {
                    old_version_id: old_vid,
                    new_version_id: new_vid,
                })
                .unwrap(),
            ),
        ),
        Err(e) => table_err(e),
    }
}

pub async fn delete_row(
    State(state): State<AppState>,
    Path((name, row_id)): Path<(String, u64)>,
    Query(params): Query<ScanQuery>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    let table = match catalog.get_table_mut(&name) {
        Some(t) => t,
        None => return table_err(agent_db_tables::error::TableError::TableNotFound(name)),
    };
    match table.delete(gid(params.group_id), row_id) {
        Ok(vid) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(DeleteResponse { version_id: vid })
                    .unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})),
            ),
        ),
        Err(e) => table_err(e),
    }
}

pub async fn scan_rows(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<ScanQuery>,
) -> impl IntoResponse {
    let catalog = state.table_catalog.read().await;
    let table = match catalog.get_table(&name) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("table not found: {}", name) })),
            )
        },
    };

    let group_id = gid(params.group_id);
    let rows = match params.when.as_deref() {
        Some("all") => table.scan_all(group_id),
        _ => {
            if let Some(ts) = params.as_of {
                table.scan_as_of(group_id, ts)
            } else {
                table.scan_active(group_id)
            }
        },
    };

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(1000).min(10_000);
    let total = rows.len();
    let rows: Vec<RowData> = rows
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(decoded_to_row_data)
        .collect();

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(ScanResponse { count: total, rows })
                .unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})),
        ),
    )
}

pub async fn rows_by_node(
    State(state): State<AppState>,
    Path((name, node_id)): Path<(String, u64)>,
    Query(params): Query<ScanQuery>,
) -> impl IntoResponse {
    let catalog = state.table_catalog.read().await;
    let table = match catalog.get_table(&name) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("table not found: {}", name) })),
            )
        },
    };

    let rows: Vec<RowData> = table
        .rows_by_node(gid(params.group_id), node_id)
        .into_iter()
        .map(decoded_to_row_data)
        .collect();
    let count = rows.len();

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(ScanResponse { count, rows })
                .unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})),
        ),
    )
}

pub async fn compact_table(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut catalog = state.table_catalog.write().await;
    let table = match catalog.get_table_mut(&name) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("table not found: {}", name) })),
            )
        },
    };

    let now = agent_db_core::types::current_timestamp();
    let result = agent_db_tables::compaction::compact_table(table, now);

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(CompactResponse {
                versions_removed: result.versions_removed,
                pages_compacted: result.pages_compacted,
            })
            .unwrap(),
        ),
    )
}

pub async fn table_stats(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let catalog = state.table_catalog.read().await;
    let table = match catalog.get_table(&name) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("table not found: {}", name) })),
            )
        },
    };

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(TableStatsResponse {
                name: table.schema.name.clone(),
                active_rows: table.active_row_count(),
                total_versions: table.total_version_count(),
                pages: table.page_count(),
                generation: table.generation(),
            })
            .unwrap(),
        ),
    )
}
