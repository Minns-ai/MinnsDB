// Export/Import handlers for EventGraphDB REST API — streaming binary v2

use crate::errors::ApiError;
use crate::state::AppState;
use agent_db_graph::ImportMode;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
pub struct ImportQuery {
    /// Import mode: "replace" (default) or "merge"
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_mode() -> String {
    "replace".to_string()
}

#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub success: bool,
    pub memories_imported: u64,
    pub strategies_imported: u64,
    pub graph_nodes_imported: u64,
    pub graph_edges_imported: u64,
    pub total_records: u64,
    pub mode: String,
}

// ========== ChannelWriter ==========

/// `std::io::Write` adapter that sends chunks over an mpsc channel.
/// Used to bridge sync export code to an async streaming HTTP response.
struct ChannelWriter {
    tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
    buf: Vec<u8>,
}

const CHANNEL_BUF_CAP: usize = 64 * 1024; // 64 KB

impl ChannelWriter {
    fn new(tx: mpsc::Sender<Result<Bytes, std::io::Error>>) -> Self {
        Self {
            tx,
            buf: Vec::with_capacity(CHANNEL_BUF_CAP),
        }
    }
}

impl std::io::Write for ChannelWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        if self.buf.len() >= CHANNEL_BUF_CAP {
            self.flush()?;
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let chunk = Bytes::from(std::mem::replace(
            &mut self.buf,
            Vec::with_capacity(CHANNEL_BUF_CAP),
        ));
        self.tx
            .blocking_send(Ok(chunk))
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receiver dropped"))
    }
}

impl Drop for ChannelWriter {
    fn drop(&mut self) {
        // Best-effort flush on drop — cannot propagate errors.
        if !self.buf.is_empty() {
            let chunk = Bytes::from(std::mem::take(&mut self.buf));
            let _ = self.tx.blocking_send(Ok(chunk));
        }
    }
}

// ========== Export handler (streaming response) ==========

/// POST /api/admin/export
///
/// Export all persisted state as a streaming binary v2 response.
/// Content-Type: application/octet-stream
pub async fn export_handler(State(state): State<AppState>) -> Result<Response, ApiError> {
    info!("Starting streaming database export");

    // Step 1: Flush caches (async)
    state
        .engine
        .export_prepare()
        .await
        .map_err(|e| ApiError::Internal(format!("Export prepare failed: {}", e)))?;

    // Step 2: Create channel for streaming
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    // Step 3: Spawn blocking task for sync export
    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let backend = match engine.export_backend() {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.blocking_send(Err(std::io::Error::other(
                    format!("export_backend: {}", e),
                )));
                return;
            },
        };

        let mut writer = ChannelWriter::new(tx.clone());
        match agent_db_graph::export::export_to_writer(backend, &mut writer) {
            Ok(count) => {
                if let Err(e) = writer.flush() {
                    warn!("Export flush error: {}", e);
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
                info!("Export completed: {} records", count);
            },
            Err(e) => {
                warn!("Export failed: {}", e);
                let _ = tx.blocking_send(Err(std::io::Error::other(
                    format!("export: {}", e),
                )));
            },
        }
    });

    // Step 4: Stream response
    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"eventgraphdb-export.bin\"",
            ),
        ],
        body,
    )
        .into_response())
}

// ========== Import handler (streaming request body) ==========

/// POST /api/admin/import
///
/// Import state from a streaming binary v2 request body.
/// Query parameter `mode` controls import behavior:
/// - "replace" (default): wipe existing data then import
/// - "merge": upsert imported records into existing data
pub async fn import_handler(
    State(state): State<AppState>,
    Query(query): Query<ImportQuery>,
    body: Body,
) -> Result<Json<ImportResponse>, ApiError> {
    let mode = match query.mode.as_str() {
        "replace" => ImportMode::Replace,
        "merge" => ImportMode::Merge,
        other => {
            return Err(ApiError::BadRequest(format!(
                "Invalid import mode '{}'. Use 'replace' or 'merge'.",
                other
            )))
        },
    };

    info!("Starting streaming database import: mode={}", query.mode);
    let start = std::time::Instant::now();

    // Create a duplex pipe: async writer -> sync reader
    let (duplex_writer, duplex_reader) = tokio::io::duplex(256 * 1024);

    // Spawn async task to pump body chunks into the pipe
    let pump_handle = tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let mut writer = duplex_writer;
        let mut stream = body.into_data_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if let Err(e) = writer.write_all(&chunk).await {
                        warn!("Import pump write error: {}", e);
                        return Err(e);
                    }
                },
                Err(e) => {
                    warn!("Import pump body read error: {}", e);
                    return Err(std::io::Error::other(e.to_string()));
                },
            }
        }
        // Flush and close the writer to signal EOF to the reader
        writer.shutdown().await?;
        Ok(())
    });

    // Spawn blocking task for sync import
    let engine = state.engine.clone();
    let import_result = tokio::task::spawn_blocking(move || {
        let sync_reader = tokio_util::io::SyncIoBridge::new(duplex_reader);
        engine.import_sync(sync_reader, mode)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Import task panicked: {}", e)))?
    .map_err(|e| ApiError::Internal(format!("Import failed: {}", e)))?;

    // Wait for pump to finish (it should already be done or will finish quickly)
    let _ = pump_handle.await;

    // Finalize: reinitialize in-memory stores (async)
    state
        .engine
        .import_finalize()
        .await
        .map_err(|e| ApiError::Internal(format!("Import finalize failed: {}", e)))?;

    info!(
        "Import completed: {} records in {}ms (mode={})",
        import_result.total_records,
        start.elapsed().as_millis(),
        query.mode
    );

    Ok(Json(ImportResponse {
        success: true,
        memories_imported: import_result.memories_imported,
        strategies_imported: import_result.strategies_imported,
        graph_nodes_imported: import_result.graph_nodes_imported,
        graph_edges_imported: import_result.graph_edges_imported,
        total_records: import_result.total_records,
        mode: query.mode,
    }))
}
