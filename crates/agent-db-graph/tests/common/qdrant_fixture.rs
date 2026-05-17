//! Per-binary shared Qdrant container.
//!
//! Calling [`qdrant_url`] starts a Qdrant container the first time it is
//! invoked in a process and returns the same gRPC URL for every subsequent
//! call. Tests share the container; each test SHOULD use a unique
//! `collection_prefix` (via [`test_vectors_config`]) to avoid cross-test
//! contamination of collection state.
//!
//! The container is started lazily, so binaries whose tests do not touch the
//! engine never pay the startup cost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use agent_db_graph::vectors::VectorsConfig;
use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};
use tokio::sync::OnceCell;

const QDRANT_IMAGE: &str = "qdrant/qdrant";
const QDRANT_TAG: &str = "v1.13.0";
const QDRANT_GRPC: u16 = 6334;

/// Pinned: an [`OnceCell`] holds the URL of a started container. The
/// container handle is leaked into the static for the lifetime of the
/// process so it is not dropped (which would stop the container) while
/// tests are still running.
static QDRANT: OnceCell<String> = OnceCell::const_new();

/// Monotonically increasing counter used by [`test_vectors_config`] to
/// generate unique collection prefixes within a test binary.
static PREFIX_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Start the shared Qdrant container if it is not already running and
/// return its gRPC URL.
pub async fn qdrant_url() -> &'static str {
    QDRANT
        .get_or_init(|| async {
            let image = GenericImage::new(QDRANT_IMAGE, QDRANT_TAG)
                .with_exposed_port(QDRANT_GRPC.tcp())
                .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening"))
                .with_startup_timeout(Duration::from_secs(60));
            let container: ContainerAsync<GenericImage> =
                image.start().await.expect("start shared qdrant container");
            let port = container
                .get_host_port_ipv4(QDRANT_GRPC)
                .await
                .expect("read mapped qdrant port");
            // Keep the container alive for the rest of the process: leak the
            // handle into a static. Dropping it would stop the container
            // mid-test.
            Box::leak(Box::new(container));
            format!("http://127.0.0.1:{port}")
        })
        .await
        .as_str()
}

/// Build a [`VectorsConfig`] pointing at the shared container with a unique
/// per-call `collection_prefix`, so concurrent tests do not collide.
pub async fn test_vectors_config() -> VectorsConfig {
    let url = qdrant_url().await;
    let n = PREFIX_COUNTER.fetch_add(1, Ordering::SeqCst);
    let prefix = format!("test_{n}");
    VectorsConfig::new(url, agent_db_graph::vectors::DEFAULT_DIM).with_collection_prefix(prefix)
}
