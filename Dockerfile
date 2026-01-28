# ============================================
# EventGraphDB Production Dockerfile
# ============================================
# Multi-stage build for optimized production image
# Excludes visualizer (client/) - for local testing only

# ============================================
# Stage 1: Builder
# ============================================
FROM rust:1.75-slim-bookworm as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (for layer caching)
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY server ./server
COPY examples ./examples

# Build release binary with optimizations
RUN cargo build --release --package eventgraphdb-server

# Verify binary exists
RUN ls -lh /build/target/release/eventgraphdb-server

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

LABEL maintainer="EventGraphDB Team"
LABEL description="EventGraphDB - Event-driven graph database with self-evolution"
LABEL version="0.2.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash eventgraph && \
    mkdir -p /data && \
    chown -R eventgraph:eventgraph /app /data

# Copy binary from builder
COPY --from=builder --chown=eventgraph:eventgraph \
    /build/target/release/eventgraphdb-server /app/server

# Set permissions
RUN chmod +x /app/server

# Switch to non-root user
USER eventgraph

# Expose ports
# 8080 - HTTP/WebSocket API
# 9090 - Metrics endpoint (Prometheus)
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Environment variables (can be overridden)
ENV RUST_LOG=info
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV STORAGE_BACKEND=persistent
ENV REDB_PATH=/data/eventgraph.redb
ENV REDB_CACHE_SIZE_MB=256
ENV MEMORY_CACHE_SIZE=10000
ENV STRATEGY_CACHE_SIZE=5000
ARG SERVICE_PROFILE=normal
ENV SERVICE_PROFILE=${SERVICE_PROFILE}

# NER Service Configuration (override at runtime)
ENV NER_SERVICE_URL=http://localhost:8081/ner
ENV NER_REQUEST_TIMEOUT_MS=5000

# Volume for persistent data
VOLUME ["/data"]

# Run server
ENTRYPOINT ["/app/server"]
