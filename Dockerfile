# ============================================
# MinnsDB Production Dockerfile
# ============================================
# Multi-stage build for optimized production image
# Excludes visualizer (client/) - for local testing only

# ============================================
# Stage 1: Builder
# ============================================
FROM rust:1.93-slim-bookworm AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (for layer caching)
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY ml ./ml
COPY server ./server
COPY examples ./examples

# Build release binary with optimizations
RUN cargo build --release --package minnsdb-server

# Verify binary exists
RUN ls -lh /build/target/release/minnsdb-server

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

LABEL maintainer="MinnsDB Team"
LABEL description="MinnsDB - Event-driven graph database with self-evolution"
LABEL version="0.9.5"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash minns && \
    mkdir -p /data && \
    chown -R minns:minns /app /data

# Copy binary from builder
COPY --from=builder --chown=minns:minns \
    /build/target/release/minnsdb-server /app/server

# Set permissions
RUN chmod +x /app/server

# Switch to non-root user
USER minns

# Expose ports
# 3000 - HTTP/WebSocket API
# 9090 - Metrics endpoint (Prometheus - not implemented)
EXPOSE 3000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

# Service profile (normal or free)
ARG SERVICE_PROFILE=normal
ENV SERVICE_PROFILE=${SERVICE_PROFILE}

# Environment variables (can be overridden at runtime)
ENV RUST_LOG=info
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=3000
ENV STORAGE_BACKEND=persistent
ENV REDB_PATH=/data/minns.redb

# NER Service Configuration (override at runtime)
ENV NER_SERVICE_URL=http://localhost:8081/ner
ENV NER_REQUEST_TIMEOUT_MS=5000

# NOTE: Cache sizes and limits are set in server config based on SERVICE_PROFILE
# FREE profile: 64MB cache, 1K memories, 500 strategies, 50K max nodes, no Louvain
# NORMAL profile: 256MB cache, 10K memories, 5K strategies, 1M max nodes, Louvain enabled

# Volume for persistent data
VOLUME ["/data"]

# Run server
ENTRYPOINT ["/app/server"]
