# EventGraphDB - Hertz Deployment Plan

**Target Platform:** Hertz Cloud
**Version:** 1.0.0
**Last Updated:** 2026-01-20

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Build Process](#build-process)
5. [Deployment Steps](#deployment-steps)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Scaling](#scaling)
9. [Security](#security)
10. [Rollback](#rollback)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This document outlines the complete deployment process for EventGraphDB on Hertz cloud infrastructure.

### Deployment Targets

| Environment | URL | Purpose |
|------------|-----|---------|
| Development | dev.eventgraph.hertz.app | Testing and development |
| Staging | staging.eventgraph.hertz.app | Pre-production validation |
| Production | api.eventgraph.hertz.app | Live production system |

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Hertz Load Balancer                       │
│                  (TLS termination, DDoS)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Server 1 │  │ Server 2 │  │ Server 3 │  (Auto-scaling group)
│ API+DB   │  │ API+DB   │  │ API+DB   │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   ▼
         ┌─────────────────┐
         │  Shared Storage │  (Network volume for data/)
         │   (Persistent)  │
         └─────────────────┘
```

---

## Prerequisites

### 1. Hertz Account Setup

```bash
# Install Hertz CLI
curl -fsSL https://cli.hertz.app/install.sh | sh

# Login
hertz auth login

# Create project
hertz projects create eventgraphdb \
  --region us-west-2 \
  --tier production
```

### 2. Domain Configuration

```bash
# Add custom domain
hertz domains add api.eventgraph.hertz.app \
  --project eventgraphdb

# Configure DNS (add these records to your DNS provider)
# A record: api.eventgraph.hertz.app → <hertz-lb-ip>
# TLS auto-provisioned via Let's Encrypt
```

### 3. Environment Variables

Create `.env.production`:
```bash
# Server
RUST_LOG=info
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
SERVER_WORKERS=4

# Storage
STORAGE_BACKEND=persistent
REDB_PATH=/data/eventgraph.redb
REDB_CACHE_SIZE_MB=256
MEMORY_CACHE_SIZE=10000
STRATEGY_CACHE_SIZE=5000

# Graph
MAX_GRAPH_SIZE=1000000
PERSISTENCE_INTERVAL=1000

# Auth
JWT_SECRET=${JWT_SECRET}  # Set via Hertz secrets
API_KEY_HASH=${API_KEY_HASH}  # Set via Hertz secrets

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
TRACING_ENDPOINT=${TRACING_ENDPOINT}
```

---

## Architecture

### Single-Server Deployment (Development/Staging)

```
┌───────────────────────────────────┐
│       Hertz Container              │
│  ┌─────────────────────────────┐  │
│  │  EventGraphDB Server        │  │
│  │  (Rust binary)              │  │
│  │  - HTTP/WebSocket API       │  │
│  │  - Graph Engine             │  │
│  │  - Storage Engine           │  │
│  └─────────────────────────────┘  │
│  ┌─────────────────────────────┐  │
│  │  Data Volume (/data)        │  │
│  │  - eventgraph.redb          │  │
│  │  - WAL files                │  │
│  └─────────────────────────────┘  │
└───────────────────────────────────┘
```

### Multi-Server Deployment (Production)

```
                    ┌─────────────────┐
                    │ Load Balancer   │
                    │ (Hertz LB)      │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
        ┌───────▼────┐ ┌────▼─────┐ ┌───▼──────┐
        │  Server 1  │ │ Server 2 │ │ Server 3 │
        │            │ │          │ │          │
        │ Read+Write │ │ Read+Write│ │Read+Write│
        └────────────┘ └──────────┘ └──────────┘
                │            │            │
                └────────────┼────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Network Volume │
                    │  /data (shared) │
                    └─────────────────┘
```

**Note:** For production, use **sharded architecture** instead of shared volume for better scalability.

### Sharded Production Architecture (Recommended)

```
                    ┌─────────────────┐
                    │  API Gateway    │
                    │  (routing)      │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
        ┌───────▼────┐ ┌────▼─────┐ ┌───▼──────┐
        │  Shard 1   │ │ Shard 2  │ │ Shard 3  │
        │            │ │          │ │          │
        │ Buckets    │ │ Buckets  │ │ Buckets  │
        │  0-999     │ │1000-1999 │ │2000-2999 │
        └────────────┘ └──────────┘ └──────────┘
             │              │              │
        ┌────▼───┐     ┌───▼────┐    ┌───▼────┐
        │ Volume │     │ Volume │    │ Volume │
        │  1     │     │   2    │    │   3    │
        └────────┘     └────────┘    └────────┘
```

---

## Build Process

### 1. Optimize Rust Build

`Cargo.toml`:
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = 'abort'
strip = true
```

### 2. Build Docker Image

`Dockerfile.hertz`:
```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM rust:1.75-slim as builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY server ./server

# Build release binary
RUN cargo build --release --package eventgraphdb-server

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/target/release/eventgraphdb-server /app/server

# Create data directory
RUN mkdir -p /data && chmod 755 /data

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Run
CMD ["/app/server"]
```

### 3. Build and Push

```bash
# Build
docker build -f Dockerfile.hertz -t eventgraphdb:latest .

# Tag for Hertz registry
docker tag eventgraphdb:latest \
  registry.hertz.app/eventgraphdb/server:1.0.0

# Push
hertz images push registry.hertz.app/eventgraphdb/server:1.0.0
```

---

## Deployment Steps

### Development Environment

```bash
# 1. Create deployment config
cat > hertz.dev.yaml <<EOF
name: eventgraphdb-dev
region: us-west-2
tier: development

image: registry.hertz.app/eventgraphdb/server:1.0.0

resources:
  cpu: 1
  memory: 2Gi
  storage: 10Gi

env:
  RUST_LOG: debug
  STORAGE_BACKEND: persistent
  REDB_PATH: /data/eventgraph.redb

volumes:
  - name: data
    path: /data
    size: 10Gi

ports:
  - port: 8080
    protocol: http
  - port: 9090
    protocol: http
    internal: true

health_check:
  path: /health
  interval: 30s
  timeout: 3s
EOF

# 2. Deploy
hertz deploy -f hertz.dev.yaml

# 3. Verify
hertz logs eventgraphdb-dev --follow
```

### Staging Environment

```bash
# 1. Create staging config
cat > hertz.staging.yaml <<EOF
name: eventgraphdb-staging
region: us-west-2
tier: staging

image: registry.hertz.app/eventgraphdb/server:1.0.0

resources:
  cpu: 2
  memory: 4Gi
  storage: 50Gi

replicas: 2

env_from_secret: eventgraphdb-staging-secrets

volumes:
  - name: data
    path: /data
    size: 50Gi

ports:
  - port: 8080
    protocol: http
    domain: staging.eventgraph.hertz.app

autoscaling:
  min: 2
  max: 4
  cpu_target: 70

health_check:
  path: /health
  interval: 30s
EOF

# 2. Create secrets
hertz secrets create eventgraphdb-staging-secrets \
  --from-env-file .env.staging

# 3. Deploy
hertz deploy -f hertz.staging.yaml

# 4. Run integration tests
curl https://staging.eventgraph.hertz.app/health
```

### Production Environment

```bash
# 1. Create production config
cat > hertz.production.yaml <<EOF
name: eventgraphdb-prod
region: us-west-2
tier: production

image: registry.hertz.app/eventgraphdb/server:1.0.0

resources:
  cpu: 4
  memory: 8Gi
  storage: 200Gi

replicas: 3

env_from_secret: eventgraphdb-prod-secrets

volumes:
  - name: data
    path: /data
    size: 200Gi
    type: network
    backup: true
    backup_schedule: "0 2 * * *"  # Daily at 2 AM

ports:
  - port: 8080
    protocol: http
    domain: api.eventgraph.hertz.app
  - port: 9090
    protocol: http
    internal: true

autoscaling:
  min: 3
  max: 10
  cpu_target: 60
  memory_target: 70

health_check:
  path: /health
  interval: 15s
  timeout: 3s
  healthy_threshold: 2
  unhealthy_threshold: 3

rolling_update:
  max_surge: 1
  max_unavailable: 0

monitoring:
  enabled: true
  prometheus: true
  grafana_dashboard: true

alerts:
  - name: high_error_rate
    condition: error_rate > 0.05
    channel: slack
  - name: high_latency
    condition: p95_latency > 100ms
    channel: slack
  - name: low_memory
    condition: memory_usage > 0.9
    channel: pagerduty
EOF

# 2. Create production secrets
hertz secrets create eventgraphdb-prod-secrets \
  JWT_SECRET=$(openssl rand -base64 32) \
  API_KEY_HASH=$(openssl rand -hex 64) \
  --from-env-file .env.production

# 3. Deploy with canary strategy
hertz deploy -f hertz.production.yaml \
  --strategy canary \
  --canary-weight 10  # 10% traffic to new version

# 4. Monitor canary
hertz canary status eventgraphdb-prod

# 5. Promote to 100% if healthy
hertz canary promote eventgraphdb-prod

# 6. Verify
curl https://api.eventgraph.hertz.app/health
```

---

## Configuration

### Environment-Specific Settings

#### Development
```yaml
resources:
  cpu: 1 core
  memory: 2 GB
  storage: 10 GB

replicas: 1
autoscaling: disabled

logging:
  level: debug
  format: pretty

persistence:
  backup: disabled
```

#### Staging
```yaml
resources:
  cpu: 2 cores
  memory: 4 GB
  storage: 50 GB

replicas: 2
autoscaling:
  min: 2
  max: 4

logging:
  level: info
  format: json

persistence:
  backup: enabled
  retention: 7 days
```

#### Production
```yaml
resources:
  cpu: 4 cores
  memory: 8 GB
  storage: 200 GB

replicas: 3
autoscaling:
  min: 3
  max: 10

logging:
  level: info
  format: json
  shipping: enabled

persistence:
  backup: enabled
  retention: 30 days
  replication: 3x
```

### Tuning Parameters

```bash
# For high-throughput workloads
REDB_CACHE_SIZE_MB=512
MEMORY_CACHE_SIZE=50000
STRATEGY_CACHE_SIZE=20000
SERVER_WORKERS=8

# For low-latency workloads
REDB_CACHE_SIZE_MB=1024
MEMORY_CACHE_SIZE=100000
PERSISTENCE_INTERVAL=100  # More frequent flushes

# For memory-constrained environments
REDB_CACHE_SIZE_MB=128
MEMORY_CACHE_SIZE=5000
STRATEGY_CACHE_SIZE=2000
MAX_GRAPH_SIZE=50000
```

---

## Monitoring

### Hertz Built-in Monitoring

```bash
# View metrics dashboard
hertz dashboard eventgraphdb-prod

# View logs
hertz logs eventgraphdb-prod --follow

# View resource usage
hertz stats eventgraphdb-prod
```

### Custom Metrics Endpoint

The server exposes Prometheus metrics on port 9090:

```
# HELP eventgraph_events_total Total events processed
# TYPE eventgraph_events_total counter
eventgraph_events_total{type="cognitive"} 12345

# HELP eventgraph_graph_nodes Current number of graph nodes
# TYPE eventgraph_graph_nodes gauge
eventgraph_graph_nodes 45000

# HELP eventgraph_query_duration_seconds Query duration
# TYPE eventgraph_query_duration_seconds histogram
eventgraph_query_duration_seconds_bucket{le="0.01"} 9500
eventgraph_query_duration_seconds_bucket{le="0.05"} 9900
```

### Grafana Dashboard

```bash
# Import pre-built dashboard
hertz grafana import \
  --dashboard eventgraphdb-dashboard.json \
  --project eventgraphdb-prod
```

**Dashboard Panels:**
1. Events per second
2. Graph size (nodes/edges)
3. Query latency (p50, p95, p99)
4. Memory formation rate
5. Strategy usage
6. Cache hit rate
7. Disk I/O
8. Memory usage

### Alerting Rules

```yaml
alerts:
  # High error rate
  - name: high_error_rate
    query: rate(eventgraph_errors_total[5m]) > 0.05
    severity: warning
    notification: slack

  # High latency
  - name: high_latency_p95
    query: eventgraph_query_duration_seconds{quantile="0.95"} > 0.1
    severity: warning
    notification: slack

  # Low cache hit rate
  - name: low_cache_hit_rate
    query: eventgraph_cache_hit_rate < 0.8
    severity: info
    notification: slack

  # Disk space
  - name: low_disk_space
    query: disk_usage_percent > 0.85
    severity: critical
    notification: pagerduty

  # Memory pressure
  - name: high_memory_usage
    query: memory_usage_percent > 0.9
    severity: critical
    notification: pagerduty
```

---

## Scaling

### Vertical Scaling

```bash
# Increase resources
hertz scale eventgraphdb-prod \
  --cpu 8 \
  --memory 16Gi \
  --storage 500Gi

# Apply changes with zero downtime
hertz deploy -f hertz.production.yaml --rolling-update
```

### Horizontal Scaling (Sharding)

**Step 1: Deploy Router**

```yaml
# router.yaml
name: eventgraphdb-router
image: registry.hertz.app/eventgraphdb/router:1.0.0

routing_strategy: hash  # hash(goal_bucket_id) % num_shards

backends:
  - name: shard-1
    url: http://eventgraphdb-shard-1:8080
    buckets: 0-999

  - name: shard-2
    url: http://eventgraphdb-shard-2:8080
    buckets: 1000-1999

  - name: shard-3
    url: http://eventgraphdb-shard-3:8080
    buckets: 2000-2999
```

**Step 2: Deploy Shards**

```bash
for i in 1 2 3; do
  cat > shard-$i.yaml <<EOF
name: eventgraphdb-shard-$i
image: registry.hertz.app/eventgraphdb/server:1.0.0

resources:
  cpu: 4
  memory: 8Gi
  storage: 200Gi

env:
  SHARD_ID: $i
  BUCKET_RANGE_START: $((($i - 1) * 1000))
  BUCKET_RANGE_END: $(($i * 1000 - 1))

volumes:
  - name: data-$i
    path: /data
    size: 200Gi
EOF

  hertz deploy -f shard-$i.yaml
done
```

**Step 3: Update DNS**

```bash
# Point domain to router
hertz domains update api.eventgraph.hertz.app \
  --target eventgraphdb-router
```

### Auto-Scaling Configuration

```yaml
autoscaling:
  min_replicas: 3
  max_replicas: 10

  # CPU-based
  cpu_target: 60  # Target 60% CPU utilization

  # Memory-based
  memory_target: 70  # Target 70% memory utilization

  # Custom metric-based
  custom_metrics:
    - name: eventgraph_queue_depth
      target: 100  # Keep queue below 100 events

  scale_up:
    stabilization_window: 60s
    policies:
      - type: percent
        value: 50  # Increase by 50% per step
        period: 60s

  scale_down:
    stabilization_window: 300s  # Wait 5 min before scaling down
    policies:
      - type: percent
        value: 25  # Decrease by 25% per step
        period: 60s
```

---

## Security

### TLS Configuration

```yaml
tls:
  enabled: true
  auto_cert: true  # Let's Encrypt auto-provisioning
  min_version: "1.3"
  ciphers:
    - TLS_AES_128_GCM_SHA256
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
```

### Network Policies

```yaml
network_policies:
  # Allow ingress only from load balancer
  ingress:
    - from:
        - source: hertz-lb
      ports:
        - 8080

  # Allow egress to monitoring
  egress:
    - to:
        - destination: prometheus
      ports:
        - 9090
```

### Secrets Management

```bash
# Create secrets
hertz secrets create eventgraphdb-secrets \
  --from-literal JWT_SECRET="$(openssl rand -base64 32)" \
  --from-literal API_KEY="sk_live_$(openssl rand -hex 32)" \
  --from-file .env.production

# Rotate secrets
hertz secrets rotate eventgraphdb-secrets JWT_SECRET

# Audit access
hertz secrets audit eventgraphdb-secrets
```

### Authentication Middleware

Implemented in `server/src/auth.rs`:

```rust
pub async fn auth_middleware(
    req: Request<Body>,
    next: Next<Body>,
) -> Result<Response, AuthError> {
    // Extract token
    let token = req.headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or(AuthError::MissingToken)?;

    // Validate JWT
    let claims = validate_jwt(token)?;

    // Check scopes
    let required_scopes = get_required_scopes(&req);
    if !has_required_scopes(&claims, &required_scopes) {
        return Err(AuthError::InsufficientPermissions);
    }

    // Add claims to request extensions
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}
```

---

## Rollback

### Automated Rollback

```bash
# Hertz automatically rolls back if:
# - Health checks fail
# - Error rate > 5%
# - Crash loop detected

# Manual rollback
hertz rollback eventgraphdb-prod

# Rollback to specific version
hertz rollback eventgraphdb-prod --to-revision 42
```

### Database Rollback

```bash
# List backups
hertz backups list eventgraphdb-prod

# Restore from backup
hertz backups restore eventgraphdb-prod \
  --backup-id backup-2026-01-20-02-00 \
  --target /data

# Verify restoration
hertz exec eventgraphdb-prod -- ls -lh /data
```

### Blue-Green Deployment

```yaml
# Deploy new version (green) alongside existing (blue)
deployment_strategy: blue_green

blue_green:
  # Deploy green
  preview_url: green.eventgraph.hertz.app

  # Run smoke tests
  smoke_tests:
    - curl https://green.eventgraph.hertz.app/health
    - run_integration_tests.sh

  # If tests pass, switch traffic
  switch_traffic: manual  # or "automatic"
```

---

## Troubleshooting

### Common Issues

#### 1. High Latency

**Symptoms:** p95 > 100ms

**Diagnosis:**
```bash
# Check resource usage
hertz stats eventgraphdb-prod

# Check cache hit rate
curl http://localhost:9090/metrics | grep cache_hit_rate

# Check disk I/O
hertz exec eventgraphdb-prod -- iostat -x 1
```

**Solutions:**
- Increase cache sizes (REDB_CACHE_SIZE_MB, MEMORY_CACHE_SIZE)
- Add more replicas for horizontal scaling
- Upgrade to larger instance (more CPU/RAM)

#### 2. Memory Pressure

**Symptoms:** OOM kills, high swap usage

**Diagnosis:**
```bash
# Check memory usage
hertz logs eventgraphdb-prod | grep "out of memory"

# Check heap size
curl http://localhost:9090/metrics | grep heap_size
```

**Solutions:**
```bash
# Reduce cache sizes
MEMORY_CACHE_SIZE=5000  # Down from 10000
STRATEGY_CACHE_SIZE=2000  # Down from 5000

# Enable memory limits
MAX_GRAPH_SIZE=50000  # Limit graph size

# Redeploy with more memory
hertz scale eventgraphdb-prod --memory 16Gi
```

#### 3. Disk Space Full

**Symptoms:** Write failures, backup failures

**Diagnosis:**
```bash
hertz exec eventgraphdb-prod -- df -h /data
```

**Solutions:**
```bash
# Expand volume
hertz volumes resize eventgraphdb-prod-data --size 500Gi

# Clean old backups
hertz backups cleanup eventgraphdb-prod --keep-last 7

# Enable compression
ENABLE_COMPRESSION=true
```

#### 4. Connection Timeouts

**Symptoms:** 504 Gateway Timeout errors

**Diagnosis:**
```bash
# Check server health
curl https://api.eventgraph.hertz.app/health

# Check logs
hertz logs eventgraphdb-prod --tail 100 | grep timeout
```

**Solutions:**
- Increase timeout in load balancer config
- Optimize slow queries
- Add more replicas

### Debug Mode

```bash
# Enable debug logging
hertz env set eventgraphdb-prod RUST_LOG=debug

# Restart to apply
hertz restart eventgraphdb-prod

# View debug logs
hertz logs eventgraphdb-prod --follow --level debug

# Disable debug when done
hertz env set eventgraphdb-prod RUST_LOG=info
hertz restart eventgraphdb-prod
```

### Health Check Endpoint

```bash
# Basic health
curl https://api.eventgraph.hertz.app/health

# Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "storage": {
    "status": "healthy",
    "disk_usage_percent": 42.5
  },
  "cache": {
    "hit_rate": 0.94
  }
}
```

---

## Maintenance

### Backup Schedule

```bash
# Daily backups at 2 AM UTC
hertz backups schedule eventgraphdb-prod \
  --cron "0 2 * * *" \
  --retention 30

# Manual backup
hertz backups create eventgraphdb-prod

# Verify backup
hertz backups verify eventgraphdb-prod-backup-<id>
```

### Updates

```bash
# Update to new version
hertz images push registry.hertz.app/eventgraphdb/server:1.1.0

hertz deploy -f hertz.production.yaml \
  --image registry.hertz.app/eventgraphdb/server:1.1.0 \
  --strategy rolling \
  --health-check-enabled

# Monitor rollout
hertz rollout status eventgraphdb-prod

# If issues, automatic rollback occurs
```

### Database Maintenance

```bash
# Compact database (reduce file size)
hertz exec eventgraphdb-prod -- \
  /app/server --compact --db-path /data/eventgraph.redb

# Verify integrity
hertz exec eventgraphdb-prod -- \
  /app/server --verify --db-path /data/eventgraph.redb

# Rebuild indexes
hertz exec eventgraphdb-prod -- \
  /app/server --rebuild-indexes --db-path /data/eventgraph.redb
```

---

## Cost Optimization

### Resource Sizing

| Workload | CPU | Memory | Storage | Est. Cost/Month |
|----------|-----|--------|---------|-----------------|
| Development | 1 core | 2 GB | 10 GB | $20 |
| Staging | 2 cores | 4 GB | 50 GB | $80 |
| Production (small) | 4 cores | 8 GB | 200 GB | $250 |
| Production (medium) | 8 cores | 16 GB | 500 GB | $600 |
| Production (large) | 16 cores | 32 GB | 1 TB | $1200 |

### Cost Reduction Tips

1. **Use Spot Instances for Dev/Staging** (60% savings)
2. **Enable Auto-scaling** (pay only for what you need)
3. **Compress Storage** (40-60% reduction)
4. **Archive Old Data** (move to cold storage after 90 days)
5. **Optimize Cache Sizes** (reduce memory footprint)

---

## Checklist

### Pre-Deployment

- [ ] Build and test Docker image locally
- [ ] Run integration tests
- [ ] Review resource requirements
- [ ] Configure secrets
- [ ] Set up monitoring and alerts
- [ ] Create backup strategy
- [ ] Plan rollback procedure

### Deployment

- [ ] Deploy to development
- [ ] Deploy to staging
- [ ] Run smoke tests on staging
- [ ] Deploy to production (canary)
- [ ] Monitor canary metrics
- [ ] Promote canary to 100%
- [ ] Verify production health

### Post-Deployment

- [ ] Monitor error rates for 24 hours
- [ ] Check performance metrics
- [ ] Verify backups working
- [ ] Test rollback procedure
- [ ] Update documentation
- [ ] Notify stakeholders

---

**End of Deployment Plan**

For questions or issues, contact: devops@eventgraph.hertz.app
