# EventGraphDB Upgrade Guide

> **Prompt-ready operational runbook.** Paste this entire document as context when asking an AI assistant to help you upgrade EventGraphDB.

---

## Role

You are an operational assistant for EventGraphDB. Your job is to guide the operator through safe, zero-data-loss upgrades of a running EventGraphDB instance deployed as a Docker container.

### Golden Rules

1. **Always export before a major upgrade.** If you are unsure whether the upgrade is major, export anyway. An export file costs minutes; lost data costs days.
2. **Never delete the old volume until the new instance is verified.** Keep the old Docker volume (or a `tar` backup) until you have confirmed counts, spot-checked data, and run a full health check on the new version.
3. **Import is NOT atomic.** If an import is interrupted (crash, network timeout, OOM), the target database may be partially written. Always import into a **fresh volume**, never into a volume that holds data you cannot afford to lose.
4. **The export file is the single source of truth during migration.** Treat it like a database backup: checksum it, copy it to a second location, and never modify it.

---

## Decision Matrix

| Scenario | Procedure | Section |
|---|---|---|
| Patch or minor version bump, same `SCHEMA_MAJOR` | Simple restart (swap image, keep volume) | [Simple Restart](#simple-restart-upgrade) |
| New struct fields added (e.g. new optional column) | Simple restart — versioned msgpack envelope auto-handles new/missing fields | [Simple Restart](#simple-restart-upgrade) |
| `SCHEMA_MAJOR` bump (breaking storage change) | Full export/import cycle | [Full Export/Import Cycle](#full-exportimport-cycle) |
| Cross-host migration (move to new server) | Export from old host, transfer file, import on new host | [Full Export/Import Cycle](#full-exportimport-cycle) |
| Downgrade to an older version | Not supported across `SCHEMA_MAJOR` boundaries — the old binary will reject a newer schema (`TooNew` error) | [Troubleshooting](#troubleshooting) |

**How to check `SCHEMA_MAJOR`:** Look at the release notes for the target version, or inspect `crates/agent-db-storage/src/schema.rs` — the constants `SCHEMA_MAJOR` and `SCHEMA_MINOR` are defined there. Currently: **SCHEMA_MAJOR = 1, SCHEMA_MINOR = 0**.

---

## Pre-flight Checks

Run these commands **before** starting the upgrade. Record the output — you will compare against it after the upgrade.

### 1. Health baseline

```bash
# Record current node/edge counts and version
curl -s http://localhost:3000/api/health | jq .
```

Expected response shape:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 86400,
  "is_healthy": true,
  "node_count": 1234,
  "edge_count": 5678,
  "processing_rate": 42.0
}
```

Save `node_count` and `edge_count` — these are your verification targets.

### 2. Current image and health status

```bash
docker inspect eventgraphdb --format '{{.Config.Image}} | Health: {{.State.Health.Status}}'
```

### 3. Disk space

```bash
# Export file size ≈ uncompressed database size. Ensure 2× free space.
docker exec eventgraphdb du -sh /data/
df -h /var/lib/docker/volumes/
```

---

## Simple Restart Upgrade

Use this when the new version has the **same `SCHEMA_MAJOR`** (patch/minor bump, or additive struct changes).

### Docker CLI

```bash
# 1. Pull the new image
docker pull ghcr.io/<owner>/eventgraphdb:v1.2.3

# 2. Stop and remove the container (volume is preserved)
docker stop eventgraphdb
docker rm eventgraphdb

# 3. Start with the same volume
docker run -d \
  --name eventgraphdb \
  -p 3000:3000 \
  -v eventgraph-data:/data \
  -e RUST_LOG=info \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  ghcr.io/<owner>/eventgraphdb:v1.2.3

# 4. Wait for health and verify
sleep 5
curl -s http://localhost:3000/api/health | jq '{status, version, node_count, edge_count}'
```

### Docker Compose

```bash
# In your docker-compose.yml, update the image tag:
#   image: ghcr.io/<owner>/eventgraphdb:v1.2.3

docker compose pull eventgraphdb
docker compose up -d eventgraphdb

# Verify
sleep 5
curl -s http://localhost:3000/api/health | jq '{status, version, node_count, edge_count}'
```

Confirm that `node_count` and `edge_count` match the pre-flight baseline.

---

## Full Export/Import Cycle

Use this when `SCHEMA_MAJOR` has changed, or for cross-host migration.

### Step 1 — Export

```bash
curl -X POST http://localhost:3000/api/admin/export \
  --max-time 3600 \
  --output eventgraphdb-export.bin
```

The server streams a binary Wire v2 file. The response header is `Content-Type: application/octet-stream` with filename `eventgraphdb-export.bin`.

### Step 2 — Verify the export file

```bash
# Check EGDB magic bytes (first 4 bytes should be 0x45 0x47 0x44 0x42 = "EGDB")
xxd -l 5 eventgraphdb-export.bin
# Expected: 00000000: 4547 4442 02    EGDB.
# The 5th byte (02) is the wire format version.

# Record file size for later comparison
ls -lh eventgraphdb-export.bin
```

### Step 3 — Backup the old volume (safety net)

```bash
docker run --rm -v eventgraph-data:/data -v "$(pwd)":/backup \
  alpine tar czf /backup/eventgraph-data-backup.tar.gz -C /data .
```

### Step 4 — Stop the old container

```bash
docker stop eventgraphdb
docker rm eventgraphdb
# Do NOT remove the old volume yet
```

### Step 5 — Start the new version with a NEW volume

```bash
docker volume create eventgraph-data-new

docker run -d \
  --name eventgraphdb \
  -p 3000:3000 \
  -v eventgraph-data-new:/data \
  -e RUST_LOG=info \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  ghcr.io/<owner>/eventgraphdb:v1.2.3

# Wait for the new instance to become healthy (empty database)
sleep 10
curl -s http://localhost:3000/api/health | jq .
```

### Step 6 — Import

```bash
curl -X POST "http://localhost:3000/api/admin/import?mode=replace" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @eventgraphdb-export.bin \
  --max-time 3600
```

Expected response:
```json
{
  "success": true,
  "memories_imported": 100,
  "strategies_imported": 50,
  "graph_nodes_imported": 1234,
  "graph_edges_imported": 5678,
  "total_records": 6070,
  "mode": "replace"
}
```

### Import Modes

| Mode | Behavior | Use Case |
|---|---|---|
| `replace` | Wipes the target database, then imports all records. Clean slate. | Version upgrades, fresh migration |
| `merge` | Upserts records — existing keys are overwritten, new keys are added. | Merging data from multiple sources |

### Step 7 — Verify

```bash
# Health check — compare counts to pre-flight baseline
curl -s http://localhost:3000/api/health | jq '{status, version, node_count, edge_count}'

# Spot-check: query a known agent's data
curl -s http://localhost:3000/api/memories/agent/1 | jq '.[0]'
curl -s http://localhost:3000/api/strategies/agent/1 | jq '.[0]'
```

Confirm:
- `node_count` matches the pre-flight value
- `edge_count` matches the pre-flight value
- Agent data is present and correct
- `is_healthy` is `true`

### Step 8 — Clean up

Once verified, remove the old volume:

```bash
docker volume rm eventgraph-data

# Optionally rename the new volume (requires stop/start cycle)
# Or simply update your compose file to use eventgraph-data-new
```

---

## Rollback Procedures

### Scenario A — Import failed, old volume untouched

The safest case. The old volume still has your data.

```bash
docker stop eventgraphdb && docker rm eventgraphdb
docker volume rm eventgraph-data-new

docker run -d \
  --name eventgraphdb \
  -p 3000:3000 \
  -v eventgraph-data:/data \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  ghcr.io/<owner>/eventgraphdb:<old-tag>
```

### Scenario B — Old volume corrupted or deleted

Restore from the tar backup created in Step 3:

```bash
docker volume create eventgraph-data-restored

docker run --rm -v eventgraph-data-restored:/data -v "$(pwd)":/backup \
  alpine sh -c "cd /data && tar xzf /backup/eventgraph-data-backup.tar.gz"

docker run -d \
  --name eventgraphdb \
  -p 3000:3000 \
  -v eventgraph-data-restored:/data \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  ghcr.io/<owner>/eventgraphdb:<old-tag>
```

### Scenario C — New version misbehaves after successful import

You still have the export file. Re-import into a fresh volume on the old version:

```bash
docker stop eventgraphdb && docker rm eventgraphdb
docker volume rm eventgraph-data-new

docker volume create eventgraph-data-rollback

docker run -d \
  --name eventgraphdb \
  -p 3000:3000 \
  -v eventgraph-data-rollback:/data \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  ghcr.io/<owner>/eventgraphdb:<old-tag>

sleep 10

curl -X POST "http://localhost:3000/api/admin/import?mode=replace" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @eventgraphdb-export.bin \
  --max-time 3600
```

---

## Wire Format Reference

The export file uses **Wire v2**, a custom binary format with integrity checking.

```
Header (21 bytes):
  magic[4]       = "EGDB" (0x45 0x47 0x44 0x42)
  version[1]     = 0x02
  record_count[8] = u64 big-endian (advisory; 0 in streaming mode)
  flags[8]       = reserved, must be 0

Records (repeated):
  tag[1]         = record type (see table below)
  key_len[4]     = u32 big-endian
  key[key_len]   = raw bytes
  value_len[4]   = u32 big-endian
  value[value_len] = raw bytes (versioned msgpack envelope)

Footer:
  tag[1]         = 0xFF
  record_count[8] = u64 big-endian (actual count)
  checksum[32]   = SHA-256 of all bytes before footer tag
```

**Record tags:**

| Tag | Type |
|---|---|
| `0x01` | Memory |
| `0x02` | Strategy |
| `0x03` | Graph Node |
| `0x04` | Graph Edge |
| `0x05` | Graph Metadata (singleton) |
| `0x06` | Transition Model (singleton) |
| `0x07` | Episode Detector (singleton) |
| `0x08` | ID Allocator (singleton) |

All integers are big-endian. Record values use the **versioned msgpack envelope**: `[0x00][version:u8][msgpack payload]`. The `0x00` magic byte distinguishes versioned data from legacy unversioned msgpack. Current data version is **2** (named/map msgpack via `rmp_serde::to_vec_named`).

---

## Automation Script

Copy-paste this script for a fully automated export/import upgrade cycle.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
OLD_TAG="${OLD_TAG:?Set OLD_TAG to current image tag}"
NEW_TAG="${NEW_TAG:?Set NEW_TAG to target image tag}"
IMAGE="${IMAGE:-ghcr.io/<owner>/eventgraphdb}"
CONTAINER="eventgraphdb"
PORT="${PORT:-3000}"
EXPORT_FILE="eventgraphdb-export-$(date +%Y%m%d-%H%M%S).bin"
OLD_VOLUME="eventgraph-data"
NEW_VOLUME="eventgraph-data-new"
BASE_URL="http://localhost:${PORT}"

# ── Helpers ──────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; }

wait_healthy() {
  local url="$1" retries=30
  for i in $(seq 1 $retries); do
    if curl -fsS "$url/api/health" >/dev/null 2>&1; then
      info "Health check passed (attempt $i/$retries)"
      return 0
    fi
    sleep 2
  done
  error "Health check failed after $retries attempts"
  return 1
}

# ── Pre-flight ───────────────────────────────────────────────────
info "Recording pre-flight health baseline..."
BASELINE=$(curl -fsS "$BASE_URL/api/health")
BASELINE_NODES=$(echo "$BASELINE" | jq -r '.node_count')
BASELINE_EDGES=$(echo "$BASELINE" | jq -r '.edge_count')
info "Baseline: node_count=$BASELINE_NODES, edge_count=$BASELINE_EDGES"

# ── Export ───────────────────────────────────────────────────────
info "Exporting data to $EXPORT_FILE ..."
curl -X POST "$BASE_URL/api/admin/export" \
  --max-time 3600 \
  --fail \
  --output "$EXPORT_FILE"

MAGIC=$(xxd -l 4 -p "$EXPORT_FILE")
if [[ "$MAGIC" != "45474442" ]]; then
  error "Export file does not start with EGDB magic bytes. Aborting."
  exit 1
fi
info "Export verified: $(ls -lh "$EXPORT_FILE" | awk '{print $5}') with valid EGDB header"

# ── Backup old volume ───────────────────────────────────────────
info "Backing up old volume..."
docker run --rm -v "$OLD_VOLUME":/data -v "$(pwd)":/backup \
  alpine tar czf "/backup/${OLD_VOLUME}-backup.tar.gz" -C /data .

# ── Stop old container ──────────────────────────────────────────
info "Stopping old container..."
docker stop "$CONTAINER" && docker rm "$CONTAINER"

# ── Pull and start new version ──────────────────────────────────
info "Pulling $IMAGE:$NEW_TAG ..."
docker pull "$IMAGE:$NEW_TAG"

docker volume create "$NEW_VOLUME"

docker run -d \
  --name "$CONTAINER" \
  -p "$PORT:3000" \
  -v "$NEW_VOLUME":/data \
  -e RUST_LOG=info \
  -e STORAGE_BACKEND=persistent \
  -e REDB_PATH=/data/eventgraph.redb \
  --restart unless-stopped \
  "$IMAGE:$NEW_TAG"

wait_healthy "$BASE_URL"

# ── Import ───────────────────────────────────────────────────────
info "Importing data (mode=replace)..."
IMPORT_RESULT=$(curl -X POST "$BASE_URL/api/admin/import?mode=replace" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@$EXPORT_FILE" \
  --max-time 3600 \
  --fail -sS)

echo "$IMPORT_RESULT" | jq .

# ── Verify ───────────────────────────────────────────────────────
info "Verifying post-import health..."
POST_HEALTH=$(curl -fsS "$BASE_URL/api/health")
POST_NODES=$(echo "$POST_HEALTH" | jq -r '.node_count')
POST_EDGES=$(echo "$POST_HEALTH" | jq -r '.edge_count')

if [[ "$POST_NODES" -eq "$BASELINE_NODES" && "$POST_EDGES" -eq "$BASELINE_EDGES" ]]; then
  info "Verification PASSED: node_count=$POST_NODES, edge_count=$POST_EDGES"
else
  error "Verification FAILED: expected nodes=$BASELINE_NODES edges=$BASELINE_EDGES, got nodes=$POST_NODES edges=$POST_EDGES"
  error "Old volume '$OLD_VOLUME' is still available for rollback."
  exit 1
fi

info "Upgrade complete. Old volume '$OLD_VOLUME' retained as safety net."
info "Run 'docker volume rm $OLD_VOLUME' after confirming everything works."
```

Usage:

```bash
OLD_TAG=v1.0.0 NEW_TAG=v2.0.0 IMAGE=ghcr.io/yourname/eventgraphdb bash upgrade.sh
```

---

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `3000` | HTTP port |
| `RUST_LOG` | `info` | Log level (`trace`, `debug`, `info`, `warn`, `error`) |
| `STORAGE_BACKEND` | `persistent` | `persistent` (redb) or `memory` |
| `REDB_PATH` | `/data/eventgraph.redb` | Path to redb database file |
| `REDB_CACHE_SIZE_MB` | `256` (normal) / `64` (free) | ReDB page cache size |
| `MEMORY_CACHE_SIZE` | `10000` (normal) / `1000` (free) | In-memory memory cache entries |
| `STRATEGY_CACHE_SIZE` | `5000` (normal) / `500` (free) | In-memory strategy cache entries |
| `SERVICE_PROFILE` | `normal` | `normal` or `free` (controls default cache sizes and limits) |
| `ENABLE_LOUVAIN` | `true` | Enable Louvain community detection (normal profile only) |
| `LOUVAIN_INTERVAL` | `1000` | Events between Louvain recalculations |
| `COMMUNITY_ALGORITHM` | `louvain` | Community detection algorithm |
| `NER_SERVICE_URL` | `http://localhost:8081/ner` | External NER service URL |
| `NER_REQUEST_TIMEOUT_MS` | `5000` | NER request timeout (ms) |
| `LLM_API_KEY` | *(none)* | API key for LLM provider |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `PLANNING_LLM_API_KEY` | falls back to `LLM_API_KEY` | Separate API key for planning |
| `PLANNING_LLM_PROVIDER` | `openai` | LLM provider for planning |
| `ENABLE_WORLD_MODEL` | `false` | Enable energy-based world model |
| `WORLD_MODEL_MODE` | `disabled` | `disabled`, `shadow`, `scoring`, `reranking`, `full` |
| `ENABLE_STRATEGY_GENERATION` | `false` | Enable LLM strategy generation |
| `ENABLE_ACTION_GENERATION` | `false` | Enable LLM action generation |
| `GENERATION_MODE` | `disabled` | `disabled`, `generate_only`, `generate_and_score`, `generate_score_and_select`, `full` |
| `ENABLE_REPAIR` | `false` | Enable automatic repair |

---

## Troubleshooting

### "Database requires migration: on-disk major=X, current=Y" (`NeedsMigration`)

The on-disk schema is older than what the new binary expects. You need the full export/import cycle:

1. Stop the new container.
2. Start the **old** version with the existing volume.
3. Export data via `POST /api/admin/export`.
4. Follow the [Full Export/Import Cycle](#full-exportimport-cycle) using the new version.

### "Database was created by a newer version: on-disk major=X, current=Y" (`TooNew`)

You are trying to run an **older** binary against a database written by a **newer** version. Downgrading across `SCHEMA_MAJOR` boundaries is not supported. Either:
- Use the newer binary version that matches the database.
- Export from the newer version and import into the older version (only works if the older version can parse the Wire v2 export file).

### "Checksum mismatch" on import

The export file is corrupted — a SHA-256 digest protects the entire wire stream. Causes:
- Incomplete download (network timeout, disk full).
- File was modified or truncated after export.

Fix: re-export from the source instance. Use `--max-time 3600` for large databases.

### Export returns HTTP 500

Ensure `STORAGE_BACKEND=persistent`. Export is not supported with the in-memory backend because there is no durable data to export.

### Export/import times out on large databases

Increase the curl timeout:

```bash
curl -X POST http://localhost:3000/api/admin/export \
  --max-time 3600 \
  --output eventgraphdb-export.bin
```

For very large databases (millions of nodes), consider `--max-time 7200` (2 hours).

### Container starts but health check fails

Check logs for startup errors:

```bash
docker logs eventgraphdb --tail 50
```

Common causes:
- `REDB_PATH` points to a non-existent directory (the container only creates `/data` by default).
- Permission issues — the container runs as user `eventgraph` (UID 1000). Ensure the volume is writable.
- Port conflict — another process is using port 3000.

### Import succeeds but counts don't match

The `node_count` and `edge_count` from `/api/health` reflect the **in-memory graph** which is loaded from the redb storage. After import, the graph engine reloads from storage. If counts still differ:
- Check the import response — it reports exactly how many of each record type were imported.
- The graph may need a moment to fully re-index. Wait 10-30 seconds and check again.
