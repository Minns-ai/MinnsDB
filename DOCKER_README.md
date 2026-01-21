# EventGraphDB - Docker Deployment Guide

Quick guide for running EventGraphDB in Docker.

## Quick Start

### 1. Build Image

```bash
docker build -t eventgraphdb:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name eventgraphdb \
  -p 8080:8080 \
  -p 9090:9090 \
  -v eventgraph-data:/data \
  eventgraphdb:latest
```

### 3. Verify

```bash
curl http://localhost:8080/health
```

## Using Docker Compose

### Basic Deployment

```bash
# Start server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
docker-compose down
```

### With Monitoring (Prometheus + Grafana)

```bash
# Start server + monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## Configuration

### Environment Variables

```bash
docker run -d \
  --name eventgraphdb \
  -p 8080:8080 \
  -e RUST_LOG=debug \
  -e REDB_CACHE_SIZE_MB=512 \
  -e MEMORY_CACHE_SIZE=50000 \
  -v eventgraph-data:/data \
  eventgraphdb:latest
```

### Available Variables

| Variable | Default | Description |
|----------|---------|-------------|
| RUST_LOG | info | Log level (debug, info, warn, error) |
| SERVER_HOST | 0.0.0.0 | Server bind address |
| SERVER_PORT | 8080 | Server port |
| STORAGE_BACKEND | persistent | Storage type (persistent or in_memory) |
| REDB_PATH | /data/eventgraph.redb | Database file path |
| REDB_CACHE_SIZE_MB | 256 | Database cache size (MB) |
| MEMORY_CACHE_SIZE | 10000 | In-memory cache entries |
| STRATEGY_CACHE_SIZE | 5000 | Strategy cache entries |

## Persistent Data

Data is stored in `/data` volume inside the container.

### Backup Data

```bash
# Create backup
docker run --rm \
  -v eventgraph-data:/data \
  -v $(pwd)/backups:/backup \
  busybox \
  tar czf /backup/eventgraph-$(date +%Y%m%d).tar.gz /data
```

### Restore Data

```bash
# Restore from backup
docker run --rm \
  -v eventgraph-data:/data \
  -v $(pwd)/backups:/backup \
  busybox \
  tar xzf /backup/eventgraph-20260120.tar.gz -C /
```

## Production Deployment

### Using Docker Swarm

```bash
docker stack deploy -c docker-compose.yml eventgraph
```

### Using Kubernetes

See `k8s/` directory for Kubernetes manifests.

## Health Check

The image includes a built-in health check:

```bash
docker inspect --format='{{.State.Health.Status}}' eventgraphdb
```

## Troubleshooting

### View Logs

```bash
docker logs eventgraphdb --follow
```

### Shell Access

```bash
docker exec -it eventgraphdb /bin/bash
```

### Check Disk Usage

```bash
docker exec eventgraphdb du -sh /data
```

### Reset Data

```bash
docker-compose down -v  # Removes volumes!
docker-compose up -d
```

## Image Details

- Base: Debian Bookworm Slim
- Size: ~100 MB (compressed)
- Non-root user: eventgraph (uid 1000)
- Exposed ports: 8080 (API), 9090 (Metrics)

## Security

- Runs as non-root user
- Minimal attack surface (slim base image)
- No unnecessary packages
- TLS termination handled by reverse proxy (recommended)

## Performance Tuning

### For High Throughput

```bash
docker run -d \
  --name eventgraphdb \
  --cpus 4 \
  --memory 8g \
  -e REDB_CACHE_SIZE_MB=1024 \
  -e MEMORY_CACHE_SIZE=100000 \
  -p 8080:8080 \
  eventgraphdb:latest
```

### For Low Latency

```bash
docker run -d \
  --name eventgraphdb \
  --cpus 8 \
  --memory 16g \
  -e REDB_CACHE_SIZE_MB=2048 \
  -e PERSISTENCE_INTERVAL=100 \
  -p 8080:8080 \
  eventgraphdb:latest
```

## Multi-Container Setup

For production, use multiple containers with load balancing:

```yaml
# docker-compose.prod.yml
services:
  eventgraph-1:
    image: eventgraphdb:latest
    # ...

  eventgraph-2:
    image: eventgraphdb:latest
    # ...

  nginx:
    image: nginx:alpine
    # Load balancer config
```

## Monitoring

Access metrics at `http://localhost:9090/metrics`

Example Prometheus query:
```promql
rate(eventgraph_events_total[5m])
```

## Support

For issues, see:
- Full documentation: `SYSTEM_SPECIFICATION.md`
- API reference: `API_SPECIFICATION.md`
- Deployment guide: `HERTZ_DEPLOYMENT_PLAN.md`
