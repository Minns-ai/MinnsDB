# NER Service Deployment Configuration

## Overview

EventGraphDB requires a separate NER (Named Entity Recognition) service for semantic memory features. Here's how to configure it for different deployment scenarios.

## Deployment Scenarios

### 1. Local Development (docker-compose)

**No environment variable needed!** The docker-compose.yml handles this automatically.

```yaml
services:
  ner-service:
    image: your-ner-service:latest
    container_name: ner-service
    ports:
      - "8081:8081"
    networks:
      - eventgraph-network

  eventgraphdb:
    environment:
      - NER_SERVICE_URL=http://ner-service:8081/ner  # Uses Docker internal network
```

**Start both services:**
```bash
docker-compose up -d
```

Docker's internal networking resolves `ner-service` automatically! ✅

---

### 2. GitHub Actions / CI/CD

**For building:** No environment variable needed during build.

**For testing/deployment:** Set in workflow if running integration tests.

Update `.github/workflows/publish-ghcr.yml` if you want to test with NER:

```yaml
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    env:
      # Only needed if running integration tests in CI
      NER_SERVICE_URL: http://mock-ner-service:8081/ner
```

**You don't need to set GitHub Secrets** unless:
- Running integration tests in CI that need a real NER service
- Deploying to a hosted environment from GitHub Actions

---

### 3. Production Deployment

#### Option A: Both services in same Docker network (Recommended)

Use docker-compose in production:

```bash
# No environment variables needed!
docker-compose -f docker-compose.yml up -d
```

The services communicate via Docker's internal network.

#### Option B: Separate infrastructure

If NER service is hosted separately (different server/cloud):

```bash
# Set environment variable pointing to external NER service
docker run -d \
  -e NER_SERVICE_URL=http://ner-service.your-domain.com:8081/ner \
  -e NER_REQUEST_TIMEOUT_MS=5000 \
  -p 8080:8080 \
  ghcr.io/your-org/eventgraphdb:latest
```

#### Option C: Kubernetes

```yaml
# deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: ner-service
spec:
  selector:
    app: ner-service
  ports:
    - port: 8081

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eventgraphdb
spec:
  template:
    spec:
      containers:
      - name: eventgraphdb
        image: ghcr.io/your-org/eventgraphdb:latest
        env:
        - name: NER_SERVICE_URL
          value: "http://ner-service:8081/ner"  # Uses k8s service DNS
```

---

## Environment Variables

### Required

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NER_SERVICE_URL` | Full URL to NER endpoint | `http://localhost:8081/ner` | `http://ner-service:8081/ner` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `NER_REQUEST_TIMEOUT_MS` | Request timeout in milliseconds | `5000` |
| `NER_MODEL` | Specific model to request | `None` |

### Example .env file

```bash
# .env
NER_SERVICE_URL=http://ner-service:8081/ner
NER_REQUEST_TIMEOUT_MS=5000
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

---

## Testing the Connection

### 1. Check NER service is running

```bash
curl http://localhost:8081/health
# Should return: {"status": "healthy", "model": "..."}
```

### 2. Test NER extraction

```bash
curl -X POST http://localhost:8081/ner \
  -H "Content-Type: application/json" \
  -d '{"text":"Apple Inc. was founded by Steve Jobs."}'
```

### 3. Check EventGraphDB can reach NER

```bash
# Process an event with semantic memory
curl -X POST http://localhost:8080/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event": {
      "id": 1,
      "agent_id": 1,
      "session_id": 1,
      "timestamp": 1234567890,
      "event_type": {"Communication": {"message": "Apple Inc. was founded by Steve Jobs."}},
      "context": {"fingerprint": 0, "location": "", "goal": "", "tools_available": []}
    },
    "enable_semantic": true
  }'

# Check logs for NER activity
docker logs eventgraphdb | grep NER
```

---

## Docker Network Configuration

### Internal Communication (Recommended)

When using docker-compose, services communicate via internal network:

```
ner-service:8081 ← eventgraphdb
    ↑                    ↓
Internal Docker Network
```

**Benefits:**
- ✅ No external IP needed
- ✅ No firewall rules needed
- ✅ Faster (no network overhead)
- ✅ More secure (not exposed)

### External Communication

Only needed if services on different hosts:

```
NER Service              EventGraphDB
(Server A)              (Server B)
    ↑                       ↓
  Public IP: 203.0.113.5
```

Set: `NER_SERVICE_URL=http://203.0.113.5:8081/ner`

---

## GitHub Secrets (Optional)

**You only need GitHub Secrets if:**
- Deploying from GitHub Actions to production
- NER service requires authentication
- Using paid NER services

### To add a GitHub Secret:

1. Go to: `https://github.com/YOUR_ORG/eventgraphdb/settings/secrets/actions`
2. Click "New repository secret"
3. Name: `NER_SERVICE_URL`
4. Value: `http://your-ner-service.com:8081/ner`

Then use in workflow:

```yaml
- name: Deploy
  env:
    NER_SERVICE_URL: ${{ secrets.NER_SERVICE_URL }}
```

**But for most cases, you DON'T need this!** ✅

---

## Quick Start Checklist

### Local Development with docker-compose

- [x] Update docker-compose.yml (already done ✅)
- [ ] Build/pull NER service image: `docker pull your-ner-image`
- [ ] Start services: `docker-compose up -d`
- [ ] Test NER: `curl http://localhost:8081/health`
- [ ] Test EventGraphDB: `curl http://localhost:8080/health`

**No environment variables needed!** 🎉

### Production Deployment

**If using docker-compose:**
- [ ] Same as local development
- [ ] No extra config needed

**If NER is on different server:**
- [ ] Note NER service IP/hostname
- [ ] Set `NER_SERVICE_URL=http://ner-ip:8081/ner`
- [ ] Test connectivity from EventGraphDB server

---

## Troubleshooting

### Error: "Failed to connect to NER service"

**Check 1:** Is NER service running?
```bash
docker ps | grep ner-service
curl http://localhost:8081/health
```

**Check 2:** Can EventGraphDB reach NER?
```bash
# From inside EventGraphDB container
docker exec -it eventgraphdb curl http://ner-service:8081/health
```

**Check 3:** Check NER_SERVICE_URL
```bash
docker exec -it eventgraphdb env | grep NER
```

### Error: "Connection timeout"

**Solution:** Increase timeout
```bash
export NER_REQUEST_TIMEOUT_MS=10000
```

### NER service takes too long to start

**Solution:** Use `depends_on` with health check (already configured in docker-compose.yml ✅)

---

## Summary

### For most users: **No configuration needed!** ✅

Just use docker-compose:
```bash
docker-compose up -d
```

The NER service URL is automatically configured via Docker networking.

### Only set environment variables if:
- ❌ NER service on different server
- ❌ Custom NER endpoint
- ❌ Special deployment (k8s, cloud)

### GitHub Secrets?
- ❌ Not needed for building
- ❌ Not needed for docker-compose
- ✅ Only if deploying from GitHub Actions to production
