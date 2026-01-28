# Setup Guide: NER Service on Separate Server

Since your NER service runs on a **separate machine**, you need to configure EventGraphDB to connect to the external IP/hostname.

## Quick Setup

### 1. Find Your NER Service Address

First, determine the IP address or hostname where your NER service is running:

```bash
# On the NER service machine, find its IP
ip addr show  # Linux
ipconfig      # Windows

# Or if it has a hostname
hostname -f
```

**Example addresses:**
- Private IP: `192.168.1.100`
- Public IP: `203.0.113.5`
- Hostname: `ner-service.mydomain.com`

### 2. Test NER Service is Accessible

From the **EventGraphDB machine**, test connectivity:

```bash
# Replace with your actual NER service IP/hostname
curl http://192.168.1.100:8081/health

# Should return:
# {"status": "healthy", "model": "..."}
```

**If this doesn't work:**
- Check firewall on NER machine allows port 8081
- Ensure NER service is listening on `0.0.0.0` not `127.0.0.1`
- Check network connectivity between machines

### 3. Create .env File

On the **EventGraphDB machine**, create a `.env` file:

```bash
cd /path/to/EventGraphDB
nano .env
```

Add this content (replace with your actual IP):

```bash
# .env
# Replace with your NER service IP or hostname
NER_SERVICE_URL=http://192.168.1.100:8081/ner

# Optional: increase timeout if network is slow
NER_REQUEST_TIMEOUT_MS=5000

# Your OpenAI key
OPENAI_API_KEY=sk-your-actual-key-here
LLM_MODEL=gpt-4o-mini
```

**IMPORTANT:** Replace `192.168.1.100` with your actual NER service address!

### 4. Start EventGraphDB

```bash
# docker-compose will automatically read .env file
docker-compose up -d

# Check logs to verify connection
docker logs eventgraphdb | grep NER
```

You should see:
```
✓ Semantic memory enabled
  NER workers: 2
```

### 5. Test End-to-End

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
      "event_type": {"Communication": {"message": "Apple Inc. was founded by Steve Jobs in California."}},
      "context": {"fingerprint": 0, "location": "", "goal": "", "tools_available": []}
    },
    "enable_semantic": true
  }'

# Check if entities were extracted
docker logs eventgraphdb | grep -i "entity\|claim"
```

---

## For GitHub Actions / CI/CD

If you're deploying from GitHub Actions, you need to set a **GitHub Secret**.

### Step 1: Add GitHub Secret

1. Go to your repo: `https://github.com/YOUR_USERNAME/EventGraphDB/settings/secrets/actions`
2. Click **"New repository secret"**
3. Add:
   - **Name:** `NER_SERVICE_URL`
   - **Value:** `http://192.168.1.100:8081/ner` (your actual address)
4. Click **"Add secret"**

### Step 2: Update GitHub Workflow

Update `.github/workflows/publish-ghcr.yml`:

```yaml
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    build-args: |
      SERVICE_PROFILE=${{ matrix.profile }}
    tags: ${{ steps.meta.outputs.tags }}
    labels: ${{ steps.meta.outputs.labels }}
    # Add this section for runtime environment
    secrets: |
      NER_SERVICE_URL=${{ secrets.NER_SERVICE_URL }}
```

Or if you're using docker run in your workflow:

```yaml
- name: Deploy to production
  run: |
    docker run -d \
      -e NER_SERVICE_URL=${{ secrets.NER_SERVICE_URL }} \
      -e OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }} \
      -e NER_REQUEST_TIMEOUT_MS=5000 \
      -p 8080:8080 \
      -v eventgraph-data:/data \
      ghcr.io/${{ github.repository_owner }}/eventgraphdb:latest
```

---

## Network Configuration Options

### Option 1: Private Network (Recommended if on same network)

If both servers are on the same private network:

```bash
# .env
NER_SERVICE_URL=http://192.168.1.100:8081/ner  # Private IP
```

**Pros:**
- ✅ Fast (no internet routing)
- ✅ Secure (not exposed to internet)
- ✅ No bandwidth costs

**Cons:**
- ❌ Both servers must be on same network

### Option 2: Public IP (If servers on different networks)

If NER service has a public IP:

```bash
# .env
NER_SERVICE_URL=http://203.0.113.5:8081/ner  # Public IP
```

**Pros:**
- ✅ Works across different networks
- ✅ Simple setup

**Cons:**
- ❌ Must expose port 8081 to internet
- ❌ Security risk (add authentication!)
- ❌ Bandwidth costs

**Security:**
```bash
# On NER machine, restrict access with firewall
sudo ufw allow from 203.0.113.10 to any port 8081  # Only allow EventGraphDB IP
```

### Option 3: Domain Name (Most Professional)

Set up a domain/subdomain pointing to NER service:

```bash
# .env
NER_SERVICE_URL=https://ner.mydomain.com/ner  # With SSL
```

**Pros:**
- ✅ Professional setup
- ✅ Can use HTTPS (encrypted)
- ✅ Easy to change IP without reconfiguring

**Cons:**
- ❌ Requires domain and SSL certificate
- ❌ More setup

### Option 4: VPN Tunnel (Most Secure)

Use VPN to connect servers securely:

```bash
# .env
NER_SERVICE_URL=http://10.8.0.2:8081/ner  # VPN IP
```

**Pros:**
- ✅ Very secure (encrypted tunnel)
- ✅ Works across networks
- ✅ No port exposure needed

**Cons:**
- ❌ Requires VPN setup (WireGuard, OpenVPN, etc.)

---

## Troubleshooting

### Error: "Failed to connect to NER service"

**Step 1:** Test from EventGraphDB container
```bash
docker exec -it eventgraphdb curl http://YOUR_NER_IP:8081/health
```

**Step 2:** Check environment variable is set
```bash
docker exec -it eventgraphdb env | grep NER_SERVICE_URL
```

**Step 3:** Check NER service logs
```bash
# On NER machine
docker logs ner-service
```

### Error: "Connection timeout"

**Cause:** Network latency or firewall blocking

**Solution 1:** Increase timeout
```bash
# .env
NER_REQUEST_TIMEOUT_MS=10000  # 10 seconds
```

**Solution 2:** Check firewall
```bash
# On NER machine
sudo ufw status
sudo ufw allow 8081
```

**Solution 3:** Verify NER is listening on correct interface
```bash
# On NER machine, check NER is listening on 0.0.0.0, not 127.0.0.1
netstat -tulpn | grep 8081
```

### Error: "NER service returns 404"

**Cause:** Wrong endpoint path

**Solution:** Verify exact endpoint
```bash
# Test directly
curl -X POST http://YOUR_NER_IP:8081/ner \
  -H "Content-Type: application/json" \
  -d '{"text":"test"}'
```

If it's different (e.g., `/api/ner` instead of `/ner`), update:
```bash
# .env
NER_SERVICE_URL=http://YOUR_NER_IP:8081/api/ner  # Adjust path
```

---

## Performance Tuning

### If NER service is on slow network:

```bash
# .env
# Increase timeout
NER_REQUEST_TIMEOUT_MS=10000

# Reduce retries (in code config)
# See: crates/agent-db-graph/src/integration.rs
# max_retries: 3 → max_retries: 1
```

### If NER service is on fast network:

```bash
# .env
# Decrease timeout for faster failure detection
NER_REQUEST_TIMEOUT_MS=2000

# Can increase workers for higher throughput
# Note: This is set in code, not env variable
```

---

## Security Best Practices

### 1. Use Private Network
Keep both services on same private network if possible.

### 2. Firewall Rules
```bash
# On NER machine
sudo ufw allow from YOUR_EVENTGRAPH_IP to any port 8081
sudo ufw deny 8081  # Deny all others
```

### 3. HTTPS (Recommended for production)
Use reverse proxy with SSL:
```bash
# .env
NER_SERVICE_URL=https://ner.mydomain.com/ner
```

### 4. Authentication (Advanced)
Add API key to NER service and pass in requests.

---

## Example Deployment Configurations

### Production: AWS

```yaml
# EventGraphDB on EC2 instance 1
# NER Service on EC2 instance 2 (same VPC)

# .env
NER_SERVICE_URL=http://10.0.1.50:8081/ner  # Private VPC IP
NER_REQUEST_TIMEOUT_MS=5000

# Security group rules:
# - EventGraphDB SG: Allow outbound to 10.0.1.50:8081
# - NER SG: Allow inbound from EventGraphDB SG on port 8081
```

### Production: Docker Swarm

```yaml
# docker-compose.yml
services:
  eventgraphdb:
    environment:
      - NER_SERVICE_URL=http://ner-service:8081/ner  # Swarm overlay network
    deploy:
      replicas: 2
    networks:
      - swarm-network

  ner-service:
    deploy:
      replicas: 1
    networks:
      - swarm-network

networks:
  swarm-network:
    driver: overlay
```

### Production: Kubernetes

```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: eventgraphdb-config
data:
  NER_SERVICE_URL: "http://ner-service.ner-namespace.svc.cluster.local:8081/ner"

# Deployment
spec:
  containers:
  - name: eventgraphdb
    envFrom:
    - configMapRef:
        name: eventgraphdb-config
```

---

## Summary Checklist

For NER service on **separate box**, you need:

- [ ] Find NER service IP/hostname
- [ ] Test connectivity: `curl http://NER_IP:8081/health`
- [ ] Create `.env` file with `NER_SERVICE_URL=http://NER_IP:8081/ner`
- [ ] Start EventGraphDB: `docker-compose up -d`
- [ ] Check logs: `docker logs eventgraphdb | grep NER`
- [ ] Test end-to-end with event processing

For GitHub deployment:
- [ ] Add `NER_SERVICE_URL` as GitHub secret
- [ ] Update workflow to use secret (if needed)

**That's it!** Your EventGraphDB will now connect to the external NER service. 🚀
