# EventGraphDB Integration Testing Guide

This guide explains how to test the complete EventGraphDB stack (Rust server + JavaScript client).

## Prerequisites

1. Rust server built: `cargo build --release`
2. JavaScript client built: `cd client && npm install && npm run build`
3. Node.js installed (v16+)

## Running the Integration Tests

### Step 1: Start the Rust Server

In one terminal:

```bash
cargo run --release --bin eventgraphdb-server
```

Or:

```bash
./target/release/eventgraphdb-server
```

Wait for the server to start. You should see:

```
🚀 Starting EventGraphDB REST API Server
Initializing GraphEngine with automatic self-evolution...
✓ GraphEngine initialized
🌐 Server listening on http://127.0.0.1:3000
📚 API documentation: http://127.0.0.1:3000/docs
❤️  Health check: http://127.0.0.1:3000/api/health
```

### Step 2: Run the Integration Tests

In another terminal:

```bash
node test-integration.js
```

## What the Tests Verify

The integration test suite verifies:

1. **Health Check** - Server is running and responsive
2. **Event Processing** - Single event can be processed
3. **Batch Processing** - Multiple events can be processed
4. **Memory Retrieval** - Agent memories can be queried
5. **Strategy Retrieval** - Learned strategies can be retrieved
6. **Episode Retrieval** - Completed episodes can be queried
7. **System Statistics** - Overall system stats are accessible
8. **Action Suggestions** - Policy guide returns recommendations

## Expected Output

```
🧪 EventGraphDB Integration Test Suite

============================================================

1️⃣  Testing health check endpoint...
   ✅ Server healthy: v0.1.0
   📊 Graph size: 0 nodes, 0 edges

2️⃣  Testing event processing...
   ✅ Event processed successfully
   📦 Created 3 nodes
   🔍 Detected 0 patterns
   ⏱️  Processing time: 15ms

3️⃣  Processing multiple events to build graph...
   ✅ Processed 2 events successfully

4️⃣  Testing memory retrieval...
   ✅ Retrieved 2 memories
   📝 Memory 1:
      - Strength: 0.85
      - Relevance: 0.90
      - Accessed: 1 times

5️⃣  Testing strategy retrieval...
   ✅ Retrieved 1 strategies
   📋 Strategy: "Null Check Pattern"
      - Quality: 0.87
      - Success rate: 100.0%
      - Steps: 3

6️⃣  Testing episode retrieval...
   ✅ Retrieved 1 episodes
   📺 Episode 1:
      - Events: 3
      - Significance: 0.75

7️⃣  Testing system statistics...
   ✅ System statistics retrieved
   📊 Total events processed: 3
   🧠 Memories formed: 2
   📋 Strategies extracted: 1
   🔁 Reinforcements applied: 0
   📺 Episodes detected: 1
   ⏱️  Avg processing time: 14.50ms

8️⃣  Testing action suggestions (Policy Guide)...
   ✅ Retrieved 2 action suggestions
   💡 Top suggestions for "null_reference_error":
      1. add_null_check (85.0% success)
         Evidence: 1 cases
      2. add_try_catch (70.0% success)
         Evidence: 1 cases

============================================================

📊 Test Results:
   ✅ Passed: 8
   ❌ Failed: 0
   📈 Success Rate: 100.0%

🎉 All tests passed! EventGraphDB is working correctly!
```

## Troubleshooting

### Error: "connect ECONNREFUSED"

**Problem**: Server is not running or not accessible.

**Solution**:
1. Make sure the server is running on port 3000
2. Check for firewall blocking localhost:3000
3. Verify server logs for errors

### Error: "Request timeout"

**Problem**: Server is slow or unresponsive.

**Solution**:
1. Check server logs for errors
2. Increase timeout in test-integration.js (currently 10s)
3. Check system resources (CPU, memory)

### Tests Pass But No Data Returned

**Problem**: Server is working but not storing/retrieving data.

**Solution**:
1. Check that GraphEngine initialization succeeded
2. Verify storage engine is working
3. Check server logs for warnings

## Manual Testing

You can also test the API manually using curl:

### Health Check
```bash
curl http://localhost:3000/api/health
```

### Process Event
```bash
curl -X POST http://localhost:3000/api/events \
  -H "Content-Type: application/json" \
  -d @event.json
```

### Get Agent Memories
```bash
curl "http://localhost:3000/api/memories/agent/1?limit=10"
```

### Get Agent Strategies
```bash
curl "http://localhost:3000/api/strategies/agent/1?limit=10"
```

### Get Action Suggestions
```bash
curl "http://localhost:3000/api/suggestions?context_hash=null_reference_error&limit=5"
```

### Get Episodes
```bash
curl "http://localhost:3000/api/episodes?limit=10"
```

### Get Statistics
```bash
curl http://localhost:3000/api/stats
```

## Next Steps

After successful integration testing:

1. **Load Testing**: Test with high event volumes
2. **Stress Testing**: Test system limits
3. **Concurrent Testing**: Test multiple agents simultaneously
4. **Persistence Testing**: Verify data survives server restarts
5. **Performance Profiling**: Measure and optimize bottlenecks

## JavaScript Client Usage

See `client/README.md` for detailed JavaScript client documentation.

Example usage:

```javascript
const { EventGraphDBClient } = require('./client/dist/index');

const client = new EventGraphDBClient({
  baseUrl: 'http://127.0.0.1:3000'
});

// Process event
const result = await client.processEvent(event);

// Get memories
const memories = await client.getAgentMemories(1, 10);

// Get suggestions
const suggestions = await client.getActionSuggestions('context_hash', undefined, 5);
```

## Continuous Integration

To run tests in CI:

```bash
# Start server in background
cargo run --release --bin eventgraphdb-server &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Run tests
node test-integration.js

# Cleanup
kill $SERVER_PID
```
