# EventGraphDB Visualizer Guide

## 🎉 Your visualizer is ready!

A real-time web-based interface to explore your EventGraphDB knowledge graph.

## Quick Start

### 1. Make sure the server is running

```bash
# Server should already be running on http://127.0.0.1:3000
# If not, start it with:
cargo run --release --bin eventgraphdb-server
```

### 2. Open the visualizer

**Windows:**
```bash
# Double-click this file:
start-visualizer.bat

# Or manually open:
client\visualizer\index.html
```

**Mac/Linux:**
```bash
open client/visualizer/index.html        # Mac
xdg-open client/visualizer/index.html    # Linux
```

### 3. Explore!

The visualizer will automatically:
- Connect to your server
- Load the graph structure
- Display nodes and edges
- Auto-refresh every 5 seconds

## What You'll See

```
┌──────────────────────────────────────────────────────────────┐
│  🕸️ EventGraphDB Visualizer                                   │
│  ● Connected   3 events • 6 nodes • 5 edges                  │
├──────────┬─────────────────────────────────┬─────────────────┤
│ Events   │  Interactive Graph              │   Details       │
│          │                                  │                 │
│ Event 1  │        ●───────●                │  Node: #2       │
│ Event 2  │       /│\     /│\               │  Type: Event    │
│ Event 3  │      ● ● ●   ● ●                │  Label:         │
│          │       \│/     \│/                │  add_null_check │
│          │        ●───────●                 │                 │
│          │                                  │  Edges:         │
│          │  [Refresh] [Fit] [Physics]      │  • Contextual   │
│          │                                  │  • Causal       │
└──────────┴─────────────────────────────────┴─────────────────┘
```

## Features

### Interactive Graph

- **Zoom**: Mouse wheel or pinch
- **Pan**: Click and drag background
- **Select**: Click any node to see details
- **Physics**: Nodes automatically organize
- **Colors**:
  - 🟣 **Purple**: Event nodes
  - 🟢 **Green**: Context nodes
  - 🟠 **Orange**: Action nodes

### Real-Time Updates

- **Auto-Refresh**: Updates every 5 seconds
- **Live Status**: Connection indicator in header
- **Stats Display**: Event count, node count, edge count

### Node Details

Click any node to see:
- Node ID and type
- Creation timestamp
- Incoming/outgoing connections
- Properties (JSON)

## Controls

| Button | Action |
|--------|--------|
| 🔄 Refresh | Manually refresh data |
| 📐 Fit | Center and fit graph |
| ⚡ Physics | Toggle physics simulation |

## Example Workflow

### 1. Generate Data

Run the integration tests to create events:

```bash
node test-integration.js
```

This creates 3 events with relationships.

### 2. View the Graph

Open the visualizer - you should see:
- **3 event nodes** (purple dots)
- **3 context nodes** (green dots)
- **5 edges** connecting them
- **Status bar** showing "3 events • 6 nodes • 5 edges"

### 3. Explore

- **Click a node** → See its properties
- **Watch edges** → See relationships (Causal, Contextual, Temporal)
- **Zoom in/out** → Examine details
- **Click Fit** → Center view

### 4. Add More Data

Process more events through the API:

```bash
curl -X POST http://localhost:3000/api/events \
  -H "Content-Type: application/json" \
  -d @your-event.json
```

Wait 5 seconds (or click Refresh) to see new nodes appear!

## Graph Legend

### Node Types

- **Event**: Actions, observations, cognitive processes
- **Context**: Environment snapshots
- **Agent**: Agent identifiers
- **Goal**: Agent objectives

### Edge Types

- **Temporal**: A happens before B (time-based)
- **Causal**: A causes B (cause-effect)
- **Contextual**: A and B share context (co-occurrence)
- **Interaction**: Agent interactions

### Edge Properties

- **Thickness**: Represents edge weight (stronger = thicker)
- **Arrow**: Shows direction of relationship
- **Hover**: Tooltip shows type, weight, confidence

## Troubleshooting

### Visualizer shows "Disconnected"

**Problem**: Can't reach server

**Solutions**:
1. Check server is running: `curl http://localhost:3000/api/health`
2. Restart server: `cargo run --release --bin eventgraphdb-server`
3. Check firewall settings

### No nodes visible

**Problem**: No data in database

**Solutions**:
1. Run tests: `node test-integration.js`
2. Process some events via API
3. Click "Refresh" button

### Module import error in browser console

**Problem**: ES modules not supported with file:// protocol

**Solutions**:
1. Use a local web server:
   ```bash
   cd client/visualizer
   python -m http.server 8080
   # Then open: http://localhost:8080
   ```

2. Or use Node.js http-server:
   ```bash
   npx http-server client/visualizer -p 8080
   ```

### Graph is too cluttered

**Solutions**:
1. Click "⚡ Physics" to turn off physics
2. Manually drag nodes to organize
3. Zoom in to focus on specific nodes
4. Reduce limit in URL: `http://localhost:3000/api/graph?limit=20`

## Advanced Usage

### Custom Refresh Rate

Edit `client/visualizer/index.html`:

```javascript
// Change from 5000ms (5 seconds) to 10000ms (10 seconds)
setInterval(refreshData, 10000);
```

### Custom Node Limit

The visualizer fetches up to 100 nodes by default. Change it:

```javascript
// In index.html, line ~335
fetch('http://127.0.0.1:3000/api/graph?limit=200')
```

### Custom Graph Styling

Edit the `options` object in `renderGraph()`:

```javascript
const options = {
    nodes: {
        shape: 'box',     // Try: dot, box, diamond, star
        size: 25,         // Node size
        font: {size: 14}, // Font size
    },
    physics: {
        barnesHut: {
            gravitationalConstant: -10000,  // Spread nodes more/less
            springLength: 300               // Edge length
        }
    }
};
```

## API Endpoints Used

The visualizer calls these endpoints:

- `GET /api/health` - Server status, node/edge counts
- `GET /api/stats` - Event processing statistics
- `GET /api/graph?limit=100` - Graph structure (nodes + edges)

## Performance

**Recommended**: 50-200 nodes for smooth interaction
**Maximum**: 1000 nodes (may lag on slower machines)

**Tips**:
- Use the `limit` parameter to control node count
- Disable physics for large graphs (click ⚡ button)
- Close browser dev tools for better performance

## Next Steps

### Enhance the Visualizer

1. **Add filtering** - Filter by node type, date, agent
2. **Add search** - Find specific nodes by ID or label
3. **Add timeline** - Replay events chronologically
4. **Add clustering** - Group related nodes
5. **Add export** - Save graph as PNG or JSON

### Integrate with Your App

The visualizer is standalone HTML - embed it in your app:

```html
<iframe src="client/visualizer/index.html"
        width="100%"
        height="600px"
        style="border: 1px solid #ccc;">
</iframe>
```

### Build a Dashboard

Create a full dashboard with multiple views:
- Graph visualization (current visualizer)
- Event timeline
- Memory browser
- Strategy explorer
- Performance metrics

## Resources

- **Vis.js Documentation**: https://visjs.github.io/vis-network/docs/
- **EventGraphDB API**: `API_REFERENCE.md`
- **Integration Tests**: `test-integration.js`
- **Visualizer README**: `client/visualizer/README.md`

## Support

**Issues?** Check:
1. Server logs: Look at the console where server is running
2. Browser console: Press F12 → Console tab
3. Network tab: Check API requests (F12 → Network)

**Questions?** See:
- `API_REFERENCE.md` - Complete API documentation
- `BUILD_STATUS.md` - System status
- `TEST_GUIDE.md` - Testing guide

---

**Enjoy exploring your knowledge graph! 🎉**
