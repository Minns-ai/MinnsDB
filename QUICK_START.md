# EventGraphDB Quick Start Guide

**How to start the server and visualizer**

---

## 🚀 Starting the Server

### Option 1: Run the .exe Directly (Fastest - Already Built!) ✅

Since you already have a release binary built, you can run it directly:

```bash
# From project root
.\target\release\eventgraphdb-server.exe
```

**Or double-click it in Windows Explorer:**
- Navigate to: `target\release\eventgraphdb-server.exe`
- Double-click to run

### Option 2: Development Mode (Recompiles on changes)

```bash
# From the project root directory
cargo run --bin eventgraphdb-server
```

This will compile and run (slower first time, but auto-recompiles on code changes).

### Option 3: Build Release Binary (If you need to rebuild)

```bash
# Build in release mode (optimized, faster)
cargo build --release --bin eventgraphdb-server

# Then run it
.\target\release\eventgraphdb-server.exe
```

### What You'll See

```
🚀 Starting EventGraphDB REST API Server
Initializing GraphEngine with automatic self-evolution...
✓ GraphEngine initialized
🌐 Server listening on http://127.0.0.1:3000
📚 API documentation: http://127.0.0.1:3000/docs
❤️  Health check: http://127.0.0.1:3000/api/health
```

**Server is now running on:** `http://127.0.0.1:3000`

---

## 🎨 Starting the Visualizer

### Step 1: Install Dependencies (First time only)

```bash
cd client/visualizer
npm install
```

### Step 2: Build the Visualizer

```bash
# Build TypeScript to JavaScript
npm run build
```

### Step 3: Start the Visualizer

**Option A: Using npm serve (Recommended)**

```bash
npm run serve
```

This will:
- Start a local web server on port 8080
- Automatically open your browser
- Serve the visualizer files

**Option B: Using Python (if you have Python installed)**

```bash
# Python 3
python -m http.server 8080

# Python 2
python -m SimpleHTTPServer 8080
```

**Option C: Using npx serve**

```bash
npx serve . -p 8080 -o
```

**Option D: Windows Batch File (Easiest)**

```bash
# From project root
start-visualizer.bat
```

This will open the visualizer in your default browser.

### Step 4: Open in Browser

Navigate to: **http://localhost:8080**

---

## 📋 Complete Startup Sequence

### Terminal 1: Start the Server

```bash
# Navigate to project root
cd C:\Users\Lenovo\OneDrive\Documents\EventGraphDB

# Start server
cargo run --bin eventgraphdb-server
```

**Keep this terminal open!** The server needs to keep running.

### Terminal 2: Start the Visualizer

```bash
# Navigate to visualizer directory
cd client/visualizer

# Install dependencies (first time only)
npm install

# Build
npm run build

# Start server
npm run serve
```

**Or use the batch file:**

```bash
# From project root
start-visualizer.bat
```

---

## ✅ Verify Everything is Working

### 1. Check Server is Running

Open in browser: **http://127.0.0.1:3000/api/health**

You should see:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "is_healthy": true,
  "node_count": 0,
  "edge_count": 0
}
```

### 2. Check Visualizer is Running

Open in browser: **http://localhost:8080**

You should see:
- EventGraphDB Visualizer interface
- Connection status (should show "Connected")
- Graph view (may be empty if no data yet)

---

## 🔧 Troubleshooting

### Server Won't Start

**Error: "Could not find binary"**
```bash
# Check if server binary exists
ls target/debug/eventgraphdb-server*
ls target/release/eventgraphdb-server*

# If not, build it first
cargo build --bin eventgraphdb-server
```

**Error: "Address already in use"**
```bash
# Port 3000 is already in use
# Either:
# 1. Stop the other service using port 3000
# 2. Or change the port in server/src/main.rs (line 446)
```

**Error: "Failed to initialize GraphEngine"**
```bash
# Check if all dependencies are installed
cargo check -p agent-db-graph
```

### Visualizer Won't Start

**Error: "npm: command not found"**
```bash
# Install Node.js from https://nodejs.org/
# Then try again
```

**Error: "Cannot find module"**
```bash
# Install dependencies
cd client/visualizer
npm install
```

**Error: "TypeScript compilation failed"**
```bash
# Check TypeScript version
npm list typescript

# Reinstall if needed
npm install typescript@latest
```

**Visualizer shows "Disconnected"**
```bash
# Make sure server is running on port 3000
# Check server terminal for errors
# Try refreshing the visualizer page
```

---

## 🎯 Quick Commands Reference

### Server Commands

```bash
# Build server
cargo build --bin eventgraphdb-server

# Run server (debug)
cargo run --bin eventgraphdb-server

# Run server (release)
cargo run --release --bin eventgraphdb-server

# Check server compiles
cargo check --bin eventgraphdb-server
```

### Visualizer Commands

```bash
# Install dependencies
cd client/visualizer && npm install

# Build
npm run build

# Start dev server
npm run serve

# Watch mode (auto-rebuild on changes)
npm run dev
```

---

## 📝 Next Steps

Once both are running:

1. **Send some events** to the server:
   ```bash
   curl -X POST http://127.0.0.1:3000/api/events \
     -H "Content-Type: application/json" \
     -d '{"event": {...}}'
   ```

2. **View in visualizer** - The graph should appear automatically

3. **Explore the API** - Visit http://127.0.0.1:3000/docs

---

## 🆘 Need Help?

- Check server logs in the terminal
- Check browser console (F12) for visualizer errors
- Verify both are running on correct ports:
  - Server: `127.0.0.1:3000`
  - Visualizer: `localhost:8080`

---

**Happy exploring! 🎉**
