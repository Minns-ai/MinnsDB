# Agentic Database

**An event-driven contextual graph database with memory-inspired learning**

[![CI](https://github.com/your-org/agent-database/workflows/CI/badge.svg)](https://github.com/your-org/agent-database/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The Agentic Database is a specialized database system designed to capture, learn from, and enhance agent behavior through contextual memory formation. Unlike traditional databases that store static data, this system creates dynamic memory structures that evolve based on agent experiences and outcomes.

### Key Features

🎯 **Event-First Architecture**: All knowledge derives from immutable event streams  
🧠 **Memory-Inspired**: Formation, consolidation, and decay processes mimic biological memory  
🕸️ **Contextual Graph**: Relationships emerge from actual agent experiences  
🔍 **Pattern Learning**: Real-time discovery and adaptive optimization  
⚡ **High Performance**: 100K+ events/sec, <50ms memory retrieval  
🦀 **Rust Implementation**: Memory-safe, concurrent, and high-performance  

## Quick Start

### Prerequisites

- Rust 1.70+
- 8+ CPU cores (recommended)
- 32+ GB RAM (recommended)
- SSD storage

### Installation

```bash
git clone https://github.com/your-org/agent-database.git
cd agent-database
cargo build --release
```

### Basic Usage

```rust
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use agent_db_events::{Event, EventType, ActionOutcome};

// Create integrated engine (all self-evolution features automatic)
let engine = GraphEngine::new().await?;

// Process events - automatic learning happens!
let event = Event {
    id: generate_event_id(),
    timestamp: current_timestamp(),
    agent_id: 1,
    agent_type: "ai-debugger".to_string(),
    session_id: 42,
    event_type: EventType::Action {
        action_name: "fix_null_error".to_string(),
        parameters: json!({"fix": "add_null_check"}),
        outcome: ActionOutcome::Success {
            result: json!({"tests_pass": true})
        },
        duration_ns: 1_500_000_000,
    },
    causality_chain: vec![],
    context: your_context,
    metadata: HashMap::new(),
};

// Process event - automatic episode detection, memory formation,
// strategy extraction, and reinforcement learning!
engine.process_event(event).await?;

// Query learned knowledge
let memories = engine.get_agent_memories(agent_id, 10).await;
let strategies = engine.get_agent_strategies(agent_id, 10).await;
let suggestions = engine.get_next_action_suggestions(context_hash, None, 5).await?;

// "What should I do next?"
for suggestion in suggestions {
    println!("{}: {:.1}% success",
        suggestion.action_name,
        suggestion.success_probability * 100.0
    );
}
```

**See [API_REFERENCE.md](API_REFERENCE.md) for complete documentation and examples.**

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐
│   Agent SDKs    │    │  Admin Tools    │
└─────────────────┘    └─────────────────┘
         │                       │
┌────────┼───────────────────────┼────────┐
│        │    Core Database      │        │
│        │                       │        │
│  ┌─────▼──┐  ┌──────────┐  ┌───▼───┐    │
│  │Event   │  │Graph     │  │Memory │    │
│  │Timeline│  │Inference │  │System │    │
│  └────────┘  └──────────┘  └───────┘    │
│                                         │
│           ┌──────────┐                  │
│           │ Storage  │                  │
│           │ Engine   │                  │
│           └──────────┘                  │
└─────────────────────────────────────────┘
```

### Data Flow

```
Agent Actions → Event Storage → Graph Inference → Memory Formation → Pattern Learning
     ↑                                                                      ↓
     └── Enhanced Behavior ← Memory Retrieval ← Context Matching ← Patterns ─┘
```

## Performance

### Target Performance (MVP)

- **Event Ingestion**: 50,000 events/second
- **Memory Retrieval**: <50ms latency
- **Graph Traversal**: <10ms for depth 6
- **Scale**: 100K nodes, 1M edges
- **Storage**: 5:1 compression ratio

### Benchmarks

Run benchmarks with:

```bash
cargo bench
```

## Development

### Project Structure

```
agent-database/
├── crates/
│   ├── agent-db-core/      # Core traits and types
│   ├── agent-db-events/    # Event system
│   ├── agent-db-storage/   # Storage engine
│   └── agent-db-graph/     # Graph operations
├── benches/                # Benchmarks
├── tests/                  # Integration tests
└── examples/               # Usage examples
```

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with optimizations
cargo test --release
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write comprehensive tests
- Document public APIs

## Documentation

- **[API Reference](API_REFERENCE.md)** ⭐ **NEW!** - Complete API documentation with examples
- [Technical Writeup](TECHNICAL_WRITEUP.md) - Deep technical details and algorithms
- [Progress Report](PROGRESS_REPORT.md) - Development progress and status
- [Technical Specification](TECHNICAL_SPECIFICATION.md) - System architecture
- [MVP Specification](MVP_SPECIFICATION.md) - Requirements and targets
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

### Phase 1: Foundation (Weeks 1-8) ✅
- Core event system
- Basic storage layer
- Simple graph operations

### Phase 2: Storage Engine (Weeks 9-16)
- Advanced storage optimization
- Temporal indexing
- Memory management

### Phase 3: Graph System (Weeks 17-24)
- Graph inference algorithms
- Advanced traversal operations
- Graph storage optimization

### Phase 4: Memory System (Weeks 25-32)
- Memory formation framework
- Context-based retrieval
- Memory consolidation

### Phase 5: Pattern Learning (Weeks 33-40)
- Pattern discovery algorithms
- Online learning capabilities
- Pattern validation framework

### Phase 6: APIs (Weeks 41-48)
- Production APIs
- SDK development
- Documentation

### Phase 7: Performance (Weeks 49-56)
- Performance optimization
- Scalability testing
- Distributed support

### Phase 8: Production (Weeks 57-64)
- Security hardening
- Operational tools
- Enterprise features

## Support

- [GitHub Issues](https://github.com/your-org/agent-database/issues)
- [Discussions](https://github.com/your-org/agent-database/discussions)
- [Documentation](https://your-org.github.io/agent-database)

---

**Built with ❤️ in Rust**