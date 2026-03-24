# Contributing to MinnsDB

Thank you for your interest in contributing to MinnsDB! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- **Rust 1.83+** — install via [rustup](https://rustup.rs/)
- **Git** — for version control
- **Docker** (optional) — for containerized development

### Building from Source

```bash
git clone https://github.com/Minns-ai/MinnsDB.git
cd MinnsDB
cargo build --release
```

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p agent-db-graph

# End-to-end server tests
cargo test -p minnsdb-server --test end_to_end -- --nocapture
```

### Code Quality

Before submitting a PR, ensure all checks pass:

```bash
cargo fmt --all                                          # Format
cargo clippy --all-targets --all-features -- -D warnings # Lint (zero warnings)
cargo test --workspace                                   # Tests
cargo audit                                              # Security audit
```

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/Minns-ai/MinnsDB/issues) to avoid duplicates
2. Open a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - MinnsDB version, OS, and Rust version
   - Relevant logs or error messages

### Suggesting Features

Open an issue with the `enhancement` label. Include:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

### Submitting Pull Requests

1. **Fork** the repository and create a branch from `main`
2. **Write tests** for any new functionality
3. **Follow the code style** — run `cargo fmt` and `cargo clippy`
4. **Keep PRs focused** — one feature or fix per PR
5. **Write a clear PR description** explaining what and why
6. **Update documentation** if your change affects the public API

### Pull Request Checklist

- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes with zero warnings
- [ ] `cargo test --workspace` passes
- [ ] New code has tests where applicable
- [ ] Documentation updated if public API changed
- [ ] Commit messages are clear and descriptive

## Architecture Overview

MinnsDB is organized as a Cargo workspace with the following key crates:

| Crate | Purpose |
|-------|---------|
| `agent-db-core` | Core types, traits, and error handling |
| `agent-db-events` | Event ingestion and processing |
| `agent-db-graph` | Temporal knowledge graph engine (the main crate) |
| `agent-db-tables` | Bi-temporal row store |
| `agent-db-storage` | ReDB persistence backend |
| `minns-wasm-runtime` | Sandboxed WASM agent runtime |
| `minns-auth` | API key authentication |
| `minns-sdk` | SDK for building WASM agent modules |
| `minnsdb-server` | Axum HTTP API server |

The graph engine (`agent-db-graph`) is the central orchestrator — it ties together storage, events, tables, NLQ, subscriptions, and the ontology system.

## Code Style

- Follow standard Rust conventions and idioms
- Use `thiserror` for library errors, `anyhow` for application errors
- Prefer `tracing` over `println!` for logging
- Write doc comments for public APIs
- Keep functions focused and reasonably sized

## Community

- [Discord](https://discord.gg/6a2cCRPwUR) — ask questions, discuss features, get help
- [GitHub Issues](https://github.com/Minns-ai/MinnsDB/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/Minns-ai/MinnsDB/discussions) — general discussion

## License

By contributing to MinnsDB, you agree that your contributions will be licensed under the [Business Source License 1.1](LICENSE). The SDK crates (`minns-sdk`, `minns-sdk-macros`) are licensed under Apache 2.0.

Copyright (c) 2026 Journey Into Product Ltd
