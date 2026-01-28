#!/bin/bash

# Quick start script to create a new rust-bert NER service
# Usage: ./create-ner-service.sh [project-name]

PROJECT_NAME="${1:-ner-service}"

echo "Creating rust-bert NER service: $PROJECT_NAME"
echo "================================================"

# Create project
cargo new "$PROJECT_NAME" --bin
cd "$PROJECT_NAME"

# Create directory structure
mkdir -p src

echo "Creating source files..."

# Create Cargo.toml
cat > Cargo.toml << 'EOF'
[package]
name = "ner-service"
version = "0.1.0"
edition = "2021"

[dependencies]
# Web framework
axum = "0.7"
tokio = { version = "1.35", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["trace", "cors"] }

# NER - rust-bert
rust-bert = "0.21"
tch = "0.15"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
once_cell = "1.19"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
/target
Cargo.lock
.env
*.swp
*.swo
*~
.DS_Store
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM rust:1.75 as builder

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    wget unzip && rm -rf /var/lib/apt/lists/*

ENV LIBTORCH_VERSION=2.1.0
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d /opt && \
    rm libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release && rm -rf src

COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

COPY --from=builder /usr/src/app/target/release/ner-service /usr/local/bin/ner-service

RUN mkdir -p /root/.cache/huggingface

EXPOSE 8081

CMD ["ner-service"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  ner-service:
    build: .
    ports:
      - "8081:8081"
    environment:
      - RUST_LOG=info
      - RUST_BACKTRACE=1
    volumes:
      - model-cache:/root/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  model-cache:
EOF

# Create .dockerignore
cat > .dockerignore << 'EOF'
target/
.git/
.gitignore
*.md
Dockerfile
.dockerignore
EOF

# Create README
cat > README.md << 'EOF'
# NER Service (Rust + rust-bert)

High-performance Named Entity Recognition service.

## Quick Start

```bash
# Run locally
cargo run --release

# Or with Docker
docker-compose up
```

## API

POST http://localhost:8081/ner
```json
{
  "text": "Apple Inc. was founded by Steve Jobs."
}
```

## Documentation

See RUST_NER_SERVICE_GUIDE.md for complete implementation details.
EOF

echo "✓ Project structure created"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. Copy the source files from RUST_NER_SERVICE_GUIDE.md to src/"
echo "3. cargo build --release"
echo "4. cargo run --release"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
