# Agentic Database Development Makefile

.PHONY: help build test bench clean fmt clippy check dev doc install-tools

# Default target
help:
	@echo "Agentic Database Development Commands:"
	@echo "  build       - Build all crates"
	@echo "  test        - Run all tests"
	@echo "  bench       - Run benchmarks"
	@echo "  check       - Run all checks (fmt, clippy, test)"
	@echo "  clean       - Clean build artifacts"
	@echo "  fmt         - Format code"
	@echo "  clippy      - Run clippy linter"
	@echo "  dev         - Start development environment"
	@echo "  doc         - Generate documentation"
	@echo "  install-tools - Install development tools"

# Build commands
build:
	cargo build --all

build-release:
	cargo build --release --all

# Test commands
test:
	cargo test --all

test-release:
	cargo test --release --all

test-doc:
	cargo test --doc --all

# Benchmark commands
bench:
	cargo bench --all

bench-baseline:
	cargo bench --all -- --save-baseline main

bench-compare:
	cargo bench --all -- --baseline main

# Quality checks
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

clippy-fix:
	cargo clippy --all-targets --all-features --fix

check: fmt-check clippy test-release

# Development commands
clean:
	cargo clean

doc:
	cargo doc --all --no-deps --open

doc-private:
	cargo doc --all --no-deps --document-private-items --open

dev: install-tools
	@echo "Starting development environment..."
	cargo watch -x "check --all" -x "test --all"

# Installation and setup
install-tools:
	@echo "Installing development tools..."
	cargo install cargo-watch
	cargo install cargo-audit
	cargo install criterion-table
	rustup component add rustfmt clippy

# Security
audit:
	cargo audit

# Database specific commands
init-db:
	@echo "Initializing test database..."
	mkdir -p test_data
	cargo run --bin agent-db-cli -- init --data-dir test_data

reset-db:
	@echo "Resetting test database..."
	rm -rf test_data
	mkdir -p test_data

# Performance testing
perf-test:
	@echo "Running performance tests..."
	cargo build --release
	cargo run --release --bin agent-db-perf-test

memory-test:
	@echo "Running memory tests..."
	valgrind --tool=memcheck --leak-check=full cargo test --release

# CI commands (used by GitHub Actions)
ci-check: fmt-check clippy test-release doc audit

ci-bench: bench

# Release commands
pre-release: clean check bench doc
	@echo "Pre-release checks passed!"

release-patch:
	cargo release patch --execute

release-minor:
	cargo release minor --execute

release-major:
	cargo release major --execute

# Examples
run-examples:
	cargo run --example basic_usage
	cargo run --example memory_demo
	cargo run --example graph_demo

# Profiling
profile:
	@echo "Profiling with perf..."
	cargo build --release
	perf record --call-graph=dwarf cargo run --release --bin agent-db-perf-test
	perf report

flamegraph:
	@echo "Generating flamegraph..."
	cargo flamegraph --bin agent-db-perf-test

# Database operations
backup-test-data:
	@echo "Backing up test data..."
	tar -czf test_data_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz test_data/

restore-test-data:
	@echo "Restoring test data..."
	@read -p "Enter backup filename: " backup; \
	tar -xzf $$backup

# Monitoring
monitor:
	@echo "Starting system monitoring..."
	top -p $(shell pgrep -f agent-db)

logs:
	@echo "Tailing logs..."
	tail -f agent_db.log

# Development environment
setup-dev: install-tools
	@echo "Setting up development environment..."
	pre-commit install
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-commit

# Container commands
docker-build:
	docker build --build-arg SERVICE_PROFILE=normal -t eventgraphdb:latest .

docker-build-free:
	docker build --build-arg SERVICE_PROFILE=free -t eventgraphdb:free .

docker-run:
	docker run -it --rm -v $(PWD)/test_data:/data eventgraphdb:latest

docker-run-free:
	docker run -it --rm -v $(PWD)/test_data:/data eventgraphdb:free

docker-bench:
	docker run -it --rm agent-db:latest cargo bench

# Integration testing
integration-test:
	@echo "Running integration tests..."
	docker-compose up -d
	cargo test --test integration_tests
	docker-compose down

# Load testing
load-test:
	@echo "Running load tests..."
	cargo run --release --bin load-tester -- --duration 60 --rate 10000

stress-test:
	@echo "Running stress tests..."
	cargo run --release --bin stress-tester -- --agents 1000 --events 1000000