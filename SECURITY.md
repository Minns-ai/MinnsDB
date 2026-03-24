# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MinnsDB, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email us at **security@minns.ai** with:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (optional)

We will acknowledge your report within **48 hours** and provide a detailed response within **5 business days**, including our assessment and planned timeline for a fix.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Measures

MinnsDB includes several built-in security features:

### Authentication
- API key authentication with blake3 hashing (keys are never stored in plaintext)
- Group-scoped permissions for multi-tenant isolation
- Root key generated at first boot and shown once

### WASM Sandboxing
- Instruction-metered execution with configurable budgets
- 64MB memory cap per module enforced by Wasmtime
- 30-second wall-time limit via epoch interruption
- Permission-controlled access to tables, graph, and HTTP
- SSRF hardening on `http_fetch` host function

### Data Integrity
- Blake3 checksums on all table pages
- Bi-temporal versioning (data is superseded, never deleted)
- ReDB transactional storage with crash recovery

### Dependencies
- `cargo audit` runs in CI on every push
- Dependency updates are reviewed for security implications
- The `time` crate is pinned to `>=0.3.47` to address known vulnerabilities

## Scope

The following are in scope for security reports:
- Authentication bypass or privilege escalation
- WASM sandbox escape
- SQL/MinnsQL injection
- Server-side request forgery (SSRF)
- Denial of service via resource exhaustion
- Data integrity violations
- Information disclosure

## Acknowledgments

We appreciate the security research community's efforts in helping keep MinnsDB safe. Reporters of valid vulnerabilities will be credited in the release notes (unless they prefer to remain anonymous).
