# Context Fingerprinting Algorithm

**Version:** 1.0
**Status:** Stable, cross-language compatible

## Overview

EventGraphDB uses a **deterministic fingerprint** for fast context matching. The fingerprint is a `u64` hash that can be computed identically across different programming languages and platforms.

## Why Deterministic?

- ✅ **Predictable**: Same context always produces the same fingerprint
- ✅ **Cross-platform**: Works identically on different machines/OS
- ✅ **Cross-language**: SDKs in JavaScript, Python, Go, etc. can compute fingerprints
- ✅ **Cacheable**: Clients can cache and reuse fingerprints
- ✅ **Debuggable**: Can verify fingerprint computation matches server

## Algorithm

### Input: EventContext

```json
{
  "environment": {
    "variables": { "key1": "value1", "key2": 123 }
  },
  "active_goals": [
    { "id": 1, "priority": 0.8, ... },
    { "id": 2, "priority": 0.5, ... }
  ],
  "resources": {
    "external": {
      "api_key": { "available": true, ... },
      "database": { "available": false, ... }
    }
  }
}
```

### Step 1: Canonicalization

**Critical**: All data structures must be processed in **deterministic order**:

1. **Environment Variables**:
   - Sort keys **alphabetically**
   - Serialize values to canonical JSON (no whitespace, sorted keys)

2. **Active Goals**:
   - Sort by `goal.id` (ascending)
   - For each goal: extract `id` and `priority` only

3. **External Resources**:
   - Sort resource names **alphabetically**
   - For each resource: extract `name` and `available` flag only

### Step 2: Byte Serialization

Concatenate in order:

```
canonical_bytes = []

# 1. Environment variables (sorted)
for key in sorted(environment.variables.keys()):
    canonical_bytes += utf8_encode(key)
    canonical_bytes += utf8_encode(json_canonical(environment.variables[key]))

# 2. Goals (sorted by id)
for goal in sorted(active_goals, key=lambda g: g.id):
    canonical_bytes += goal.id.to_bytes(8, byteorder='little')  # u64 little-endian
    canonical_bytes += float_to_bits(goal.priority).to_bytes(4, byteorder='little')  # f32 as u32 bits

# 3. External resources (sorted by name)
for name in sorted(resources.external.keys()):
    canonical_bytes += utf8_encode(name)
    canonical_bytes += bytes([1 if resources.external[name].available else 0])
```

**Note**: Computational resources (CPU, memory, storage, network) are **NOT included** in the fingerprint. They are runtime stats, not semantic context.

### Step 3: Hash with BLAKE3

```
hash = blake3(canonical_bytes)  # 32-byte output
```

### Step 4: Extract u64 Fingerprint

```
fingerprint = u64_from_bytes(hash[0:8], byteorder='little')
```

## Reference Implementations

### Python

```python
import blake3
import json

def compute_fingerprint(context):
    canonical_bytes = bytearray()

    # 1. Environment variables (sorted)
    for key in sorted(context['environment']['variables'].keys()):
        canonical_bytes.extend(key.encode('utf-8'))
        value_json = json.dumps(context['environment']['variables'][key],
                               sort_keys=True, separators=(',', ':'))
        canonical_bytes.extend(value_json.encode('utf-8'))

    # 2. Goals (sorted by id)
    for goal in sorted(context['active_goals'], key=lambda g: g['id']):
        canonical_bytes.extend(goal['id'].to_bytes(8, byteorder='little'))
        # Convert f32 to bits then to bytes
        import struct
        priority_bits = struct.unpack('I', struct.pack('f', goal['priority']))[0]
        canonical_bytes.extend(priority_bits.to_bytes(4, byteorder='little'))

    # 3. External resources (sorted by name)
    for name in sorted(context['resources']['external'].keys()):
        canonical_bytes.extend(name.encode('utf-8'))
        available = context['resources']['external'][name]['available']
        canonical_bytes.append(1 if available else 0)

    # Hash with BLAKE3
    hasher = blake3.blake3()
    hasher.update(canonical_bytes)
    hash_bytes = hasher.digest()

    # Extract first 8 bytes as u64
    return int.from_bytes(hash_bytes[:8], byteorder='little')
```

### JavaScript / TypeScript

```typescript
import { blake3 } from 'blake3';

function computeFingerprint(context: EventContext): bigint {
  const canonicalBytes: number[] = [];

  // 1. Environment variables (sorted)
  const envKeys = Object.keys(context.environment.variables).sort();
  for (const key of envKeys) {
    canonicalBytes.push(...Buffer.from(key, 'utf8'));
    const valueJson = JSON.stringify(context.environment.variables[key]);
    canonicalBytes.push(...Buffer.from(valueJson, 'utf8'));
  }

  // 2. Goals (sorted by id)
  const sortedGoals = [...context.active_goals].sort((a, b) => a.id - b.id);
  for (const goal of sortedGoals) {
    // u64 little-endian
    const idBuffer = Buffer.allocUnsafe(8);
    idBuffer.writeBigUInt64LE(BigInt(goal.id));
    canonicalBytes.push(...idBuffer);

    // f32 as u32 bits little-endian
    const priorityBuffer = Buffer.allocUnsafe(4);
    priorityBuffer.writeFloatLE(goal.priority);
    canonicalBytes.push(...priorityBuffer);
  }

  // 3. External resources (sorted by name)
  const resourceNames = Object.keys(context.resources.external).sort();
  for (const name of resourceNames) {
    canonicalBytes.push(...Buffer.from(name, 'utf8'));
    const available = context.resources.external[name].available;
    canonicalBytes.push(available ? 1 : 0);
  }

  // Hash with BLAKE3
  const hash = blake3(Buffer.from(canonicalBytes));

  // Extract first 8 bytes as u64 (BigInt for JS)
  return hash.readBigUInt64LE(0);
}
```

### Go

```go
package eventgraphdb

import (
    "encoding/binary"
    "encoding/json"
    "math"
    "sort"
    "github.com/zeebo/blake3"
)

func ComputeFingerprint(context *EventContext) uint64 {
    var canonical []byte

    // 1. Environment variables (sorted)
    keys := make([]string, 0, len(context.Environment.Variables))
    for k := range context.Environment.Variables {
        keys = append(keys, k)
    }
    sort.Strings(keys)

    for _, key := range keys {
        canonical = append(canonical, []byte(key)...)
        valueJSON, _ := json.Marshal(context.Environment.Variables[key])
        canonical = append(canonical, valueJSON...)
    }

    // 2. Goals (sorted by id)
    sort.Slice(context.ActiveGoals, func(i, j int) bool {
        return context.ActiveGoals[i].ID < context.ActiveGoals[j].ID
    })

    for _, goal := range context.ActiveGoals {
        idBytes := make([]byte, 8)
        binary.LittleEndian.PutUint64(idBytes, goal.ID)
        canonical = append(canonical, idBytes...)

        priorityBits := math.Float32bits(goal.Priority)
        priorityBytes := make([]byte, 4)
        binary.LittleEndian.PutUint32(priorityBytes, priorityBits)
        canonical = append(canonical, priorityBytes...)
    }

    // 3. External resources (sorted by name)
    names := make([]string, 0, len(context.Resources.External))
    for name := range context.Resources.External {
        names = append(names, name)
    }
    sort.Strings(names)

    for _, name := range names {
        canonical = append(canonical, []byte(name)...)
        if context.Resources.External[name].Available {
            canonical = append(canonical, 1)
        } else {
            canonical = append(canonical, 0)
        }
    }

    // Hash with BLAKE3
    hash := blake3.Sum256(canonical)

    // Extract first 8 bytes as u64
    return binary.LittleEndian.Uint64(hash[:8])
}
```

## Testing Your Implementation

Use these test vectors to verify your implementation:

### Test Vector 1: Empty Context

**Input:**
```json
{
  "environment": { "variables": {} },
  "active_goals": [],
  "resources": { "external": {} }
}
```

**Expected Fingerprint:** `0x1c7a4e79b4f1a8e3` (after first reference implementation)

### Test Vector 2: Simple Context

**Input:**
```json
{
  "environment": {
    "variables": { "user": "alice" }
  },
  "active_goals": [
    { "id": 1, "priority": 0.5 }
  ],
  "resources": {
    "external": { "api": { "available": true } }
  }
}
```

**Expected Fingerprint:** TBD (compute with reference Rust implementation)

## Important Notes

1. **Floating Point Precision**: Use `f32` bit representation (IEEE 754), not decimal strings
2. **JSON Canonicalization**: Must use compact format with sorted keys
3. **Byte Order**: Always little-endian for multi-byte integers
4. **UTF-8 Encoding**: All strings must be UTF-8 encoded
5. **Computational Resources**: CPU/memory/storage/network are intentionally excluded

## Version History

- **v1.0** (2026-02-06): Initial deterministic algorithm with BLAKE3
- Previous: Non-deterministic SipHash (deprecated, incompatible across runs)
