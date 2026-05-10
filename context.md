# context.md — Project Memory

> **Purpose:** Single source of truth for AI sessions. Read this file FIRST before exploring the codebase.
> **Update:** At the end of every session, update this file with what changed.

---

## Project Identity

- **Name:** Magnitude VectorDB
- **Repo:** https://github.com/POTATO-VE1/Magnitude
- **Module:** `github.com/POTATO-VE1/Magnitude` (go.mod)
- **Language:** Go 1.24+ (server), Python 3.9+ (client)
- **License:** MIT

## What It Is

A self-hosted vector database with:
- Pluggable indexing (Flat, IVF, HNSW, SPANN)
- Hybrid dense+sparse search (BM25 + RRF fusion)
- Multi-tenancy (Tenant → Database → Collection)
- CLIP-powered Python client for semantic image search
- Distributed features (gossip, failure detection, migration)

## Directory Structure

```
├── cmd/server/main.go          # Entrypoint — config, wiring, signal handling
├── config.yaml                 # All tunables with defaults
├── Dockerfile                  # Multi-stage: golang:1.24-alpine → alpine:3.20
├── docker-compose.yml          # Single service, volumes for data + certs
├── internal/
│   ├── api/                    # HTTP handlers + middleware
│   │   ├── handlers.go         # v1 API (flat routes)
│   │   ├── handlers_v2.go      # v2 API (tenant-scoped routes)
│   │   ├── middleware.go        # Recovery, RequestID, Metrics, Auth, RateLimit
│   │   ├── context.go          # Context key helpers
│   │   └── types.go            # Request/response types
│   ├── cache/                  # Segment cache (LRU with second-access admission)
│   ├── cluster/                # Distributed primitives
│   │   ├── coordinator.go      # Node membership, health, routing
│   │   ├── consistency.go      # ConsistencyLevel (One/Quorum/All)
│   │   ├── hash.go             # Consistent hash ring (SHA-256, 150 vnodes)
│   │   ├── rendezvous.go       # Rendezvous hashing
│   │   └── replica_client.go   # Inter-node HTTP RPC
│   ├── collection/             # Collection lifecycle
│   │   ├── collection.go       # Manager, InsertVectors, SearchVectors, HybridSearch
│   │   └── fork.go             # Collection snapshotting
│   ├── config/                 # YAML config loading + validation
│   ├── distance/               # L2, Cosine, DotProduct, Manhattan + SIMD
│   │   ├── distance.go         # Pure-Go implementations
│   │   ├── simd.c              # AVX2 C kernels
│   │   ├── simd_amd64.go       # CGO bridge
│   │   └── dispatch.go         # Runtime SIMD/pure selection
│   ├── errors/                 # VDBError with codes + HTTP status mapping
│   ├── events/                 # Flow event bus (pub-sub for test coordination)
│   ├── failure/                # Failure detector (Alive→Suspect→Dead)
│   ├── gc/                     # 3-phase GC (Mark→Fence→Sweep)
│   ├── gossip/                 # UDP gossip protocol + dedup
│   ├── index/                  # Pluggable index implementations
│   │   ├── index.go            # Index interface (Insert/Search/Delete/Len/Rebuild/Flush)
│   │   ├── flat/               # Brute-force O(N)
│   │   ├── hnsw/               # HNSW graph + snapshot persistence
│   │   ├── ivf/                # Inverted File (K-Means + dirty buffer)
│   │   ├── spann/              # SPANN (centroid HNSW + mmap postings)
│   │   └── sparse/             # BM25 inverted index
│   ├── metadata/               # SQLite-backed SysDB
│   │   ├── sqlite.go           # Collection/vector metadata, tombstones
│   │   ├── tenant.go           # Multi-tenancy schema
│   │   └── filter.go           # Metadata filter parsing ($eq, $gt, $in, etc.)
│   ├── migration/              # Data migration planner + worker
│   ├── observability/          # Prometheus metrics registration
│   ├── quantize/               # SQ8 + PQ (implementation exists, NOT wired up)
│   ├── scheduler/              # Priority-aware task scheduler
│   ├── search/                 # RRF fusion
│   ├── security/               # API key auth, rate limiter, audit logger
│   └── storage/                # WAL, compaction, bloom, mmap
│       ├── wal.go              # SQLite-backed WAL with binary encoding
│       ├── compaction.go       # Background segment materialization
│       ├── bloom.go            # Segment bloom filters
│       ├── compact_action.go   # Crash-safe compaction action files
│       ├── format.go           # Binary segment format (magic, header, checksum)
│       └── mmap_unix.go        # Memory-mapped file I/O
├── pkg/client/                 # Go HTTP client
│   ├── client.go               # Single-node client
│   └── cluster_client.go       # Cluster-aware client with hash ring
└── python-client/              # Python CLIP client (PyPI: magnitude-client)
    ├── pyproject.toml           # Hatchling build, optional [embed], [cli], [all]
    └── src/magnitude/
        ├── client.py            # VectorDBClient class
        ├── embedder.py          # CLIPEmbedder (sentence-transformers)
        ├── exceptions.py        # MagnitudeConnectionError, etc.
        └── cli/
            ├── ingest.py        # `magnitude-ingest` entry point
            └── search.py        # `magnitude-search` entry point
```

## Key Interfaces

### Index Interface (`internal/index/index.go`)
```go
type Index interface {
    Insert(id uint64, vector []float32) error
    Search(ctx context.Context, query []float32, k int, nprobe int) ([]SearchResult, error)
    Delete(id uint64) error
    Len() int
    Rebuild() error
    Flush() error
}
```

### WAL Interface (`internal/storage/wal.go`)
```go
type WAL interface {
    Append(op WALOp) (uint64, error)
    AppendBatch(ops []WALOp) ([]uint64, error)
    ReadFrom(afterSeq uint64) ([]WALEntry, error)
    Truncate(upToSeq uint64) error
    Close() error
}
```

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Flat index | ✅ Wired | Brute-force, correctness baseline |
| IVF index | ✅ Wired | K-Means + dirty buffer + background rebuild |
| HNSW index | ✅ Wired | Graph + snapshot persistence |
| SPANN index | ✅ Wired | Centroid HNSW + mmap postings |
| Sparse/BM25 | ✅ Wired | Inverted index for text search |
| Hybrid search | ✅ Wired | Dense + sparse + RRF fusion |
| Multi-tenancy | ✅ Wired | Tenant → Database → Collection |
| SIMD distance | ⚠️ Partial | AVX2 code exists, NOT called in search hot paths |
| Quantization | ❌ Stub | SQ8/PQ impl exists, NOT integrated |
| Gossip | ✅ Impl | UDP protocol, dedup, event types |
| Failure detection | ✅ Impl | Alive→Suspect→Dead state machine |
| Migration | ✅ Impl | Planner + worker with retries |
| HNSW snapshots | ✅ Wired | Binary format, Flush() writes, startup loads |
| Bloom filters | ✅ Wired | Built during compaction |
| Crash-safe compaction | ✅ Wired | Action files for multi-file atomicity |
| Task scheduler | ✅ Wired | Foreground/background priority pools |
| Flow events | ✅ Impl | Pub-sub for test coordination |

## Config Structure (config.yaml)

```yaml
server:       # addr, internalPort, certFile, keyFile, timeouts
auth:         # keyHashes [] (SHA-256, empty = no auth)
rateLimit:    # searchRPS, insertRPS, burst sizes
storage:      # dataDir, mmapCapacity
index:        # type (flat/ivf/hnsw/spann), M, efConstruction, efSearch, dirtyThreshold
gc:           # percent (runtime.SetGCPercent)
bloomFilter:  # enabled, falsePositiveRate, minVectors
walSync:      # syncMode (per-write/delayed/none), syncDelay
cluster:      # enabled, nodeID, replicationFactor, virtualNodes, consistency defaults
gossip:       # port, fanout, maxSeen, seenExpiry, probeInterval
failureDetection: # interval, timeout, suspectAfter, deadAfter
migration:    # batchSize, parallelism, maxRetries
```

## API Endpoints

### v1 (flat routes)
- `POST /v1/collections` — create collection
- `GET /v1/collections` — list collections
- `GET /v1/collections/{id}` — get collection
- `DELETE /v1/collections/{id}` — delete collection
- `POST /v1/collections/{id}/vectors` — insert vectors
- `POST /v1/collections/{id}/search` — search vectors
- `DELETE /v1/collections/{id}/vectors/{vid}` — delete vector

### v2 (tenant-scoped)
- `POST /api/v2/tenants/{t}/databases/{d}/collections` — create
- `GET /api/v2/tenants/{t}/databases/{d}/collections` — list
- `POST /api/v2/tenants/{t}/databases/{d}/collections/{id}/add` — insert
- `POST /api/v2/tenants/{t}/databases/{d}/collections/{id}/query` — search
- `POST /api/v2/tenants/{t}/databases/{d}/collections/{id}/hybrid` — hybrid search

### Internal (admin port :9090)
- `/metrics` — Prometheus
- `/debug/pprof/` — profiling

## Recent Session Log

### 2026-05-10 — Full audit + bug fixes + Python packaging

**Bugs fixed:**
- HNSW snapshot: nil rng caused panic on insert after load → initialized with time seed
- HNSW Rebuild: lock-unlock-lock race → extracted `insertLocked()`
- WAL ReadFrom: held write lock during read → changed to `sync.RWMutex`
- Compactor.Stop(): nil `done` channel if Start never called → init in constructor
- Coordinator.Heartbeat: dead nodes not re-added to ring → re-add on recovery
- Dockerfile: golang:1.25-alpine doesn't exist → changed to 1.24
- docker-compose: metrics port exposed to network → bound to 127.0.0.1
- .gitignore/.dockerignore: .env files not excluded → added patterns
- Python client: ConnectionError shadowed builtin → renamed to MagnitudeConnectionError
- Python client: auth header dropped on POST → removed headers= override
- Python client: timeout broken (session.timeout not a real attr) → pass per-request
- Python client: SSL verify=False default → changed to True

**Files created:**
- `internal/index/hnsw/snapshot.go` — HNSW binary snapshot (write + load)
- `python-client/src/magnitude/` — proper PyPI package structure
- `python-client/pyproject.toml` — hatchling build config
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**README:** Stripped from 184 lines → ~80 lines. Clean getting-started guide.

---

## Coding Conventions

- Go: `internal/` packages only, no `pkg/` except `pkg/client/`
- Tests: `*_test.go` files exist locally but NOT committed to repo
- Errors: `internal/errors/VDBError` with codes + HTTP status mapping
- Logging: `log/slog` with JSON handler
- Config: YAML with `DefaultConfig()` overlay + `Validate()`
- Commits: conventional commits (feat:, fix:, docs:, chore:)

## Known Limitations

1. **SIMD not in hot path** — AVX2 code exists but only called from dead quantization pipeline
2. **Quantization not integrated** — SQ8/PQ implementation exists but no config/API to enable
3. **No v1 hybrid endpoint** — hybrid search only on v2 tenant-scoped routes
4. **HNSW Flush reads all WAL** — O(n) to find max seqID (should be `SELECT MAX(seq_id)`)
5. **Config validation incomplete** — no validation for server.addr format, cert file existence
