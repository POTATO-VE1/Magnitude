# Magnitude — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT LAYER                       │
│                                                      │
│  ┌─────────────────┐     ┌──────────────────────┐   │
│  │  Go HTTP Client │     │  Python CLIP Client  │   │
│  │  pkg/client/    │     │  python-client/      │   │
│  └────────┬────────┘     └──────────┬───────────┘   │
│           │                         │               │
│           │   REST V2 API (HTTP/S)   │               │
└───────────┼─────────────────────────┼───────────────┘
            │                         │
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│                   SERVER LAYER                       │
│                                                      │
│   cmd/server/main.go                                 │
│   ┌───────────────────────────────────────────────┐ │
│   │  Chi Router + Middleware                       │ │
│   │  /api/v2/tenants/{t}/databases/{d}/            │ │
│   │          collections/{id}/add|query|delete     │ │
│   └───────────────────┬───────────────────────────┘ │
│                       │                             │
│   internal/           │                             │
│   ┌───────────────────▼───────────────────────────┐ │
│   │  Collection Manager                            │ │
│   │  ┌───────────────────┐ ┌────────────────────┐ │ │
│   │  │  Index Engine     │ │  SysDB (SQLite/WAL)│ │ │
│   │  │  (pluggable)      │ │  (metadata)        │ │ │
│   │  │  ┌──────────────┐ │ └────────────────────┘ │ │
│   │  │  │Flat│IVF│HNSW │ │                        │ │
│   │  │  │SPANN│Sparse   │ │ ┌────────────────────┐│ │
│   │  │  └──────────────┘ │ │  WAL (SQLite)      ││ │
│   │  └───────────────────┘ │  (vector ops log)  ││ │
│   │                        └────────────────────┘│ │
│   │  ┌─────────────────┐  ┌─────────────────────┐│ │
│   │  │  Distance Engine│  │  Storage (mmap)     ││ │
│   │  │  (L2, Cosine)   │  │  (disk, float32)    ││ │
│   │  │  + SIMD accel   │  └─────────────────────┘│ │
│   │  └─────────────────┘                          │ │
│   └───────────────────────────────────────────────┘ │
│                                                      │
│   ┌───────────────────────────────────────────────┐ │
│   │  Observability (Prometheus + pprof + slog)    │ │
│   └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Component Breakdown

### `cmd/server/`
Entrypoint. Reads `config.yaml`, configures the Go runtime (GC tuning, GOMAXPROCS), initialises SysDB and WAL, wires the Chi router with V2 route handlers on the public port and admin/metrics on the internal port, starts the HTTP/TLS listener. Handles OS signals: SIGHUP for config hot-reload, SIGINT/SIGTERM for graceful 7-step shutdown.

### `internal/api/`
HTTP handlers and middleware:
- **handlers.go**: V1 legacy collection and vector endpoints.
- **handlers_v2.go**: V2 multi-tenant API — tenant CRUD, database CRUD, scoped collection and vector operations, hybrid search.
- **middleware.go**: Request logging, rate limiting, API key authentication, CORS.
- **context.go**: Request-scoped context helpers.

### `internal/index/`
Pluggable index engine. All implementations satisfy the `Index` interface (`Insert`, `Search`, `Delete`, `Len`, `Rebuild`, `Flush`):
- **flat/**: Brute-force exact search. O(n) — used as correctness baseline.
- **ivf/**: Inverted File Index with K-Means clustering. Sub-linear search via configurable `nprobe`.
- **hnsw/**: Hierarchical Navigable Small World graph. O(log n) query, best recall/speed tradeoff. Configurable `M`, `efConstruction`, `efSearch`.
- **spann/**: SPANN index for disk-resident large-scale approximate search.
- **sparse/**: Sparse vector index (BM25-style) for text-native retrieval and hybrid search.
- **recall.go**: Recall measurement utilities for benchmarking index quality.

### `internal/distance/`
Distance metric computation:
- **distance.go**: Pure-Go L2 and cosine distance implementations.
- **simd.c / simd_amd64.go**: CGO-based SIMD-accelerated kernels for AMD64.
- **simd_stub.go / batch_pure.go**: Fallback for non-AMD64 platforms.
- **dispatch.go**: Build-tag-based dispatch selecting SIMD or pure-Go at compile time.

### `internal/metadata/`
SQLite-backed system database (SysDB):
- **sqlite.go**: Core SQLite operations — collection CRUD, vector metadata storage, WAL-mode configuration.
- **tenant.go**: Multi-tenancy schema — tenant/database CRUD, quota enforcement, cascading deletes.
- **filter.go**: Metadata filter parsing and SQL query generation for filtered vector search.

### `internal/storage/`
On-disk persistence:
- **wal.go**: SQLite-backed write-ahead log for vector insert/delete operations.
- **s3wal.go**: Optional S3-backed WAL for distributed setups.
- **mmap_unix.go**: Memory-mapped file I/O for zero-copy vector access.
- **admission.go**: Backpressure and admission control for write-heavy workloads.
- **compaction.go**: Background segment compaction.
- **format.go**: Binary serialisation format for vector data.

### `internal/collection/`
Orchestrates index + storage + metadata for a single collection:
- **collection.go**: The `Manager` — handles create/delete collection, insert/search/delete vectors, WAL replay, flush, and scoped multi-tenant operations.
- **fork.go**: Collection snapshotting (zero-copy fork).

### `internal/config/`
Configuration loading from YAML with environment variable overrides.

### `internal/observability/`
Prometheus metric registration and structured logging setup.

### `internal/security/`
API key validation and TLS configuration.

### `internal/cache/`
In-memory caching for frequently accessed metadata.

### `internal/cluster/`
Distributed coordination primitives (rendezvous hashing, node management) — for future multi-node deployments.

### `internal/gc/`
Garbage collection service — manages tombstoned vector cleanup and index rebuild scheduling.

### `internal/quantize/`
Vector quantisation utilities for memory-efficient storage.

### `pkg/client/`
Typed Go HTTP client wrapping all V2 API endpoints. Intended for use by Go applications embedding Magnitude as a dependency.

### `python-client/`
- **embedder.py**: CLIP embedding wrapper using `sentence-transformers` (`clip-ViT-B-32`).
- **ingest.py**: Batch-encodes images using CLIP, sends embeddings to the Go server via REST. Shows Rich progress bar.
- **search.py**: Encodes a text query with CLIP, queries the server for top-k results, renders them in a local HTML lightbox auto-opened in the browser.
- **vectordb_client.py**: Python HTTP client wrapping the V2 API.

## Data Flow — Ingestion

```
Image file on disk
    → Python CLIP encoder (sentence-transformers, embedder.py)
    → 512-dim float32 vector
    → POST /api/v2/tenants/{t}/databases/{d}/collections/{id}/add (REST)
    → Go server receives (rate limited, authenticated)
    → Written to WAL (storage/wal.go)
    → Indexed in chosen index engine (hnsw/, ivf/, flat/, etc.)
    → Metadata written to SQLite SysDB (metadata/)
```

## Data Flow — Search

```
Natural language query string
    → Python CLIP encoder (same model, text branch)
    → 512-dim float32 vector
    → POST /api/v2/tenants/{t}/databases/{d}/collections/{id}/query (REST)
    → Go server: Index.Search(ctx, query_vec, k, nprobe)
    → Returns top-k (id, distance, score) triples
    → Python client fetches image paths for each id
    → Renders HTML lightbox, opens in browser
```

## Design Decisions

**Why Go for the backend?**
Go's goroutine model and value-type memory layout make it ideal for a server that holds large index structures in memory and serves concurrent queries. The compiled binary has no runtime dependency, simplifying deployment.

**Why a pluggable index architecture?**
Different workloads need different tradeoffs. Flat is perfect for small datasets and correctness testing. IVF balances recall and speed for medium datasets. HNSW delivers the best recall at sub-linear cost. SPANN handles datasets too large for RAM. The common `Index` interface lets users switch with a single config change.

**Why HNSW over brute-force?**
For datasets >1K vectors, HNSW delivers O(log n) query time vs O(n) for brute-force. The tradeoff is slightly lower recall (configurable via `ef`) — acceptable for semantic image search where approximate is fine.

**Why `modernc.org/sqlite` over `mattn/go-sqlite3`?**
`modernc.org/sqlite` is a pure-Go SQLite implementation — no CGO dependency for the database layer. This simplifies cross-compilation. CGO is only required for the optional SIMD distance kernels, which have a pure-Go fallback.

**Why SQLite for metadata?**
Magnitude is designed to be self-contained. SQLite in WAL mode handles concurrent reads well and requires zero operational overhead — no separate DB process to manage.

**Why CLIP for embeddings?**
CLIP maps both text and images into the same embedding space, enabling natural language queries over image corpora without training data or fine-tuning.
