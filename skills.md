# Technical Skills Demonstrated in Magnitude

A map of the engineering disciplines, algorithms, and systems-level concepts exercised in this project.

## Systems Programming (Go)

| Skill | Where Applied |
|---|---|
| Concurrent data structures | HNSW graph with RWMutex-guarded neighbour lists |
| Memory layout optimisation | Dense float32 vector storage with contiguous slice allocation, mmap-backed regions |
| SIMD acceleration | CGO-based SIMD distance kernels (`internal/distance/simd.c`) with pure-Go fallback |
| HTTP server design | Chi router, middleware chain, context propagation, graceful 7-step shutdown |
| SQLite integration | WAL-mode metadata store via `modernc.org/sqlite` (pure-Go driver) |
| Configuration management | YAML config with hot-reload via SIGHUP (`gopkg.in/yaml.v3`) |
| Build tooling | Go modules, multi-package workspace, cross-platform build targets |
| Observability | Prometheus metrics export, pprof integration, structured JSON logging (`log/slog`) |
| Memory-mapped I/O | `mmap_unix.go` for zero-copy vector access on Linux/macOS |

## Algorithm Implementation

| Skill | Where Applied |
|---|---|
| HNSW (Hierarchical Navigable Small World) | Core vector index — O(log n) insert, sub-linear query (`internal/index/hnsw`) |
| IVF (Inverted File Index) | Voronoi-cell-based partitioning with configurable nprobe (`internal/index/ivf`) |
| SPANN | Disk-resident approximate search for large-scale workloads (`internal/index/spann`) |
| BM25 / Sparse retrieval | Sparse vector index for text-native queries (`internal/index/sparse`) |
| Flat (brute-force) | Exact nearest-neighbour baseline for correctness testing (`internal/index/flat`) |
| Cosine similarity | Default distance metric for CLIP embedding space |
| Euclidean (L2) distance | Alternative metric, selectable per collection |
| Recall measurement | Quantitative recall benchmarking against brute-force ground truth (`internal/index/recall.go`) |
| Batch processing | Python ingest pipeline with configurable batch sizes |

## Machine Learning Infrastructure

| Skill | Where Applied |
|---|---|
| CLIP (Contrastive Language–Image Pretraining) | Text → 512-dim embedding for cross-modal image retrieval |
| sentence-transformers | CLIP inference runtime (`clip-ViT-B-32`) via `python-client/embedder.py` |
| Embedding normalisation | Consistent L2-normalised vectors for cosine similarity correctness |
| Cross-modal retrieval | Natural language queries returning ranked image results |
| Batch inference | Vectorised CLIP encoding over image corpora |

## Database & Storage

| Skill | Where Applied |
|---|---|
| Multi-tenancy design | Tenant → Database → Collection isolation hierarchy with quota enforcement |
| WAL-mode SQLite | Concurrent-read-optimised metadata persistence (`internal/metadata`) |
| Write-ahead logging | Separate SQLite WAL for vector operations (`internal/storage/wal.go`) |
| S3-backed WAL | Optional remote WAL persistence for distributed setups (`internal/storage/s3wal.go`) |
| Vector serialisation | Binary float32 encoding for on-disk embedding storage |
| Admission control | Backpressure mechanism for write-heavy workloads (`internal/storage/admission.go`) |
| Compaction | Background segment compaction for storage efficiency (`internal/storage/compaction.go`) |
| Schema design | Normalised metadata schema with foreign key constraints |

## API & Client Design

| Skill | Where Applied |
|---|---|
| RESTful resource modelling | V2 API with nested resource addressing and cross-tenant isolation |
| Hybrid search | Combined dense + sparse retrieval via `/hybrid` endpoint |
| Go HTTP client | Typed Go client in `pkg/client` mirroring the server API |
| Python requests client | Thin HTTP wrapper in `python-client/vectordb_client.py` |
| CLI design | Rich-formatted terminal UI for ingest progress and search results |

## Infrastructure & Operations

| Skill | Where Applied |
|---|---|
| Graceful shutdown | 7-step ordered shutdown sequence with timeout propagation |
| Config hot-reload | SIGHUP-triggered runtime reconfiguration without restart |
| Rate limiting | Per-endpoint rate limiting (search/insert) with configurable burst |
| API key authentication | SHA-256 hashed API key validation (keys never stored in plaintext) |
| TLS support | HTTPS with configurable cert/key paths |
| Garbage collection tuning | Runtime GC percent configuration for mmap-heavy workloads |
| Docker containerisation | Multi-stage Dockerfile with health checks |
| CI/CD | GitHub Actions pipeline for build and test |

## Developer Experience

| Skill | Where Applied |
|---|---|
| Cross-platform installation docs | README covers Ubuntu, Fedora, Arch, macOS, Windows |
| Sample dataset integration | COCO 2017 val set instructions for realistic testing |
| Interactive search UI | Auto-opening HTML lightbox rendered from search results |
| Virtualenv hygiene | Python client ships with isolated venv setup |

## AI-Assisted Development

| Skill | Where Applied |
|---|---|
| Agent-directed architecture | Claude Code used for design decisions, tradeoff analysis |
| Agent-assisted debugging | Multi-session debugging of concurrency, WAL replay, and embedding bugs |
| Prompt engineering | Structured prompts to generate Go stubs and Python CLI scaffolding |
| Code review with agents | Agent-assisted review passes for API consistency and error handling |
