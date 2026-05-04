# AI Agent Workflows in Magnitude

This document describes how AI agents were used throughout the design, development, and iteration of Magnitude — from architecture decisions to debugging to documentation.

## Agent Tools Used

- **Claude Code** — primary coding agent used for Go backend development, API design, and debugging
- **Antigravity** — used for multi-file refactoring passes, documentation generation, and coordinating changes across the Go and Python layers simultaneously

## How Agents Shaped the Architecture

### HNSW Implementation
The decision to implement HNSW (Hierarchical Navigable Small World graphs) from scratch rather than binding to an external C library was informed by an agentic research session. Claude Code was prompted to compare `hnswlib` bindings, `faiss` CGO wrappers, and a pure-Go implementation across three criteria: deployment simplicity, memory control, and cross-platform build reproducibility. The pure-Go path won on all three. Claude Code then scaffolded the initial graph construction loop, which was iteratively refined through several agent-assisted debugging passes focused on edge cases in the layer assignment probability function.

### Multi-Index Architecture
Magnitude doesn't just support HNSW — the `internal/index` package defines a common `Index` interface implemented by Flat, IVF, HNSW, and SPANN backends. The decision to build this pluggable index architecture was refined through agentic design sessions where Claude Code was asked to evaluate tradeoff matrices: Flat for correctness baselines, IVF for balanced recall/speed, HNSW for best recall at sub-linear cost, and SPANN for disk-resident large-scale workloads. The agent produced the initial interface contract (`Insert`, `Search`, `Delete`, `Len`, `Rebuild`, `Flush`) and the dispatch layer that selects the index type from `config.yaml`.

### Multi-Tenancy Design
The Tenant → Database → Collection hierarchy was designed through a back-and-forth with Claude Code where I described the access isolation requirements and the agent produced three candidate schema designs. The pure-Go SQLite driver (`modernc.org/sqlite`) with WAL-mode journaling was chosen over `mattn/go-sqlite3` to eliminate the CGO dependency for the metadata layer — an agent-suggested tradeoff that simplified cross-compilation. The agent benchmarked WAL vs DELETE-mode journal, showing WAL outperforming by ~3x under concurrent read workloads typical of vector search patterns.

### REST API Surface
The V2 API route structure (`/api/v2/tenants/{tenant}/databases/{db}/collections/{id}/...`) was co-designed with Claude Code. I described the resource ownership semantics and the agent generated the Chi router stubs, middleware chain, and request validation logic. The cross-tenant isolation design — returning 404 instead of 403 to prevent existence leakage — was an agent recommendation that I adopted. I then reviewed, adjusted naming conventions, and wired in the business logic.

### Distance Metrics & SIMD Acceleration
The `internal/distance` package — including the SIMD-accelerated path via CGO (`simd.c` + `simd_amd64.go`) with a pure-Go fallback (`batch_pure.go`) — was developed with heavy agent assistance. Claude Code generated the initial C SIMD kernels for L2 and cosine distance, the build-tag dispatch layer (`dispatch.go`), and the fallback implementations. I validated correctness against brute-force reference outputs and tuned the batch sizes.

## Python Client Development

The `python-client` was developed with heavy agent assistance:

- `ingest.py`: Claude Code wrote the initial batch ingestion loop with tqdm progress tracking. I directed it to add retry logic, configurable batch sizes, and the Rich-formatted summary table at the end.
- `search.py`: The interactive query loop and the HTML lightbox renderer were agent-generated from a natural language spec I wrote describing the UX I wanted. The agent produced the Jinja-free inline HTML template in a single pass.
- `embedder.py`: The CLIP embedding wrapper was agent-scaffolded, with the model selection (`clip-ViT-B-32` via `sentence-transformers`) based on an agent recommendation that traded raw performance for setup simplicity.
- Dependency selection (`sentence-transformers` over raw `transformers` + `PIL`) was an agent recommendation based on the tradeoff between setup complexity and inference speed for CLIP ViT-B/32.

## Debugging Sessions

Several non-trivial bugs were resolved through agent-assisted debugging:

1. **HNSW graph corruption on concurrent inserts** — Claude Code identified a missing mutex around the layer-0 neighbour list update and generated the fix with proper RWMutex granularity.
2. **SQLite UNIQUE constraint violation during tenant re-creation** — agent traced it to a missing `ON CONFLICT` clause in the schema migration and patched it.
3. **CLIP embedding dimension mismatch between ingest and query** — agent spotted that `sentence-transformers` was returning normalised vs. unnormalised vectors depending on the `normalize_embeddings` flag; it added the flag explicitly in both scripts.
4. **WAL replay ordering** — during shutdown recovery, the agent identified that WAL entries needed to be replayed in strict sequence order to maintain index consistency, and refactored the `collection.NewManager` initialisation path.
5. **Mmap region sizing on 32-bit systems** — the agent flagged that the mmap capacity calculation in `storage/mmap_unix.go` could overflow on 32-bit architectures and added proper bounds checking.

## Agent-Assisted Documentation

This README, the ARCHITECTURE document, and all docstrings in the Go packages were drafted with Claude Code and then edited for accuracy and tone. The agent was given the actual source code and asked to infer intent — the output was then corrected where the inference was wrong, making it a collaborative rather than generative process.

## Agent-Assisted Infrastructure

- **GitHub Actions CI**: The CI workflow (`.github/workflows/go.yml`) was generated by Antigravity in a single pass, including the build and test steps.
- **Docker support**: The `Dockerfile` and `docker-compose.yml` were agent-generated and then reviewed for correctness — the agent correctly identified that `CGO_ENABLED=1` was needed because of the SIMD C code in `internal/distance`.
- **Graceful shutdown sequence**: The 7-step shutdown sequence in `cmd/server/main.go` (drain HTTP → stop internal server → flush indexes → close WAL → close SysDB) was designed collaboratively with Claude Code, which generated the initial ordering and timeout propagation via `context.WithTimeout`.

## Planned Agent Workflows

- **Benchmark automation**: An agent workflow to run nightly ingestion + query benchmarks against a pinned COCO dataset and commit results to `bench/results.json`
- **Auto-documentation**: A CI agent step that re-generates API docs from Go source comments on every merge to main
- **Observability dashboards**: Agent-assisted generation of Grafana dashboard JSON from the Prometheus metrics already exported via `internal/observability`
