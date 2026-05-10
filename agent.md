# Agent Guide: Magnitude VectorDB

Welcome, AI Agent! This guide helps you navigate and contribute to the Magnitude project.

## Project Summary
**Magnitude** is a high-performance, self-hosted vector database built in Go, featuring a CLIP-powered Python client for semantic image search. It supports multi-tenancy, pluggable indexing (HNSW, IVF, etc.), and SIMD acceleration.

## Tech Stack
- **Backend:** Go 1.25+, Chi (Router), SQLite (Metadata/WAL), Prometheus/pprof (Observability)
- **Frontend/Client:** Python 3.9+, CLIP (via `sentence-transformers`), Rich (CLI), HTML/CSS (Search UI)
- **Indexing:** HNSW, IVF, Flat, SPANN, Sparse (BM25)
- **Storage:** mmap for zero-copy vector access, SQLite for WAL and SysDB

## Key Directories
- `cmd/server/`: Backend entrypoint and server startup logic.
- `internal/`: Core database logic (API, indexing, distance, metadata, storage).
- `pkg/client/`: Go client library.
- `python-client/`: Python client for ingestion and search.
- `data/`: Default directory for SQLite DBs and vector files.
- `bench/`: Benchmarking scripts.
- `examples/`: Usage examples.

## Getting Started

### Running the Backend
```bash
go run cmd/server/main.go
```
The server reads `config.yaml` for configuration. It supports hot-reload via `SIGHUP`.

### Setting up the Python Client
```bash
cd python-client
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Ingestion & Search
1. **Ingest:** `python ingest.py --dir path/to/images`
2. **Search:** `python search.py` (opens a browser with results)

## Development Guidelines

### Go Backend (`internal/`)
- **Interfaces:** Most core components (Index, Distance) use interfaces. When adding a new index, implement the `Index` interface in `internal/index/`.
- **Error Handling:** Use structured logging (`log/slog`) and return clear errors.
- **Concurrency:** Leverages Go's goroutines and channels. Be mindful of mutexes in shared structures (e.g., `CollectionManager`).
- **SIMD:** Distance kernels have SIMD implementations (`simd.c`). If modifying, ensure the pure-Go fallback works.
- **SQLite:** Uses `modernc.org/sqlite` (pure Go). No CGO needed for the DB.

### Python Client (`python-client/`)
- **Embeddings:** Uses CLIP (`clip-ViT-B-32`). Model weights are downloaded to `~/.cache/torch`.
- **API Communication:** All communication with the server happens via the REST V2 API.
- **Client Library:** Use `vectordb_client.py` as the base for any new Python tools.

## Common Tasks for Agents
- **Adding an Index:** Implement the `Index` interface in a new package under `internal/index/`.
- **Extending the API:** Add handlers in `internal/api/handlers_v2.go` and update the route registration.
- **Improving Search:** Tweak `efSearch` or `nprobe` in `config.yaml` or via query params.
- **Debugging:** Use `pprof` (enabled by default on the internal admin port).

## Useful Commands
- **Lint Go:** `go vet ./...`
- **Run Tests:** `go test ./...`
- **Build Server:** `go build -o magnitude cmd/server/main.go`
- **Check Metrics:** `curl localhost:2112/metrics` (if using default config)

Refer to `ARCHITECTURE.md` for a detailed system diagram and `README.md` for user-facing instructions.
