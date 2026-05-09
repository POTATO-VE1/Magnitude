# Magnitude VectorDB

A lightweight, self-hosted vector database built in Go with a Python CLIP client for semantic image search.

## Quick Start

### 1. Clone & Generate Certs

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
mkdir -p certs && openssl req -x509 -newkey rsa:4096 \
  -keyout certs/server.key -out certs/server.crt \
  -days 365 -nodes -subj '/CN=localhost'
```

### 2. Start the Server

```bash
go run cmd/server/main.go
```

### 3. Set Up Python Client

```bash
cd python-client
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"
```

### 4. Ingest & Search Images

```bash
# Ingest images from a directory
magnitude-ingest --dir ./your-images

# Interactive search
magnitude-search
```

## Docker

```bash
# Generate certs first (same as above)
docker compose up --build
```

## Features

- **Pluggable Indexing** — Flat (brute-force), IVF, HNSW, SPANN
- **Hybrid Search** — Dense vector + sparse BM25 text search with RRF fusion
- **Multi-Tenancy** — Tenant → Database → Collection isolation with quotas
- **Bloom Filters** — Skip unnecessary segment reads
- **Crash-Safe WAL** — Binary-encoded write-ahead log with configurable sync modes
- **HNSW Snapshots** — Graph persisted to disk, fast restart without full WAL replay
- **Observability** — Prometheus metrics, pprof, structured JSON logging
- **Production-Ready** — TLS, API key auth, rate limiting, graceful shutdown, config hot-reload

## API

```bash
# Create collection
curl -k -X POST https://localhost:8443/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "dimension": 128, "metric": "cosine", "index_type": "hnsw"}'

# Insert vectors
curl -k -X POST https://localhost:8443/v1/collections/test/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2], "vectors": [[0.1, ...], [0.2, ...]]}'

# Search
curl -k -X POST https://localhost:8443/v1/collections/test/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, ...], "k": 10}'
```

## Python Client

```python
from magnitude import VectorDBClient

client = VectorDBClient("https://localhost:8443", verify_ssl=False)
client.create_collection("images", dimension=512)
client.insert("images", ids=[1], vectors=[[0.1, ...]])
results = client.search("images", query=[0.1, ...], top_k=5)
```

## Configuration

Edit `config.yaml` — all values have sensible defaults. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `server.addr` | `:8443` | HTTPS listen address |
| `index.type` | `flat` | Index type: `flat`, `ivf`, `hnsw`, `spann` |
| `auth.keyHashes` | `[]` | API key SHA-256 hashes (empty = no auth) |
| `walSync.syncMode` | `per-write` | WAL sync: `per-write`, `delayed`, `none` |

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram.

## Security

> **Warning:** Auth is disabled by default (`keyHashes: []`). Generate a key hash before exposing to any network:
> ```bash
> echo -n "your-secret-key" | sha256sum
> ```

## License

MIT — see [LICENSE](LICENSE).
