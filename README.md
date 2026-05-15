# Magnitude VectorDB

A lightweight, self-hosted vector database built in Go with a Python CLIP client for semantic image search.

## Quick Start

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
make run
```

That's it. The server starts on `http://localhost:8080` with sensible defaults — no config file, no certs, no setup required. Customize by editing `config.yaml` (optional).

### Python Client

```bash
cd python-client
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"

# Ingest images from a directory
magnitude-ingest --dir ./your-images

# Interactive search
magnitude-search
```

## Docker

```bash
docker compose up --build
```

## Features

- **Pluggable Indexing** — Flat (brute-force), IVF, HNSW, SPANN
- **Hybrid Search** — Dense vector + sparse BM25 text search with RRF fusion
- **Multi-Tenancy** — Tenant → Database → Collection isolation with quotas
- **Product Quantization** — Compress vectors for lower memory footprint
- **Crash-Safe WAL** — SQLite-backed write-ahead log with configurable sync modes
- **HNSW Snapshots** — Graph persisted to disk, fast restart without full WAL replay
- **Distributed Cluster** — Consistent hashing, gossip protocol, failure detection, auto-migration
- **Observability** — Prometheus metrics, pprof, structured JSON logging
- **Production-Ready** — TLS, API key auth, rate limiting, graceful shutdown, config hot-reload

## API

```bash
# Create collection
curl -X POST http://localhost:8080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "dimension": 128, "metric": "cosine", "index_type": "hnsw"}'

# Insert vectors
curl -X POST http://localhost:8080/v1/collections/test/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2], "vectors": [[0.1, ...], [0.2, ...]]}'

# Search
curl -X POST http://localhost:8080/v1/collections/test/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, ...], "k": 10}'
```

## Python Client

```python
from magnitude import VectorDBClient

client = VectorDBClient("http://localhost:8080")
client.create_collection("images", dimension=512)
client.insert("images", ids=[1], vectors=[[0.1, ...]])
results = client.search("images", query=[0.1, ...], top_k=5)
```

## Configuration

Edit `config.yaml` — all values have sensible defaults. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `server.addr` | `:8080` | HTTP listen address |
| `server.certFile` | `""` | TLS cert path (empty = plain HTTP) |
| `index.type` | `flat` | Index type: `flat`, `ivf`, `hnsw`, `spann` |
| `auth.keyHashes` | `[]` | API key SHA-256 hashes (empty = no auth) |
| `cluster.enabled` | `false` | Enable distributed cluster mode |

## Production Setup

For production, enable TLS and API key authentication:

```bash
# 1. Generate TLS certs
mkdir -p certs && openssl req -x509 -newkey rsa:4096 \
  -keyout certs/server.key -out certs/server.crt \
  -days 365 -nodes -subj '/CN=your-domain.com'

# 2. Generate an API key hash
echo -n "your-secret-api-key" | sha256sum
```

Then update `config.yaml`:
```yaml
server:
  addr: ":8443"
  certFile: "certs/server.crt"
  keyFile: "certs/server.key"
auth:
  keyHashes:
    - "your-sha256-hash-here"
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram.

## License

MIT — see [LICENSE](LICENSE).
