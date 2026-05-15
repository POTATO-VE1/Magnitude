# Magnitude VectorDB

A lightweight, self-hosted vector database built in Go with a Python CLIP client for semantic image search.

## Quick Start

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
make run
```

That's it. The server starts on `http://localhost:8080` with sensible defaults — no config file, no certs, no setup required. Customize by editing `config.yaml` (optional).

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

### REST (curl)

```bash
# Create a collection
curl -X POST http://localhost:8080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 128, "metric": "cosine", "index_type": "hnsw"}'

# Insert vectors
curl -X POST http://localhost:8080/v1/collections/{collection_id}/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2], "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}'

# Search
curl -X POST http://localhost:8080/v1/collections/{collection_id}/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, ...], "k": 10}'
```

### Python Client

Install the client:

```bash
cd python-client
pip install -e .
```

Simple usage (v1 API — single-tenant):

```python
from magnitude import VectorDBClient

client = VectorDBClient()  # defaults to http://localhost:8080
col = client.create_collection("docs", dimension=128, metric="cosine")
client.insert(col.id, ids=[1, 2], vectors=[[0.1, ...], [0.3, ...]])
results = client.search(col.id, query=[0.1, ...], top_k=5)

for r in results:
    print(f"  id={r.id}  score={r.score:.4f}")
```

### Go Client

```go
import "github.com/POTATO-VE1/Magnitude/pkg/client"

c := client.New("http://localhost:8080", "")
col, _ := c.CreateCollection(ctx, "docs", 128, "cosine", "hnsw")
_ = c.Insert(ctx, col.ID, ids, vectors)
results, _ := c.Search(ctx, col.ID, query, 10, 0)
```

## Semantic Image Search (CLIP)

Magnitude ships with a CLIP-powered image search pipeline. It uses the `clip-ViT-B-32` model (512D vectors) to embed images and text queries into the same vector space.

```bash
cd python-client
pip install -e ".[all]"  # installs CLIP, torch, rich, tqdm

# Index a folder of images
magnitude-ingest --dir ./photos

# Interactive search REPL — type natural language, get ranked images
magnitude-search
```

The ingest script uses the multi-tenant v2 API internally:

```python
from vectordb_client import VectorDBClient

client = VectorDBClient()  # http://127.0.0.1:8080
tenant_id = client.get_or_create_tenant("default")
db_id = client.get_or_create_database(tenant_id, "images_db")
col_id = client.get_or_create_collection(tenant_id, db_id, "clip_images", dimension=512)

client.insert_vectors(tenant_id, db_id, col_id, ids=[1], vectors=[[...]])
results = client.search_vectors(tenant_id, db_id, col_id, query_embedding=[...])
```

## API Reference

### v1 — Simple (single-tenant)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/collections` | Create collection |
| `GET` | `/v1/collections` | List collections |
| `GET` | `/v1/collections/{id}` | Get collection |
| `DELETE` | `/v1/collections/{id}` | Delete collection |
| `POST` | `/v1/collections/{id}/vectors` | Insert vectors |
| `POST` | `/v1/collections/{id}/search` | Search vectors |
| `DELETE` | `/v1/collections/{id}/vectors/{vid}` | Delete vector |
| `GET` | `/v1/health` | Health check |

### v2 — Multi-tenant (tenant → database → collection)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/tenants` | Create tenant |
| `GET` | `/api/v2/tenants` | List tenants |
| `POST` | `/api/v2/tenants/{t}/databases` | Create database |
| `POST` | `/api/v2/tenants/{t}/databases/{d}/collections` | Create collection |
| `POST` | `/api/v2/tenants/{t}/databases/{d}/collections/{c}/add` | Insert vectors |
| `POST` | `/api/v2/tenants/{t}/databases/{d}/collections/{c}/query` | Search |
| `POST` | `/api/v2/tenants/{t}/databases/{d}/collections/{c}/hybrid` | Hybrid search |
| `POST` | `/api/v2/tenants/{t}/databases/{d}/collections/{c}/delete` | Delete vector |

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
