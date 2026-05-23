# Magnitude

A fast, self-hosted vector database written in Go. Single binary, zero dependencies.

![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)

## Quickstart — Vector Database

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
make build
./magnitude          # http://localhost:8080
```

That's it. No Docker, no Python, no external services.

### Python Client

```bash
pip install magnitude-client
```

```python
from magnitude import VectorDBClient

client = VectorDBClient("http://localhost:8080")
col = client.create_collection("docs", dimension=768, metric="cosine")
client.insert(col.id, ids=[1, 2], vectors=[vec1, vec2], metadata=[{"title": "doc1"}, {"title": "doc2"}])
results = client.search(col.id, query=query_vec, top_k=10)
```

### Go Client

```go
import "github.com/POTATO-VE1/Magnitude/pkg/client"

c := client.New("http://localhost:8080", "")
col, _ := c.CreateCollection(ctx, "docs", 128, "cosine", "hnsw")
c.Insert(ctx, col.ID, ids, vectors)
results, _ := c.Search(ctx, col.ID, query, 10, 0)
```

### REST API

```bash
# Create collection
curl -X POST http://localhost:8080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 768, "metric": "cosine"}'

# Insert vectors
curl -X POST http://localhost:8080/v1/collections/{id}/add \
  -H "Content-Type: application/json" \
  -d '{"ids": [1], "vectors": [[0.1, 0.2, ...]], "metadata": [{"key": "value"}]}'

# Search
curl -X POST http://localhost:8080/v1/collections/{id}/query \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, ...], "k": 10}'
```

---

## Image Search (Optional)

Semantic image search using SigLIP embeddings. Requires Python + PyTorch.

### Setup

```bash
cd python-client
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints-cpu.txt -e ".[image-search]"
```

### Ingest & Search

```bash
# Download sample images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./images

# Ingest
magnitude-ingest --dir ./images/val2017 --host http://localhost:8080

# Start web UI
magnitude-ui
# Open http://localhost:3333
```

### Python API

```python
from magnitude import VectorDBClient
from magnitude import SigLIPEmbedder  # requires pip install magnitude-client[embed]

embedder = SigLIPEmbedder()
client = VectorDBClient("http://localhost:8080")

col = client.create_collection("images", dimension=768)
vectors = embedder.embed_images(["cat.jpg", "dog.jpg"])
client.insert(col.id, ids=[1, 2], vectors=vectors)

results = client.search(col.id, query=embedder.embed_text("a cute cat"), top_k=5)
```

---

## Features

- **Pluggable indexes**: Flat, IVF, HNSW, SPANN, Sparse
- **Hybrid search**: Dense + sparse vectors with RRF fusion
- **Multi-tenancy**: Tenant → Database → Collection hierarchy with quotas
- **Pre-filtered HNSW**: Metadata filtering during graph traversal, not post-filter
- **CGO AVX2 SIMD**: Hardware-accelerated distance computation on x86-64
- **Clustering**: Gossip protocol, consistent hashing, automatic data migration
- **WAL durability**: SQLite WAL + optional S3 WAL for distributed setups

## Configuration

All settings in `config.yaml`. Data persisted in `./data`.

```yaml
server:
  addr: ":8443"
  certFile: "certs/server.crt"
  keyFile: "certs/server.key"
auth:
  keyHashes:
    - "<SHA-256 hash of your API key>"
```

For clustering, WAL internals, and indexing architecture, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License
MIT
