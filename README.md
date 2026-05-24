# Magnitude

A fast, self-hosted vector database written in Go. Single binary, zero dependencies.

![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6?logo=typescript&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-1.70+-000000?logo=rust&logoColor=white)

## Quickstart

```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
make build
./magnitude          # http://localhost:8080 (REST) + :9090 (gRPC)
```

## Features

- **Pluggable indexes**: Flat, IVF, HNSW, SPANN, Sparse
- **Hybrid search**: Dense + sparse vectors with RRF fusion
- **Pre-filtered HNSW**: Metadata filtering during graph traversal
- **Multi-tenancy**: Tenant → Database → Collection hierarchy with quotas
- **CGO SIMD**: AVX2 (x86-64) + NEON (ARM64) distance computation
- **Clustering**: Gossip protocol, consistent hashing, automatic migration
- **WAL durability**: SQLite WAL + optional S3 WAL
- **gRPC + REST**: Dual protocol support
- **Snapshots**: Collection-level backup/restore

## SDKs

| Language | Package | Install |
|----------|---------|---------|
| Python | `magnitude-client` | `pip install magnitude-client` |
| TypeScript | `magnitude-js` | `npm install magnitude-js` |
| Rust | `magnitude` | `cargo add magnitude` |
| Java | `com.magnitude:magnitude-java` | Maven Central |
| .NET | `Magnitude.Client` | `dotnet add package Magnitude.Client` |
| Ruby | `magnitude` | `gem install magnitude` |
| Go | `pkg/client` | `go get github.com/POTATO-VE1/Magnitude/pkg/client` |

## Integrations

| Framework | Package | Install |
|-----------|---------|---------|
| LangChain (Python) | `langchain-magnitude` | `pip install langchain-magnitude` |
| LangChain (JS) | `@langchain/magnitude` | `npm install @langchain/magnitude` |
| LlamaIndex | `llama-index-vector-stores-magnitude` | `pip install llama-index-vector-stores-magnitude` |
| Vercel AI SDK | `@magnitude/ai-sdk` | `npm install @magnitude/ai-sdk` |
| Semantic Kernel | `SemanticKernel.Magnitude` | NuGet |
| Haystack | `haystack-magnitude` | `pip install haystack-magnitude` |

## API Reference

### REST API (port 8080)

#### Health
```
GET /v1/health
```

#### Tenants (Admin)
```
POST   /api/v2/tenants                          — Create tenant
GET    /api/v2/tenants                          — List tenants
GET    /api/v2/tenants/{id}                     — Get tenant
DELETE /api/v2/tenants/{id}                     — Delete tenant
```

#### Databases (Admin)
```
POST   /api/v2/tenants/{t}/databases            — Create database
GET    /api/v2/tenants/{t}/databases            — List databases
DELETE /api/v2/tenants/{t}/databases/{d}        — Delete database
```

#### Collections
```
POST   /api/v2/tenants/{t}/databases/{d}/collections     — Create collection
GET    /api/v2/tenants/{t}/databases/{d}/collections     — List collections
GET    /api/v2/tenants/{t}/databases/{d}/collections/{id} — Get collection
DELETE /api/v2/tenants/{t}/databases/{d}/collections/{id} — Delete collection
```

#### Vectors
```
POST   /api/v2/tenants/{t}/databases/{d}/collections/{id}/add      — Insert vectors
POST   /api/v2/tenants/{t}/databases/{d}/collections/{id}/query    — Search vectors
POST   /api/v2/tenants/{t}/databases/{d}/collections/{id}/delete   — Delete vectors
POST   /api/v2/tenants/{t}/databases/{d}/collections/{id}/hybrid   — Hybrid search
```

#### Snapshots
```
POST   /v1/collections/{id}/snapshot   — Export collection snapshot
POST   /v1/collections/{id}/restore    — Restore from snapshot
```

### gRPC API (port 9090)

See `internal/grpc/proto/magnitude.proto` for the full service definition.

## SDK Usage

### Python
```python
from magnitude import VectorDBClient

client = VectorDBClient("http://localhost:8080")
col = client.create_collection("docs", dimension=768, metric="cosine")
client.insert(col.id, ids=[1, 2], vectors=[vec1, vec2])
results = client.search(col.id, query=query_vec, top_k=10)
```

### TypeScript
```typescript
import { MagnitudeClient } from "magnitude-js";

const client = new MagnitudeClient("http://localhost:8080");
const tenant = await client.createTenant("my-tenant");
const db = await client.createDatabase(tenant.id, "my-db");
const col = await client.createCollection(tenant.id, db.id, "docs", { dimension: 768 });
await client.insert(tenant.id, db.id, col.id, [1], [[0.1, 0.2]]);
const results = await client.search(tenant.id, db.id, col.id, [0.1, 0.2], { topK: 10 });
```

### Rust
```rust
use magnitude::{MagnitudeClient, SearchOptions};

let client = MagnitudeClient::new("http://localhost:8080");
let col = client.create_collection("t", "db", "docs", 768, "cosine").await?;
client.insert("t", "db", &col.id, &[1, 2], &[vec1, vec2], None).await?;
let results = client.search("t", "db", &col.id, &query, Some(SearchOptions { top_k: Some(10), ..Default::default() })).await?;
```

### Go
```go
import "github.com/POTATO-VE1/Magnitude/pkg/client"

c := client.New("http://localhost:8080", "")
col, _ := c.CreateCollection(ctx, "docs", 128, "cosine", "hnsw")
c.Insert(ctx, col.ID, ids, vectors)
results, _ := c.Search(ctx, col.ID, query, 10, 0)
```

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

## Image Search (Optional)

Semantic image search using SigLIP embeddings. Requires Python + PyTorch.

```bash
cd python-client
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints-cpu.txt -e ".[image-search]"
magnitude-ingest --dir ./images --host http://localhost:8080
magnitude-ui   # http://localhost:3333
```

## License
MIT
