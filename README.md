# Magnitude

A fast, self-hosted vector database written in Go. Magnitude ships with a Python client designed for semantic image search using OpenAI's CLIP model.

## Quickstart: Semantic Image Search

This guide gets you up and running locally. The database runs directly on your machine—**Docker is completely optional.**

### 1. Start the Server

Boot up the VectorDB server. It starts on port `8080` with zero configuration required.

**Option A: Native**
```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
make run
```

**Option B: Docker**
```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
docker compose up --build
```

### 2. Prepare the Python Environment

Open a new terminal window. Leave the Go server running.

```bash
cd python-client
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

### 3. Download Sample Data (MS-COCO 5k)

We'll use the COCO 2017 validation set, which contains exactly 5,000 images (approx 1GB).

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./images
```

### 4. Ingest Images

The ingest script converts every image into a 512-dimensional vector using the CLIP model and loads them into Magnitude.

```bash
magnitude-ingest --dir ./images/val2017
```

### 5. Search

Launch the interactive search interface:

```bash
magnitude-search
```

Type a query like `red car` or `dog catching a frisbee`. The client converts your text to a vector, searches the database, and opens the results in your web browser.

---

## Client Usage

### Python Client

The Python package provides a clean interface for programmatic access to the database.

```python
from magnitude import VectorDBClient

# Connect to local server
client = VectorDBClient("http://localhost:8080")

# Create a collection
col = client.create_collection("documents", dimension=128, metric="cosine")

# Insert vectors
client.insert("documents", ids=[1, 2], vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]])

# Search
results = client.search("documents", query=[0.1, 0.2, ...], top_k=5)
for r in results:
    print(f"ID: {r.id}, Score: {r.score}")
```

### Go Client

```go
import "github.com/POTATO-VE1/Magnitude/pkg/client"

c := client.New("http://localhost:8080", "")
col, _ := c.CreateCollection(ctx, "docs", 128, "cosine", "hnsw")
_ = c.Insert(ctx, col.ID, ids, vectors)
results, _ := c.Search(ctx, col.ID, query, 10, 0)
```

---

## Architecture & APIs

Magnitude exposes two distinct REST APIs to support different deployment scales.

- **v1 API (`/v1/collections`)**: A flat structure. Best for simple applications, local development, and single-tenant use cases. This is what the Python and Go code snippets above use.
- **v2 API (`/api/v2/tenants`)**: A strict `Tenant → Database → Collection` hierarchy. Designed for SaaS and enterprise deployments where strict data isolation is required. The `magnitude-ingest` CLI tool uses this internally to ensure image datasets are properly isolated.

For full system architecture and internal design, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Deployment

### Docker

If you prefer containerization, Magnitude includes a lightweight Docker setup.

```bash
docker compose up --build
```

### Production Configuration

Edit `config.yaml` to secure the server before exposing it to the internet.

1. **Enable TLS**: Generate certificates and add the paths to `certFile` and `keyFile`.
2. **Enable Authentication**: Add SHA-256 hashes of your API keys to the `auth.keyHashes` list.

```yaml
server:
  addr: ":8443"
  certFile: "certs/server.crt"
  keyFile: "certs/server.key"
auth:
  keyHashes:
    - "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
```

## License

MIT
