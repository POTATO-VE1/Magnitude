# Magnitude

A fast, self-hosted vector database written in Go with a built-in Web UI for semantic image search.

![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)

## Quickstart

### 1. Build and Run the Server

First, clone the repository:
```bash
git clone https://github.com/POTATO-VE1/Magnitude.git
cd Magnitude
```

**Native (Go 1.25+):**
```bash
make build
make run    # Server starts on http://localhost:8080
```

**Docker:**
```bash
docker compose up --build -d  # Server starts on http://localhost:8080
```

### 2. Install the Client & UI

Requires Python 3.9+. We use `--extra-index-url` to install CPU-only PyTorch to avoid massive GPU binaries.

```bash
cd python-client
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

### 3. Ingest Images & Search

Download a sample dataset (e.g., MS-COCO) and ingest it:

**Option A: Quick Test (~5k images, 1GB)**
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./images
magnitude-ingest --dir ./images/val2017 --host http://localhost:8080
```

**Option B: Full Scale Test (~118k images, 18GB)**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d ./images
magnitude-ingest --dir ./images/train2017 --host http://localhost:8080 --batch-size 64
```

**Start the Web UI:**
```bash
magnitude-ui
```
Open **http://localhost:3333** to search your images using text queries (e.g., "a red car").

---

## Client Usage

### Python

```python
from magnitude import VectorDBClient, CLIPEmbedder

embedder = CLIPEmbedder()
client = VectorDBClient("http://localhost:8080")

col = client.create_collection("my-images", dimension=512)
vectors = embedder.embed_images(["cat.jpg", "dog.jpg"])
client.insert(col.id, ids=[1, 2], vectors=vectors, metadata=[{"filename": "cat.jpg"}, {"filename": "dog.jpg"}])

results = client.search(col.id, query=embedder.embed_text("a cute cat"), top_k=5)
for r in results:
    print(f"File: {r.metadata['filename']}, Score: {r.score:.4f}")
```

### Go

```go
import "github.com/POTATO-VE1/Magnitude/pkg/client"

c := client.New("http://localhost:8080", "")
col, _ := c.CreateCollection(ctx, "docs", 128, "cosine", "hnsw")
_ = c.Insert(ctx, col.ID, ids, vectors)
results, _ := c.Search(ctx, col.ID, query, 10, 0)
```

---

## Configuration & Production

All settings are managed in `config.yaml`. Data is persisted in `./data`.

To secure for production, edit `config.yaml`:
```yaml
server:
  addr: ":8443"
  certFile: "certs/server.crt"
  keyFile: "certs/server.key"
auth:
  keyHashes:
    - "<SHA-256 hash of your API key>"
```

For distributed clustering, WAL internals, and indexing architecture, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License
MIT
