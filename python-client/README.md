# magnitude-client

Python client for the [Magnitude](https://github.com/POTATO-VE1/Magnitude) vector database.

## Installation

```bash
pip install magnitude-client
```

With CLIP embedding support:

```bash
pip install magnitude-client[embed]
```

With CLI tools (ingest + search):

```bash
pip install magnitude-client[all]
```

## Quick Start

```python
from magnitude import VectorDBClient

# Connect
client = VectorDBClient("https://localhost:8443")

# Create a collection
client.create_collection("my-images", dimension=512)

# Insert vectors
client.insert(
    "my-images",
    ids=[1, 2, 3],
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    metadata=[
        {"filename": "cat.jpg"},
        {"filename": "dog.jpg"},
        {"filename": "bird.jpg"},
    ],
)

# Search
results = client.search("my-images", query=[0.1, 0.2, ...], top_k=5)
for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}, File: {r.metadata.get('filename')}")
```

## CLIP Image Search

```python
from magnitude import VectorDBClient, CLIPEmbedder

embedder = CLIPEmbedder()
client = VectorDBClient("https://localhost:8443")

# Embed and insert images
vectors = embedder.embed_images(["cat.jpg", "dog.jpg"])
client.insert("images", ids=[1, 2], vectors=vectors)

# Search by text
query = embedder.embed_text("a cute cat")
results = client.search("images", query=query, top_k=5)
```

## CLI Tools

After installing with `pip install magnitude-client[all]`:

```bash
# Ingest images from a directory
magnitude-ingest --dir ./photos --collection my-images

# Interactive search
magnitude-search --collection my-images
```

## API Reference

### `VectorDBClient(base_url, api_key=None, verify_ssl=False, timeout=30)`

Main client class.

#### Methods

- `create_collection(name, dimension, metric="cosine", index_type="hnsw")` → `Collection`
- `get_collection(name_or_id)` → `Collection`
- `list_collections()` → `List[Collection]`
- `delete_collection(name_or_id)` → `None`
- `insert(collection, ids, vectors, metadata=None)` → `None`
- `search(collection, query, top_k=10, nprobe=0, filter=None)` → `List[SearchResult]`
- `delete_vector(collection, vector_id)` → `None`
- `health()` → `bool`

### `CLIPEmbedder(model_name="clip-ViT-B-32")`

CLIP embedding utilities. Requires `pip install magnitude-client[embed]`.

#### Methods

- `embed_text(text)` → `List[float]`
- `embed_texts(texts)` → `List[List[float]]`
- `embed_images(paths)` → `List[Optional[List[float]]]`
- `dimension` → `int` (property)
