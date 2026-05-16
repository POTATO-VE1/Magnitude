import os
import time
import mimetypes
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Base directory for allowed image paths. Override with MAGNITUDE_IMAGE_ROOT env var.
IMAGE_ROOT = os.path.realpath(
    os.environ.get("MAGNITUDE_IMAGE_ROOT", os.path.expanduser("~"))
)

# We assume the user has the standalone client or the magnitude.client available.
# To match the ingest/search scripts, we use the v2 API.
# Let's import the standalone one from the root, or the new one if they fixed it.
# Actually, the ingest script in src/magnitude/cli/ingest.py uses:
# from magnitude.client import VectorDBClient
# wait, let me check what magnitude-search uses.
import sys

# Make sure we can import the standalone client if needed, or use the package one.
try:
    from magnitude.client import VectorDBClient
except ImportError:
    print(
        "Error: Could not import VectorDBClient. Make sure you are running from the python-client root."
    )
    sys.exit(1)

from magnitude.embedder import SigLIPEmbedder

app = FastAPI(title="Magnitude")

# Global state
embedder = None
client = None
col_id = None

from typing import Optional


class SearchRequest(BaseModel):
    query: str
    collection_id: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global embedder, client, col_id
    print("Loading CLIP model... (this may take a few seconds)")
    t0 = time.time()
    embedder = SigLIPEmbedder()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    client = VectorDBClient("http://localhost:8080")
    try:
        collections = client.list_collections()
        if collections:
            # Try to find clip_images, otherwise just pick the first one
            default_col = next((c for c in collections if c.name == "clip_images"), collections[0])
            col_id = default_col.id
            print(f"Connected to VectorDB. Default collection: {default_col.name} ({col_id})")
        else:
            print("Connected to VectorDB. No collections found yet.")
    except Exception as e:
        print(f"Failed to connect to VectorDB: {e}")


@app.get("/api/collections")
def get_collections_api():
    try:
        collections = client.list_collections()
        # Convert Collection objects to dict
        col_dicts = [
            {"id": c.id, "name": c.name, "vector_count": c.vector_count}
            for c in collections
        ]
        return {"collections": col_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
def search_api(req: SearchRequest):
    if not req.query.strip():
        return {"results": [], "metrics": {}}

    try:
        # Embed
        t0 = time.time()
        query_emb = embedder.embed_text(req.query.strip())
        embed_ms = (time.time() - t0) * 1000

        # Use provided collection or fallback to default
        target_col = req.collection_id if req.collection_id else col_id

        # Search
        t1 = time.time()
        results = client.search(target_col, query_emb, top_k=20)
        search_ms = (time.time() - t1) * 1000

        formatted = []
        for r in results or []:
            score = r.score
            vid = r.id
            meta = r.metadata
            path = meta.get("path", "")
            filename = meta.get(
                "filename", os.path.basename(path) if path else "unknown"
            )

            formatted.append(
                {
                    "id": vid,
                    "score": score,
                    "filename": filename,
                    "path": path,
                }
            )

        return {
            "results": formatted,
            "metrics": {
                "embed_ms": embed_ms,
                "search_ms": search_ms,
                "total_ms": embed_ms + search_ms,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image")
def get_image(path: str):
    """Serve local images requested by the frontend."""
    if not path:
        raise HTTPException(status_code=400, detail="Missing path parameter")

    # Resolve to absolute, canonical path (follows symlinks, removes ..)
    abs_path = os.path.realpath(path)

    # Path traversal guard: must be under the allowed image root
    if not abs_path.startswith(IMAGE_ROOT + os.sep) and abs_path != IMAGE_ROOT:
        raise HTTPException(
            status_code=403, detail="Access denied: path outside allowed directory"
        )

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Only serve actual image files
    mime, _ = mimetypes.guess_type(abs_path)
    if not mime or not mime.startswith("image/"):
        raise HTTPException(status_code=403, detail="Not an image file")

    return FileResponse(abs_path)


# Serve static files (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def main():
    print("Starting Magnitude UI Server on http://localhost:3333")
    uvicorn.run("magnitude.cli.ui:app", host="127.0.0.1", port=3333, reload=True)


if __name__ == "__main__":
    main()
