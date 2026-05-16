"""CLI tool for batch-ingesting images into Magnitude."""

import argparse
import os
import sys
from typing import List


def get_image_paths(directory: str) -> List[str]:
    """Recursively find all image paths in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    paths = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in exts:
                paths.append(os.path.join(root, file))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest images into Magnitude VectorDB"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--host", type=str, default="https://localhost:8443", help="Server URL"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument(
        "--tenant", type=str, default="default", help="Tenant name (v2 API)"
    )
    parser.add_argument(
        "--db", type=str, default="images_db", help="Database name (v2 API)"
    )
    parser.add_argument(
        "--collection", type=str, default="clip_images", help="Collection name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding"
    )
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric")
    parser.add_argument("--index-type", type=str, default="hnsw", help="Index type")
    args = parser.parse_args()

    try:
        from magnitude import VectorDBClient, SigLIPEmbedder
    except ImportError:
        print("Error: Install with pip install magnitude-client[all]")
        sys.exit(1)

    print("Initializing SigLIP embedder...")
    embedder = SigLIPEmbedder()
    client = VectorDBClient(args.host, api_key=args.api_key)

    # Setup collection
    print("Setting up collection...")
    try:
        col = client.create_collection(
            args.collection,
            dimension=embedder.dimension,
            metric=args.metric,
            index_type=args.index_type,
        )
        print(f"  Created collection: {col}")
    except Exception:
        # Collection may already exist
        try:
            col = client.get_collection(args.collection)
            print(f"  Using existing collection: {col}")
        except Exception as e:
            print(f"Failed to get/create collection: {e}")
            sys.exit(1)

    # Get images
    image_paths = get_image_paths(args.dir)
    if not image_paths:
        print(f"No images found in {args.dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images. Starting ingestion...")

    # Batch process
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback without tqdm
        def tqdm(iterable, **kwargs):
            return iterable

    vector_id = 1
    inserted = 0
    failed = 0

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Ingesting"):
        batch_paths = image_paths[i : i + args.batch_size]

        embeddings = embedder.embed_images(batch_paths)

        valid_ids = []
        valid_vecs = []
        valid_meta = []

        for path, emb in zip(batch_paths, embeddings):
            if emb is not None:
                valid_ids.append(vector_id)
                valid_vecs.append(emb)
                valid_meta.append(
                    {
                        "filename": os.path.basename(path),
                        "path": os.path.abspath(path),
                    }
                )
                vector_id += 1

        if valid_vecs:
            try:
                client.insert(col.id, valid_ids, valid_vecs, valid_meta)
                inserted += len(valid_vecs)
            except Exception as e:
                failed += len(valid_vecs)
                print(f"  Batch error (skipping): {e}")

    print(f"\nIngestion complete! Inserted: {inserted}, Failed: {failed}")


if __name__ == "__main__":
    main()
