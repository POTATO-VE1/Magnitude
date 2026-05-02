import os
import argparse
from typing import List
from tqdm import tqdm

from embedder import CLIPEmbedder
from vectordb_client import VectorDBClient

def get_image_paths(directory: str) -> List[str]:
    """Recursively find all image paths in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                paths.append(os.path.join(root, file))
    return paths

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest images into VectorDB")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--tenant", type=str, default="default", help="Tenant name")
    parser.add_argument("--db", type=str, default="images_db", help="Database name")
    parser.add_argument("--collection", type=str, default="clip_images", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    args = parser.parse_args()

    print("Initializing CLIP embedder...")
    embedder = CLIPEmbedder()
    client = VectorDBClient()

    # 1. Setup DB
    print("Setting up VectorDB...")
    try:
        tenant_id = client.get_or_create_tenant(args.tenant)
    except Exception as e:
        print(f"Tenant setup failed. ({e})")
        return
        
    try:
        db_id = client.get_or_create_database(tenant_id, args.db)
    except Exception as e:
        print(f"Database setup failed. ({e})")
        return
        
    try:
        # DIMENSION MUST MATCH CLIP OUT (512)
        col_id = client.get_or_create_collection(tenant_id, db_id, args.collection, dimension=512)
    except Exception as e:
        print(f"Collection setup failed. ({e})")
        return

    # 2. Get images
    image_paths = get_image_paths(args.dir)
    print(f"Found {len(image_paths)} images. Starting ingestion...")

    # 3. Batch process
    vector_id = 1
    inserted = 0
    failed = 0
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i:i+args.batch_size]
        
        # Embed
        embeddings = embedder.embed_images(batch_paths)
        
        valid_ids = []
        valid_vecs = []
        valid_meta = []
        
        for path, emb in zip(batch_paths, embeddings):
            if emb is not None:
                valid_ids.append(vector_id)
                valid_vecs.append(emb)
                valid_meta.append({
                    "filename": os.path.basename(path),
                    "path": os.path.abspath(path)
                })
                vector_id += 1
                
        if len(valid_vecs) > 0:
            try:
                client.insert_vectors(tenant_id, db_id, col_id, valid_ids, valid_vecs, valid_meta)
                inserted += len(valid_vecs)
            except Exception as e:
                failed += len(valid_vecs)
                tqdm.write(f"  Batch error (skipping): {e}")

    print(f"\nIngestion complete! Inserted: {inserted}, Failed: {failed}")

if __name__ == "__main__":
    main()