from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from typing import List, Union

class CLIPEmbedder:
    def __init__(self, model_name: str = 'clip-ViT-B-32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CLIP outputs 512 dimensions for clip-ViT-B-32
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Append None so we maintain parallel arrays, the caller should filter
                images.append(None)
                
        # Filter out bad images
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [img for img in images if img is not None]

        embeddings = self.model.encode(valid_images, convert_to_numpy=True).tolist()

        # Reconstruct full list with None for failed ones
        result = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb
            
        return result

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
