"""CLIP embedding utilities for text and image encoding.

Requires the `embed` optional dependency group:
    pip install magnitude-client[embed]
"""

from pathlib import Path
from typing import List, Optional, Union

from magnitude.exceptions import MagnitudeError


class CLIPEmbedder:
    """CLIP-based embedder for text and images.

    Converts text queries and images into dense float vectors that can be
    stored in and searched against a Magnitude vector collection.

    Args:
        model_name: Sentence-Transformers model name. Default: "clip-ViT-B-32".

    Requires:
        pip install magnitude-client[embed]

    Example:
        >>> embedder = CLIPEmbedder()
        >>> text_vec = embedder.embed_text("a dog playing")
        >>> image_vecs = embedder.embed_images(["photo1.jpg", "photo2.jpg"])
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise MagnitudeError(
                "CLIP dependencies not installed. "
                "Install with: pip install magnitude-client[embed]"
            ) from e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (lazy, computed on first use)."""
        if self._dimension is None:
            vec = self.model.encode("test", convert_to_numpy=True)
            self._dimension = len(vec)
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Encode a text string into a float vector.

        Args:
            text: Input text.

        Returns:
            Float vector of length `dimension`.
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple text strings into float vectors.

        Args:
            texts: List of input texts.

        Returns:
            List of float vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_images(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        """Encode images into float vectors.

        Args:
            image_paths: List of file paths to images.

        Returns:
            List of float vectors (None for images that failed to load).
        """
        from PIL import Image

        images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception:
                continue

        if not images:
            return [None] * len(image_paths)

        embeddings = self.model.encode(images, convert_to_numpy=True).tolist()

        result = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb

        return result
