"""SigLIP embedding utilities for text and image encoding.

Uses Google's SigLIP model via the native ``transformers`` library to
produce aligned text and image embeddings in a shared 768-d vector space.

Requires the ``embed`` optional dependency group::

    pip install magnitude-client[embed]
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from magnitude.exceptions import MagnitudeError


class SigLIPEmbedder:
    """SigLIP-based embedder for text and images.

    Converts text queries and images into dense float vectors that live in the
    **same** semantic space, so cosine similarity between a text vector and an
    image vector is meaningful.

    Args:
        model_name: HuggingFace model ID.  Default ``google/siglip-base-patch16-224``.

    Requires::

        pip install magnitude-client[embed]

    Example::

        >>> embedder = SigLIPEmbedder()
        >>> text_vec = embedder.embed_text("a dog playing")
        >>> image_vecs = embedder.embed_images(["photo1.jpg", "photo2.jpg"])
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        try:
            from transformers import SiglipModel, AutoProcessor
            import torch
        except ImportError as e:
            raise MagnitudeError(
                "SigLIP dependencies not installed. "
                "Install with: pip install magnitude-client[embed]"
            ) from e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (lazy, computed on first use)."""
        if self._dimension is None:
            vec = self.embed_text("test")
            self._dimension = len(vec)
        return self._dimension

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Encode a single text string into a normalised float vector.

        Args:
            text: Input text.

        Returns:
            Float vector of length ``dimension``.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple text strings into normalised float vectors.

        Args:
            texts: List of input texts.

        Returns:
            List of float vectors.
        """
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(self.device)

        with self._torch.no_grad():
            text_embeds = self.model.text_model(
                **{k: v for k, v in inputs.items()}
            ).pooler_output

        # The SigLIP text model already pools, but we must still L2-normalize
        # so cosine similarity works correctly against stored image vectors.
        embeds_np = text_embeds.cpu().numpy()
        norms = np.linalg.norm(embeds_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeds_np = embeds_np / norms
        return embeds_np.tolist()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def embed_images(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        """Encode images into normalised float vectors.

        Args:
            image_paths: List of file paths to images.

        Returns:
            List of float vectors (``None`` for images that failed to load).
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

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            img_embeds = self.model.vision_model(**inputs).pooler_output

        embeds_np = img_embeds.cpu().numpy()
        norms = np.linalg.norm(embeds_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeds_np = embeds_np / norms

        result: List[Optional[List[float]]] = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeds_np.tolist()):
            result[idx] = emb

        return result
