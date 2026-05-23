"""Magnitude VectorDB Python Client.

A Python client for the Magnitude vector database.

Usage:
    from magnitude import VectorDBClient

    client = VectorDBClient("http://localhost:8080")
    client.create_collection("my-col", dimension=512)
    client.insert("my-col", ids=[1, 2], vectors=[[0.1, ...], [0.2, ...]])
    results = client.search("my-col", query=[0.1, ...], top_k=10)

Optional — SigLIP embedder (requires pip install magnitude-client[embed]):
    from magnitude import SigLIPEmbedder
"""

from magnitude.client import VectorDBClient
from magnitude.exceptions import (
    MagnitudeError,
    MagnitudeConnectionError,
    CollectionNotFoundError,
    AuthenticationError,
)

__version__ = "0.1.0"
__all__ = [
    "VectorDBClient",
    "MagnitudeError",
    "MagnitudeConnectionError",
    "CollectionNotFoundError",
    "AuthenticationError",
]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "SigLIPEmbedder":
        try:
            from magnitude.embedder import SigLIPEmbedder

            return SigLIPEmbedder
        except ImportError:
            raise ImportError(
                "SigLIPEmbedder requires extra dependencies. "
                "Install with: pip install magnitude-client[embed]"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
