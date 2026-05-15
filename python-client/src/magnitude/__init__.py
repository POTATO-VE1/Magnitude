"""Magnitude VectorDB Python Client.

A clean Python client for the Magnitude vector database, featuring
CLIP-powered semantic image search.

Usage:
    from magnitude import VectorDBClient

    client = VectorDBClient("http://localhost:8080")
    client.create_collection("my-col", dimension=512)
    client.insert("my-col", ids=[1, 2], vectors=[[0.1, ...], [0.2, ...]])
    results = client.search("my-col", query=[0.1, ...], top_k=10)
"""

from magnitude.client import VectorDBClient
from magnitude.embedder import CLIPEmbedder
from magnitude.exceptions import (
    MagnitudeError,
    MagnitudeConnectionError,
    CollectionNotFoundError,
    AuthenticationError,
)

__version__ = "0.1.0"
__all__ = [
    "VectorDBClient",
    "CLIPEmbedder",
    "MagnitudeError",
    "MagnitudeConnectionError",
    "CollectionNotFoundError",
    "AuthenticationError",
]
