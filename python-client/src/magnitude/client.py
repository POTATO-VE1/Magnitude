"""Magnitude VectorDB HTTP client.

Provides a clean, Pythonic API for interacting with the Magnitude vector
database server. Handles connection pooling, authentication, and error
translation automatically.
"""

import json
import urllib3
from typing import Any, Dict, List, Optional, Union

import requests

from magnitude.exceptions import (
    AuthenticationError,
    CollectionNotFoundError,
    MagnitudeConnectionError,
    MagnitudeError,
    ServerError,
)


class Collection:
    """Represents a collection in the database."""

    def __init__(
        self,
        id: str,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        vector_count: int = 0,
    ):
        self.id = id
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.vector_count = vector_count

    def __repr__(self) -> str:
        return (
            f"Collection(id={self.id!r}, name={self.name!r}, "
            f"dim={self.dimension}, metric={self.metric!r}, "
            f"vectors={self.vector_count})"
        )


class SearchResult:
    """A single search result."""

    def __init__(
        self,
        id: int,
        distance: float,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.distance = distance
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, score={self.score:.4f})"


class VectorDBClient:
    """Client for the Magnitude VectorDB HTTP API.

    Args:
        base_url: Server URL (e.g., "http://localhost:8080").
        api_key: Optional API key for authentication.
        verify_ssl: Whether to verify TLS certificates. Set False for self-signed certs.
        timeout: Request timeout in seconds.

    Example:
        >>> client = VectorDBClient("http://localhost:8080")
        >>> col = client.create_collection("images", dimension=512)
        >>> client.insert("images", ids=[1], vectors=[[0.1, ...]])
        >>> results = client.search("images", query=[0.1, ...], top_k=5)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify_ssl
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ── Collection operations ────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimensionality.
            metric: Distance metric ("l2", "cosine", "dot", "manhattan").
            index_type: Index type ("flat", "ivf", "hnsw", "spann").

        Returns:
            The created Collection.
        """
        data = self._post(
            "/v1/collections",
            {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "index_type": index_type,
            },
        )
        return Collection(
            id=data["id"],
            name=data["name"],
            dimension=data["dimension"],
            metric=data.get("metric", metric),
            index_type=data.get("index_type", index_type),
            vector_count=data.get("vector_count", 0),
        )

    def get_collection(self, name_or_id: str) -> Collection:
        """Get a collection by name or ID.

        Args:
            name_or_id: Collection name or UUID.

        Returns:
            The Collection.

        Raises:
            CollectionNotFoundError: If not found.
        """
        import uuid
        is_uuid = False
        try:
            uuid.UUID(name_or_id)
            is_uuid = True
        except ValueError:
            pass

        if not is_uuid:
            # Look up by name
            cols = self.list_collections()
            for c in cols:
                if c.name == name_or_id:
                    return c
            raise CollectionNotFoundError(f"Collection {name_or_id!r} not found")

        try:
            data = self._get(f"/v1/collections/{name_or_id}")
            return Collection(
                id=data["id"],
                name=data["name"],
                dimension=data["dimension"],
                metric=data.get("metric", "cosine"),
                index_type=data.get("index_type", "hnsw"),
                vector_count=data.get("vector_count", 0),
            )
        except ServerError as e:
            if e.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection {name_or_id!r} not found"
                ) from e
            raise

    def list_collections(self) -> List[Collection]:
        """List all collections.

        Returns:
            List of Collection objects.
        """
        data = self._get("/v1/collections")
        if not data:
            return []
        return [
            Collection(
                id=c["id"],
                name=c["name"],
                dimension=c["dimension"],
                metric=c.get("metric", "cosine"),
                index_type=c.get("index_type", "hnsw"),
                vector_count=c.get("vector_count", 0),
            )
            for c in data
        ]

    def delete_collection(self, name_or_id: str) -> None:
        """Delete a collection by name or ID.

        Args:
            name_or_id: Collection name or UUID.
        """
        self._delete(f"/v1/collections/{name_or_id}")

    # ── Vector operations ────────────────────────────────────────────────

    def insert(
        self,
        collection: str,
        ids: List[int],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert vectors into a collection.

        Args:
            collection: Collection name or ID.
            ids: Unique vector IDs.
            vectors: List of float vectors (must match collection dimension).
            metadata: Optional metadata dicts (one per vector).
        """
        payload: Dict[str, Any] = {"ids": ids, "vectors": vectors}
        if metadata:
            payload["metadata"] = metadata
        self._post(f"/v1/collections/{collection}/vectors", payload)

    def search(
        self,
        collection: str,
        query: List[float],
        top_k: int = 10,
        nprobe: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors.

        Args:
            collection: Collection name or ID.
            query: Query vector.
            top_k: Number of results to return.
            nprobe: Number of clusters to probe (IVF only, 0 = default).
            filter: Optional metadata filter.

        Returns:
            List of SearchResult sorted by relevance.
        """
        payload: Dict[str, Any] = {"query": query, "k": top_k}
        if nprobe > 0:
            payload["nprobe"] = nprobe
        if filter:
            payload["filter"] = filter
        data = self._post(f"/v1/collections/{collection}/search", payload)
        if not data:
            return []
        return [
            SearchResult(
                id=r.get("ID", r.get("id", 0)),
                distance=r.get("Distance", r.get("distance", 0.0)),
                score=r.get("Score", r.get("score", 0.0)),
                metadata=r.get("metadata"),
            )
            for r in data
        ]

    def delete_vector(self, collection: str, vector_id: int) -> None:
        """Delete a vector by ID.

        Args:
            collection: Collection name or ID.
            vector_id: Vector ID to delete.
        """
        self._delete(f"/v1/collections/{collection}/vectors/{vector_id}")

    # ── Health ───────────────────────────────────────────────────────────

    def health(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if healthy.
        """
        try:
            self._get("/v1/health")
            return True
        except MagnitudeError:
            return False

    # ── Internal helpers ─────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, payload: Any) -> Any:
        try:
            resp = self.session.post(
                self._url(path),
                json=payload,
                timeout=self.timeout,
            )
        except requests.ConnectionError as e:
            raise MagnitudeConnectionError(
                f"Failed to connect to {self.base_url}"
            ) from e

        return self._handle_response(resp)

    def _get(self, path: str) -> Any:
        try:
            resp = self.session.get(self._url(path), timeout=self.timeout)
        except requests.ConnectionError as e:
            raise MagnitudeConnectionError(
                f"Failed to connect to {self.base_url}"
            ) from e

        return self._handle_response(resp)

    def _delete(self, path: str) -> Any:
        try:
            resp = self.session.delete(self._url(path), timeout=self.timeout)
        except requests.ConnectionError as e:
            raise MagnitudeConnectionError(
                f"Failed to connect to {self.base_url}"
            ) from e

        return self._handle_response(resp)

    def _handle_response(self, resp: requests.Response) -> Any:
        if resp.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")
        if resp.status_code == 404:
            raise CollectionNotFoundError("Resource not found")

        try:
            body = resp.json()
        except json.JSONDecodeError:
            if resp.status_code >= 400:
                raise ServerError(resp.status_code, resp.text)
            return None

        if resp.status_code >= 400:
            error_msg = body.get("error", resp.text)
            raise ServerError(resp.status_code, error_msg)

        data = body.get("data")
        return data
