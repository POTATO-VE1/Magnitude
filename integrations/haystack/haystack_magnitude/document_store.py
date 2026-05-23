"""Haystack DocumentStore implementation for Magnitude."""

from typing import Any, Dict, List, Optional

from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.dataclasses import Document
from haystack.document_stores.types import DocumentStore

from magnitude import VectorDBClient


@component
class MagnitudeDocumentStore(DocumentStore):
    """Haystack DocumentStore backed by a Magnitude vector database.

    Usage:
        from haystack_magnitude import MagnitudeDocumentStore

        store = MagnitudeDocumentStore(
            client=VectorDBClient("http://localhost:8080"),
            collection_name="documents",
        )
    """

    def __init__(
        self,
        client: VectorDBClient,
        collection_name: str = "documents",
        tenant_id: str = "default_tenant",
        database_id: str = "default",
        dimension: int = 768,
        metric: str = "cosine",
    ):
        self.client = client
        self.collection_name = collection_name
        self.tenant_id = tenant_id
        self.database_id = database_id
        self.dimension = dimension
        self.metric = metric
        self._collection_id: Optional[str] = None

    def _get_collection_id(self) -> str:
        if self._collection_id is None:
            cols = self.client.list_collections(self.tenant_id, self.database_id)
            for col in cols:
                if col["name"] == self.collection_name:
                    self._collection_id = col["id"]
                    break
            if self._collection_id is None:
                col = self.client.create_collection(
                    self.tenant_id,
                    self.database_id,
                    self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                )
                self._collection_id = col["id"]
        return self._collection_id

    def write_documents(self, documents: List[Document], policy: str = "NONE") -> None:
        if not documents:
            return

        ids = []
        vectors = []
        metadatas = []

        for doc in documents:
            if doc.embedding is None:
                continue
            doc_id = hash(doc.id) & 0xFFFFFFFFFFFFFFFF
            ids.append(doc_id)
            vectors.append(doc.embedding)
            meta = dict(doc.meta or {})
            meta["_content"] = doc.content
            metadatas.append(meta)

        if ids:
            self.client.insert(
                self.tenant_id,
                self.database_id,
                self._get_collection_id(),
                ids=ids,
                vectors=vectors,
                metadata=metadatas,
            )

    def count_documents(self) -> int:
        cols = self.client.list_collections(self.tenant_id, self.database_id)
        for col in cols:
            if col["name"] == self.collection_name:
                return col.get("vector_count", 0)
        return 0

    def delete_documents(self, document_ids: List[str]) -> None:
        if document_ids:
            numeric_ids = [hash(did) & 0xFFFFFFFFFFFFFFFF for did in document_ids]
            self.client.delete_vectors(
                self.tenant_id,
                self.database_id,
                self._get_collection_id(),
                ids=numeric_ids,
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        results = self.client.search(
            self.tenant_id,
            self.database_id,
            self._get_collection_id(),
            query=query_embedding,
            top_k=top_k,
            filter=filters,
        )

        docs = []
        for r in results:
            meta = dict(r.get("metadata", {}))
            content = meta.pop("_content", "")
            docs.append(
                Document(
                    id=str(r["id"]),
                    content=content,
                    meta=meta,
                    score=r.get("score", 0.0),
                )
            )
        return docs
