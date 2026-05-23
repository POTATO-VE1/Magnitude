"""LlamaIndex VectorStore implementation for Magnitude."""

from typing import Any, List, Optional

from llama_index.core.vectorstores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from pydantic import Field

from magnitude import VectorDBClient


class MagnitudeVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by a Magnitude database.

    Example:
        from llama_index_vector_stores_magnitude import MagnitudeVectorStore

        store = MagnitudeVectorStore(
            client=VectorDBClient("http://localhost:8080"),
            collection_name="my-docs",
        )
    """

    stores_text: bool = True
    flat_metadata: bool = False

    client: Any = Field(description="Magnitude VectorDBClient")
    collection_name: str = Field(description="Collection name")
    tenant_id: str = Field(default="default_tenant")
    database_id: str = Field(default="default")
    _collection_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def collection_id(self) -> str:
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
                    dimension=768,
                    metric="cosine",
                )
                self._collection_id = col["id"]
        return self._collection_id

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        if not nodes:
            return []

        ids = []
        vectors = []
        metadatas = []

        for node in nodes:
            node_id = node.node_id or str(hash(node.get_content()) & 0xFFFFFFFFFFFFFFFF)
            ids.append(
                int(node_id, 16)
                if node_id.startswith("0x")
                else hash(node_id) & 0xFFFFFFFFFFFFFFFF
            )
            vectors.append(node.get_embedding())
            meta = dict(node.metadata or {})
            meta["_content"] = node.get_content()
            meta["_node_type"] = type(node).__name__
            metadatas.append(meta)

        self.client.insert(
            self.tenant_id,
            self.database_id,
            self.collection_id,
            ids=ids,
            vectors=vectors,
            metadata=metadatas,
        )
        return [str(id) for id in ids]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        numeric_id = hash(ref_doc_id) & 0xFFFFFFFFFFFFFFFF
        self.client.delete_vectors(
            self.tenant_id,
            self.database_id,
            self.collection_id,
            ids=[numeric_id],
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("query_embedding is required")

        results = self.client.search(
            self.tenant_id,
            self.database_id,
            self.collection_id,
            query=query.query_embedding,
            top_k=query.similarity_top_k,
            filter=query.filters if query.filters else None,
        )

        nodes = []
        similarities = []
        ids = []

        for r in results:
            meta = dict(r.get("metadata", {}))
            content = meta.pop("_content", "")
            node_type = meta.pop("_node_type", "TextNode")

            node = TextNode(
                text=content,
                id_=str(r["id"]),
                metadata=meta,
            )
            nodes.append(node)
            similarities.append(r.get("score", 0.0))
            ids.append(str(r["id"]))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
