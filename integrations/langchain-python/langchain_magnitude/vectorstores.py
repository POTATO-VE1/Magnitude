"""LangChain VectorStore implementation for Magnitude."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from magnitude import VectorDBClient


class MagnitudeVectorStore(VectorStore):
    """LangChain VectorStore backed by a Magnitude vector database.

    Example:
        from langchain_magnitude import MagnitudeVectorStore
        from langchain_openai import OpenAIEmbeddings

        store = MagnitudeVectorStore(
            client=VectorDBClient("http://localhost:8080"),
            collection_name="my-docs",
            embedding_function=OpenAIEmbeddings(),
        )
        store.add_texts(["hello world"], metadatas=[{"source": "test"}])
        results = store.similarity_search("hello", k=5)
    """

    def __init__(
        self,
        client: VectorDBClient,
        collection_name: str,
        embedding_function: Embeddings,
        tenant_id: str = "default_tenant",
        database_id: str = "default",
        collection_id: Optional[str] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.tenant_id = tenant_id
        self.database_id = database_id
        self._collection_id = collection_id

    @property
    def collection_id(self) -> str:
        if self._collection_id is None:
            cols = self.client.list_collections(self.tenant_id, self.database_id)
            for col in cols:
                if col["name"] == self.collection_name:
                    self._collection_id = col["id"]
                    break
            if self._collection_id is None:
                dim = len(self.embedding_function.embed_query("test"))
                col = self.client.create_collection(
                    self.tenant_id,
                    self.database_id,
                    self.collection_name,
                    dimension=dim,
                    metric="cosine",
                )
                self._collection_id = col["id"]
        return self._collection_id

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = self.embedding_function.embed_documents(texts_list)
        ids = [str(uuid.uuid4()) for _ in texts_list]

        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        for i, meta in enumerate(metadatas):
            meta["text"] = texts_list[i]

        self.client.insert(
            self.tenant_id,
            self.database_id,
            self.collection_id,
            ids=[hash(id) & 0xFFFFFFFFFFFFFFFF for id in ids],
            vectors=embeddings,
            metadata=metadatas,
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function.embed_query(query)

        results = self.client.search(
            self.tenant_id,
            self.database_id,
            self.collection_id,
            query=embedding,
            top_k=k,
            filter=filter,
        )

        docs = []
        for r in results:
            meta = r.get("metadata", {})
            text = meta.pop("text", "")
            docs.append(
                (
                    Document(page_content=text, metadata=meta),
                    r.get("distance", 0.0),
                )
            )
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if ids:
            numeric_ids = [hash(id) & 0xFFFFFFFFFFFFFFFF for id in ids]
            self.client.delete_vectors(
                self.tenant_id,
                self.database_id,
                self.collection_id,
                ids=numeric_ids,
            )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[VectorDBClient] = None,
        collection_name: str = "langchain",
        tenant_id: str = "default_tenant",
        database_id: str = "default",
        **kwargs: Any,
    ) -> MagnitudeVectorStore:
        if client is None:
            client = VectorDBClient("http://localhost:8080")

        store = cls(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
            tenant_id=tenant_id,
            database_id=database_id,
        )
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store
