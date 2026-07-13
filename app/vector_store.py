import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.config import get_settings


def chunk_id(document_uuid: str, version: int, chunk_index: int) -> str:
    return f"{document_uuid}:{version}:{chunk_index}"


def _quoted(values: Iterable[str]) -> str:
    return ", ".join(json.dumps(value) for value in values)


class MilvusChunkStore:
    alias = "enterprise_rag"

    def __init__(self, embeddings: Optional[OpenAIEmbeddings] = None) -> None:
        self.settings = get_settings()
        self._collection: Optional[Collection] = None
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_base_url,
            check_embedding_ctx_length=False,
            chunk_size=10,
            request_timeout=self.settings.model_timeout_seconds,
            max_retries=self.settings.model_max_retries,
        )

    def connect(self) -> None:
        connections.connect(
            alias=self.alias,
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

    def ensure_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection
        try:
            self.connect()
            name = self.settings.milvus_collection
            if utility.has_collection(name, using=self.alias):
                collection = Collection(name, using=self.alias)
                vector_field = next(
                    field
                    for field in collection.schema.fields
                    if field.name == "embedding"
                )
                if vector_field.params.get("dim") != self.settings.embedding_dimension:
                    raise ValueError(
                        "Milvus embedding dimension does not match EMBEDDING_DIMENSION"
                    )
                collection.load()
                self._collection = collection
                return collection

            fields = [
                FieldSchema("chunk_id", DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema("document_uuid", DataType.VARCHAR, max_length=36),
                FieldSchema("version_uuid", DataType.VARCHAR, max_length=36),
                FieldSchema("version_number", DataType.INT64),
                FieldSchema("department_uuid", DataType.VARCHAR, max_length=36),
                FieldSchema("visibility", DataType.VARCHAR, max_length=20),
                FieldSchema("page_number", DataType.INT64),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("source_name", DataType.VARCHAR, max_length=1024),
                FieldSchema("content", DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    "embedding",
                    DataType.FLOAT_VECTOR,
                    dim=self.settings.embedding_dimension,
                ),
            ]
            schema = CollectionSchema(fields, description="Enterprise RAG chunks")
            collection = Collection(name, schema=schema, using=self.alias)
            collection.create_index(
                "embedding",
                {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 128},
                },
            )
            collection.load()
            self._collection = collection
            return collection
        except Exception:
            self._collection = None
            raise

    def insert_chunks(self, chunks: Sequence[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        collection = self.ensure_collection()
        texts = [chunk["content"] for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        entities = [
            [chunk["chunk_id"] for chunk in chunks],
            [chunk["document_uuid"] for chunk in chunks],
            [chunk["version_uuid"] for chunk in chunks],
            [chunk["version_number"] for chunk in chunks],
            [chunk["department_uuid"] for chunk in chunks],
            [chunk["visibility"] for chunk in chunks],
            [chunk.get("page_number") or 0 for chunk in chunks],
            [chunk["chunk_index"] for chunk in chunks],
            [chunk["source_name"][:1024] for chunk in chunks],
            [chunk["content"][:65535] for chunk in chunks],
            vectors,
        ]
        collection.insert(entities)
        collection.flush()
        return len(chunks)

    def search(
        self, question: str, version_uuids: Sequence[str], limit: int
    ) -> List[Dict[str, Any]]:
        if not version_uuids:
            return []
        collection = self.ensure_collection()
        query_vector = self.embeddings.embed_query(question)
        expression = f"version_uuid in [{_quoted(version_uuids)}]"
        results = collection.search(
            [query_vector],
            "embedding",
            {
                "metric_type": "COSINE",
                "params": {"ef": max(32, limit * 4)},
            },
            limit=limit,
            expr=expression,
            output_fields=[
                "document_uuid",
                "version_uuid",
                "version_number",
                "page_number",
                "chunk_index",
                "source_name",
                "content",
            ],
        )
        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            hits.append(
                {
                    "score": float(hit.score),
                    "chunk_id": hit.id,
                    **{
                        field: hit.entity.get(field)
                        for field in (
                            "document_uuid",
                            "version_uuid",
                            "version_number",
                            "page_number",
                            "chunk_index",
                            "source_name",
                            "content",
                        )
                    },
                }
            )
        return hits

    def delete_version(self, version_uuid: str) -> None:
        collection = self.ensure_collection()
        collection.delete(f"version_uuid == {json.dumps(version_uuid)}")
        collection.flush()

    def update_document_metadata(
        self, document_uuid: str, department_uuid: str, visibility: str
    ) -> None:
        collection = self.ensure_collection()
        fields = [
            "chunk_id",
            "document_uuid",
            "version_uuid",
            "version_number",
            "department_uuid",
            "visibility",
            "page_number",
            "chunk_index",
            "source_name",
            "content",
            "embedding",
        ]
        iterator = collection.query_iterator(
            batch_size=500,
            expr=f"document_uuid == {json.dumps(document_uuid)}",
            output_fields=fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for entity in batch:
                    entity["department_uuid"] = department_uuid
                    entity["visibility"] = visibility
                collection.upsert(batch)
            collection.flush()
        finally:
            iterator.close()

    def ready(self) -> bool:
        self.ensure_collection()
        return True
