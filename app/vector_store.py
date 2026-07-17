import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

from app.config import get_settings


def chunk_id(document_uuid: str, version: int, chunk_index: int) -> str:
    return f"{document_uuid}:{version}:{chunk_index}"


def parent_chunk_id(document_uuid: str, version: int, parent_index: int) -> str:
    return f"{document_uuid}:{version}:parent:{parent_index}"


def _quoted(values: Iterable[str]) -> str:
    return ", ".join(json.dumps(value) for value in values)


DENSE_FIELD = "embedding"
SPARSE_FIELD = "sparse_embedding"
CONTENT_FIELD = "content"
BM25_FUNCTION = "content_bm25"
OUTPUT_FIELDS = [
    "document_uuid",
    "version_uuid",
    "version_number",
    "page_number",
    "page_start",
    "page_end",
    "chunk_index",
    "parent_chunk_id",
    "parent_content",
    "parent_index",
    "section_title",
    "section_path",
    "chunking_strategy",
    "chunking_version",
    "source_name",
    "content",
]
PARENT_CHILD_FIELDS = {
    "parent_chunk_id",
    "parent_content",
    "parent_index",
    "section_title",
    "section_path",
    "page_start",
    "page_end",
    "chunking_strategy",
    "chunking_version",
}


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
                self._validate_collection_schema(collection)
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
                FieldSchema("page_start", DataType.INT64),
                FieldSchema("page_end", DataType.INT64),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("parent_chunk_id", DataType.VARCHAR, max_length=512),
                FieldSchema("parent_content", DataType.VARCHAR, max_length=65535),
                FieldSchema("parent_index", DataType.INT64),
                FieldSchema("section_title", DataType.VARCHAR, max_length=2048),
                FieldSchema("section_path", DataType.VARCHAR, max_length=4096),
                FieldSchema("chunking_strategy", DataType.VARCHAR, max_length=64),
                FieldSchema("chunking_version", DataType.VARCHAR, max_length=64),
                FieldSchema("source_name", DataType.VARCHAR, max_length=1024),
                FieldSchema(
                    CONTENT_FIELD,
                    DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                ),
                FieldSchema(
                    DENSE_FIELD,
                    DataType.FLOAT_VECTOR,
                    dim=self.settings.embedding_dimension,
                ),
                FieldSchema(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(
                fields,
                description="Enterprise RAG chunks with dense and BM25 sparse retrieval",
                functions=[
                    Function(
                        name=BM25_FUNCTION,
                        function_type=FunctionType.BM25,
                        input_field_names=[CONTENT_FIELD],
                        output_field_names=[SPARSE_FIELD],
                    )
                ],
            )
            collection = Collection(name, schema=schema, using=self.alias)
            collection.create_index(
                DENSE_FIELD,
                {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 128},
                },
            )
            collection.create_index(
                SPARSE_FIELD,
                {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        "bm25_k1": 1.2,
                        "bm25_b": 0.75,
                    },
                },
            )
            collection.load()
            self._collection = collection
            return collection
        except Exception:
            self._collection = None
            raise

    def _validate_collection_schema(self, collection: Collection) -> None:
        fields = {field.name: field for field in collection.schema.fields}
        missing_parent_child_fields = sorted(PARENT_CHILD_FIELDS - fields.keys())
        if missing_parent_child_fields:
            raise ValueError(
                "Milvus parent-child collection is missing fields: "
                + ", ".join(missing_parent_child_fields)
            )
        if DENSE_FIELD not in fields:
            raise ValueError(f"Milvus collection is missing {DENSE_FIELD} field")
        vector_field = fields[DENSE_FIELD]
        if vector_field.params.get("dim") != self.settings.embedding_dimension:
            raise ValueError("Milvus embedding dimension does not match EMBEDDING_DIMENSION")
        if self.settings.retrieval_mode == "dense":
            return
        if CONTENT_FIELD not in fields:
            raise ValueError(f"Milvus hybrid collection is missing {CONTENT_FIELD} field")
        if not fields[CONTENT_FIELD].params.get("enable_analyzer"):
            raise ValueError("Milvus hybrid collection content field must enable analyzer")
        if SPARSE_FIELD not in fields:
            raise ValueError(f"Milvus hybrid collection is missing {SPARSE_FIELD} field")
        if fields[SPARSE_FIELD].dtype != DataType.SPARSE_FLOAT_VECTOR:
            raise ValueError("Milvus hybrid collection sparse field has invalid type")
        functions = getattr(collection.schema, "functions", []) or []
        has_bm25 = any(
            getattr(function, "type", None) == FunctionType.BM25
            and CONTENT_FIELD in list(getattr(function, "input_field_names", []) or [])
            and SPARSE_FIELD in list(getattr(function, "output_field_names", []) or [])
            for function in functions
        )
        if not has_bm25:
            raise ValueError("Milvus hybrid collection is missing BM25 function")

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
            [chunk.get("page_start") or 0 for chunk in chunks],
            [chunk.get("page_end") or 0 for chunk in chunks],
            [chunk["chunk_index"] for chunk in chunks],
            [chunk["parent_chunk_id"] for chunk in chunks],
            [chunk["parent_content"][:65535] for chunk in chunks],
            [chunk["parent_index"] for chunk in chunks],
            [str(chunk.get("section_title") or "")[:2048] for chunk in chunks],
            [str(chunk.get("section_path") or "")[:4096] for chunk in chunks],
            [chunk["chunking_strategy"][:64] for chunk in chunks],
            [chunk["chunking_version"][:64] for chunk in chunks],
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
        mode = self.settings.retrieval_mode

        dense_hits: List[Dict[str, Any]] = []
        sparse_hits: List[Dict[str, Any]] = []
        if mode in {"dense", "hybrid"}:
            dense_hits = self.search_dense(
                question,
                version_uuids,
                self.settings.retrieval_dense_limit,
            )
        if mode in {"sparse", "hybrid"}:
            sparse_hits = self.search_sparse(
                question,
                version_uuids,
                self.settings.retrieval_sparse_limit,
            )
        if mode == "dense":
            return self._single_path_results(dense_hits, "dense", limit)
        if mode == "sparse":
            return self._single_path_results(sparse_hits, "sparse", limit)
        return self._rrf_fuse(dense_hits, sparse_hits, limit)

    def search_dense(
        self, question: str, version_uuids: Sequence[str], limit: int
    ) -> List[Dict[str, Any]]:
        if not version_uuids:
            return []
        collection = self.ensure_collection()
        expression = f"version_uuid in [{_quoted(version_uuids)}]"
        return self._dense_search(collection, question, expression, limit)

    def search_sparse(
        self, question: str, version_uuids: Sequence[str], limit: int
    ) -> List[Dict[str, Any]]:
        if not version_uuids:
            return []
        collection = self.ensure_collection()
        expression = f"version_uuid in [{_quoted(version_uuids)}]"
        return self._sparse_search(collection, question, expression, limit)

    def _dense_search(
        self,
        collection: Collection,
        question: str,
        expression: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_limit = limit or self.settings.retrieval_dense_limit
        query_vector = self.embeddings.embed_query(question)
        results = collection.search(
            [query_vector],
            DENSE_FIELD,
            {
                "metric_type": "COSINE",
                "params": {
                    "ef": max(32, search_limit * 4)
                },
            },
            limit=search_limit,
            expr=expression,
            output_fields=OUTPUT_FIELDS,
        )
        return self._hits_to_candidates(results[0], "dense_score")

    def _sparse_search(
        self,
        collection: Collection,
        question: str,
        expression: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_limit = limit or self.settings.retrieval_sparse_limit
        results = collection.search(
            [question],
            SPARSE_FIELD,
            {
                "metric_type": "BM25",
                "params": {"drop_ratio_search": 0.2},
            },
            limit=search_limit,
            expr=expression,
            output_fields=OUTPUT_FIELDS,
        )
        return self._hits_to_candidates(results[0], "sparse_score")

    @staticmethod
    def _hits_to_candidates(hits: Sequence[Any], score_field: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for hit in hits:
            score = float(hit.score)
            candidates.append(
                {
                    "score": score,
                    "dense_score": score if score_field == "dense_score" else None,
                    "sparse_score": score if score_field == "sparse_score" else None,
                    "retrieval_sources": [
                        "dense" if score_field == "dense_score" else "sparse"
                    ],
                    "chunk_id": hit.id,
                    **{field: hit.entity.get(field) for field in OUTPUT_FIELDS},
                }
            )
        return candidates

    @staticmethod
    def _single_path_results(
        hits: Sequence[Dict[str, Any]], source: str, limit: int
    ) -> List[Dict[str, Any]]:
        score_field = "dense_score" if source == "dense" else "sparse_score"
        results = []
        for hit in hits:
            item = dict(hit)
            item["score"] = float(item.get(score_field) or 0)
            item["retrieval_sources"] = [source]
            results.append(item)
        return sorted(
            results,
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )[:limit]

    def _rrf_fuse(
        self,
        dense_hits: Sequence[Dict[str, Any]],
        sparse_hits: Sequence[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for source, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
            score_field = "dense_score" if source == "dense" else "sparse_score"
            for rank, hit in enumerate(hits, start=1):
                chunk_key = str(hit.get("chunk_id") or "")
                if not chunk_key:
                    continue
                item = fused.setdefault(
                    chunk_key,
                    {
                        **hit,
                        "score": 0.0,
                        "dense_score": None,
                        "sparse_score": None,
                        "retrieval_sources": [],
                    },
                )
                item["score"] += 1 / (self.settings.retrieval_rrf_k + rank)
                item[score_field] = hit.get(score_field)
                if source not in item["retrieval_sources"]:
                    item["retrieval_sources"].append(source)
        return sorted(
            fused.values(),
            key=lambda item: (
                float(item.get("score") or 0),
                float(item.get("dense_score") or 0),
                float(item.get("sparse_score") or 0),
            ),
            reverse=True,
        )[:limit]

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
            "page_start",
            "page_end",
            "chunk_index",
            "parent_chunk_id",
            "parent_content",
            "parent_index",
            "section_title",
            "section_path",
            "chunking_strategy",
            "chunking_version",
            "source_name",
            "content",
            DENSE_FIELD,
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
