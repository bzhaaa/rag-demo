from typing import Any, Dict, List, Sequence

from app.config import Settings
from app.rag.contracts.models import Candidate
from app.rag.contracts.protocols import KnowledgeRetriever
from app.vector_store import MilvusChunkStore


class MilvusKnowledgeRetriever:
    def __init__(self, store: MilvusChunkStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def retrieve(
        self, query: str, version_uuids: Sequence[str], limit: int
    ) -> Dict[str, List[Candidate]]:
        channels: Dict[str, List[Candidate]] = {}
        if self.settings.retrieval_mode in {"dense", "hybrid"}:
            channels["dense"] = [
                Candidate.from_mapping(item)
                for item in self.store.search_dense(
                    query,
                    version_uuids,
                    self.settings.retrieval_dense_limit,
                )
            ]
        if self.settings.retrieval_mode in {"sparse", "hybrid"}:
            channels["sparse"] = [
                Candidate.from_mapping(item)
                for item in self.store.search_sparse(
                    query,
                    version_uuids,
                    self.settings.retrieval_sparse_limit,
                )
            ]
        return channels


class LegacyRetrieverAdapter:
    def __init__(self, store: Any) -> None:
        self.store = store

    def retrieve(
        self, query: str, version_uuids: Sequence[str], limit: int
    ) -> Dict[str, List[Candidate]]:
        return {
            "fused": [
                Candidate.from_mapping(item)
                for item in self.store.search(query, version_uuids, limit)
            ]
        }


def create_knowledge_retriever(
    store: Any, settings: Settings
) -> KnowledgeRetriever:
    if isinstance(store, MilvusChunkStore) or (
        hasattr(store, "search_dense") and hasattr(store, "search_sparse")
    ):
        return MilvusKnowledgeRetriever(store, settings)
    return LegacyRetrieverAdapter(store)
