import time
from typing import Any, Dict, List

from app.config import Settings
from app.rag.contracts.models import Candidate
from app.rag.contracts.protocols import CandidateFusion, KnowledgeRetriever
from app.rag.contracts.state import RAGState
from app.rag.utils import timing_with


class KnowledgeRetrievalModule:
    def __init__(
        self,
        retriever: KnowledgeRetriever,
        fusion: CandidateFusion,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.fusion = fusion
        self.settings = settings

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates: List[Candidate] = []
        for query in state.get("queries", []):
            channels = self.retriever.retrieve(
                query,
                state.get("version_uuids", []),
                self.settings.retrieval_candidate_count,
            )
            candidates.extend(
                self.fusion.fuse(
                    channels,
                    self.settings.retrieval_candidate_count,
                )
            )
        merged = self.fusion.merge_queries(candidates)
        mappings = [candidate.to_mapping() for candidate in merged]
        return {
            **state,
            "candidates": mappings,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "retrieval": {
                    "mode": self.settings.retrieval_mode,
                    "rrf_k": self.settings.retrieval_rrf_k,
                    "dense_limit": self.settings.retrieval_dense_limit,
                    "sparse_limit": self.settings.retrieval_sparse_limit,
                    "dense_hit_count": self._source_count(merged, "dense"),
                    "sparse_hit_count": self._source_count(merged, "sparse"),
                    "candidate_count": len(merged),
                    "fused_candidate_count": len(merged),
                    "candidates": [self.candidate_summary(item) for item in merged],
                },
            },
            "timings": timing_with(
                state.get("timings", {}),
                "retrieval",
                time.perf_counter() - started,
            ),
        }

    @staticmethod
    def candidate_summary(candidate: Candidate) -> Dict[str, Any]:
        return {
            "chunk_id": candidate.chunk_id,
            "document_uuid": candidate.document_uuid,
            "version_uuid": candidate.version_uuid,
            "source_type": candidate.source_type,
            "score": candidate.score,
            "dense_score": candidate.dense_score,
            "sparse_score": candidate.sparse_score,
            "retrieval_sources": candidate.retrieval_sources,
            "rerank_score": candidate.rerank_score,
            "parent_chunk_id": candidate.parent_chunk_id,
            "section_path": candidate.section_path,
            "page_start": candidate.page_start,
            "page_end": candidate.page_end,
            "chunking_strategy": candidate.chunking_strategy,
            "chunking_version": candidate.chunking_version,
        }

    @staticmethod
    def _source_count(candidates: List[Candidate], source: str) -> int:
        return sum(1 for item in candidates if source in item.retrieval_sources)
