from typing import Dict, List, Mapping, Sequence

from app.rag.contracts.models import Candidate


class RRFFusionModule:
    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = max(1, rrf_k)

    def fuse(
        self,
        channels: Mapping[str, Sequence[Candidate]],
        limit: int,
    ) -> List[Candidate]:
        if "fused" in channels:
            return self._sort(channels["fused"])[:limit]
        populated = [list(items) for items in channels.values() if items]
        if len(channels) <= 1:
            return self._sort(populated[0] if populated else [])[:limit]

        fused: Dict[str, Candidate] = {}
        for source in ("dense", "sparse"):
            score_field = "dense_score" if source == "dense" else "sparse_score"
            for rank, candidate in enumerate(channels.get(source, []), start=1):
                if not candidate.chunk_id:
                    continue
                existing = fused.get(candidate.chunk_id)
                if existing is None:
                    existing = Candidate.from_mapping(candidate.to_mapping())
                    existing.score = 0.0
                    existing.dense_score = None
                    existing.sparse_score = None
                    existing.retrieval_sources = []
                    fused[candidate.chunk_id] = existing
                existing.score += 1 / (self.rrf_k + rank)
                setattr(existing, score_field, getattr(candidate, score_field))
                if source not in existing.retrieval_sources:
                    existing.retrieval_sources.append(source)
        return self._sort(list(fused.values()))[:limit]

    def merge_queries(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        merged: Dict[str, Candidate] = {}
        for candidate in candidates:
            if not candidate.chunk_id:
                continue
            existing = merged.get(candidate.chunk_id)
            if existing is None or candidate.score > existing.score:
                merged[candidate.chunk_id] = candidate
        return self._sort(list(merged.values()))

    @staticmethod
    def _sort(candidates: Sequence[Candidate]) -> List[Candidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.score,
                item.dense_score or 0,
                item.sparse_score or 0,
            ),
            reverse=True,
        )
