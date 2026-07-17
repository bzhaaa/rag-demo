import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests  # type: ignore[import-untyped,unused-ignore]

from app.config import Settings
from app.rag.types import CandidateReranker


class RerankerError(RuntimeError):
    pass


class DefaultCandidateReranker:
    def __init__(
        self,
        min_score: Optional[float] = None,
        max_chunks_per_document: int = 3,
    ) -> None:
        self.min_score = min_score
        self.max_chunks_per_document = max(1, max_chunks_per_document)

    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            content = str(candidate.get("content") or "")
            normalized = " ".join(content.split())
            if not normalized:
                continue
            score = float(candidate.get("score") or 0)
            if self.min_score is not None and score < self.min_score:
                continue
            existing = deduped.get(normalized)
            if existing is None or score > float(existing.get("score") or 0):
                deduped[normalized] = candidate

        sorted_candidates = sorted(
            deduped.values(),
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )
        per_document: Dict[str, int] = {}
        result: List[Dict[str, Any]] = []
        for candidate in sorted_candidates:
            document_key = str(
                candidate.get("document_uuid")
                or candidate.get("url")
                or candidate.get("source_name")
                or candidate.get("chunk_id")
                or ""
            )
            count = per_document.get(document_key, 0)
            if count >= self.max_chunks_per_document:
                continue
            per_document[document_key] = count + 1
            result.append(candidate)
        return result


class IdentityCandidateReranker:
    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return list(candidates)


class ExternalCandidateReranker:
    def __init__(
        self,
        endpoint: str,
        model: str = "",
        api_key: str = "",
        timeout_seconds: int = 45,
        post: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.post = post or requests.post

    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.endpoint or not candidates:
            return []
        documents = [str(candidate.get("content") or "") for candidate in candidates]
        payload: Dict[str, Any] = {
            "query": question,
            "documents": documents,
        }
        if self.model:
            payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            scores = self._parse_scores(response.json(), len(candidates))
        except Exception as exc:
            raise RerankerError(str(exc)) from exc
        ranked = []
        for index, score in scores:
            candidate = dict(candidates[index])
            candidate["rerank_score"] = score
            ranked.append(candidate)
        return ranked

    @staticmethod
    def _parse_scores(data: Any, candidate_count: int) -> List[tuple[int, float]]:
        raw_items = data
        if isinstance(data, dict):
            if "results" in data:
                raw_items = data["results"]
            elif "data" in data:
                raw_items = data["data"]
            else:
                raise ValueError("reranker response must contain results or data")
        if not isinstance(raw_items, list):
            raise ValueError("reranker results or data must be a list")
        scores: List[tuple[int, float]] = []
        for position, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", position))
            if index < 0 or index >= candidate_count:
                raise ValueError("reranker index out of range")
            raw_score = item.get("relevance_score", item.get("score", 0))
            if raw_score is None:
                raise ValueError("relevance_score must be numeric")
            score = float(raw_score)
            if not math.isfinite(score) or score < 0 or score > 1:
                raise ValueError("relevance_score must be between 0 and 1")
            if any(existing_index == index for existing_index, _ in scores):
                raise ValueError("duplicate reranker index")
            scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores


def create_candidate_reranker(settings: Settings) -> CandidateReranker:
    fallback = DefaultCandidateReranker(
        settings.retrieval_min_score,
        settings.retrieval_max_chunks_per_document,
    )
    if settings.reranker_type == "identity":
        return IdentityCandidateReranker()
    if settings.reranker_type == "external":
        return ExternalCandidateReranker(
            endpoint=settings.reranker_endpoint,
            model=settings.reranker_model,
            api_key=settings.reranker_api_key,
            timeout_seconds=settings.model_timeout_seconds,
        )
    return fallback
