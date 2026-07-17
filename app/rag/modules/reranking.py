import threading
import time
from typing import Any, Dict, List, Sequence, Tuple, cast

from app.config import Settings
from app.rag.contracts.models import Candidate
from app.rag.contracts.protocols import CandidateReranker, FallbackRelevanceGrader
from app.rag.contracts.state import RAGState
from app.rag.rerankers import RerankerError
from app.rag.utils import timing_with


class RerankingModule:
    def __init__(
        self,
        reranker: CandidateReranker,
        fallback_grader: FallbackRelevanceGrader,
        settings: Settings,
    ) -> None:
        self.reranker = reranker
        self.fallback_grader = fallback_grader
        self.settings = settings
        self._grading_lock = threading.Lock()
        self._grading_failures = 0
        self._grading_circuit_open_until = 0.0

    def run_knowledge(self, state: RAGState) -> RAGState:
        return self._run(state, "knowledge", "candidates", "relevant", "grading")

    def run_web(self, state: RAGState) -> RAGState:
        return self._run(
            state,
            "web",
            "web_candidates",
            "web_relevant",
            "web_grading",
        )

    def _run(
        self,
        state: RAGState,
        stage: str,
        candidate_key: str,
        relevant_key: str,
        timing_key: str,
    ) -> RAGState:
        started = time.perf_counter()
        state_dict: Dict[str, Any] = dict(state)
        candidates, relevant, diagnostics = self.grade_candidates(
            state["question"],
            cast(Sequence[Dict[str, Any]], state_dict.get(candidate_key, [])),
            stage,
        )
        current_diagnostics = state.get("diagnostics", {})
        state_dict.update(
            {
                candidate_key: candidates,
                relevant_key: relevant,
                "diagnostics": {
                **current_diagnostics,
                "reranking": {
                    **current_diagnostics.get("reranking", {}),
                    stage: diagnostics,
                },
            },
                "timings": timing_with(
                    state.get("timings", {}),
                    timing_key,
                    time.perf_counter() - started,
                ),
            }
        )
        return cast(RAGState, state_dict)

    def grade_candidates(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        stage: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], [], {
                "stage": stage,
                "input_count": 0,
                "passed_count": 0,
                "relevant_count": 0,
                "failure_count": 0,
                "skipped": "no_candidates",
                "min_score": self.settings.reranker_min_score,
                "top_k": self.settings.reranker_top_k,
                "results": [],
            }
        try:
            reranked = self.reranker.rerank(question, candidates)
        except Exception as exc:
            return self._handle_failure(question, candidates, stage, exc)
        return self._admit(candidates, reranked, stage)

    def _admit(
        self,
        candidates: Sequence[Dict[str, Any]],
        reranked: Sequence[Dict[str, Any]],
        stage: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        original = list(candidates)
        ranked = list(reranked)
        relevant = [
            candidate
            for candidate in ranked
            if float(candidate.get("rerank_score") or 0)
            >= self.settings.reranker_min_score
        ][: self.settings.reranker_top_k]
        ranked_by_chunk_id = {
            item.get("chunk_id"): item for item in ranked if item.get("chunk_id")
        }
        passed_ids = {item.get("chunk_id") for item in relevant}
        results = []
        for candidate in original:
            diagnostic_candidate = (
                ranked_by_chunk_id.get(candidate.get("chunk_id")) or candidate
            )
            passed = candidate.get("chunk_id") in passed_ids
            results.append(
                {
                    **self.candidate_summary(diagnostic_candidate),
                    "relevant": bool(passed),
                    "passed": bool(passed),
                }
            )
        self.reset_grading_failures()
        return original, relevant, {
            "stage": stage,
            "input_count": len(original),
            "passed_count": len(relevant),
            "relevant_count": len(relevant),
            "failure_count": 0,
            "failed": False,
            "min_score": self.settings.reranker_min_score,
            "top_k": self.settings.reranker_top_k,
            "failure_strategy": None,
            "endpoint": self.settings.reranker_endpoint,
            "model": self.settings.reranker_model,
            "results": results,
        }

    def _handle_failure(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        stage: str,
        exc: Exception,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        strategy = self.settings.reranker_failure_strategy
        base = {
            "stage": stage,
            "input_count": len(candidates),
            "passed_count": 0,
            "relevant_count": 0,
            "failure_count": len(candidates),
            "failed": True,
            "failure_strategy": strategy,
            "error_type": type(exc).__name__,
            "error": self._truncate(str(exc), 300),
            "min_score": self.settings.reranker_min_score,
            "top_k": self.settings.reranker_top_k,
            "endpoint": self.settings.reranker_endpoint,
            "model": self.settings.reranker_model,
            "results": [],
        }
        if strategy == "reject":
            return list(candidates), [], base
        if strategy == "vector":
            fallback = [
                candidate
                for candidate in sorted(
                    candidates,
                    key=lambda item: float(item.get("score") or 0),
                    reverse=True,
                )
                if float(candidate.get("score") or 0)
                >= float(self.settings.retrieval_min_score or 0)
            ][: self.settings.reranker_top_k]
            return list(candidates), fallback, {
                **base,
                "passed_count": len(fallback),
                "relevant_count": len(fallback),
                "failure_count": 1,
                "fallback": "vector",
                "results": [
                    {
                        **self.candidate_summary(candidate),
                        "relevant": candidate in fallback,
                        "passed": candidate in fallback,
                    }
                    for candidate in candidates
                ],
            }
        if strategy == "llm":
            reranked = list(candidates)
            if self.grading_circuit_is_open():
                return reranked, [], {
                    **base,
                    "skipped": "circuit_open",
                    "fallback": "llm",
                }
            responses = self.fallback_grader.grade_relevance(
                question,
                reranked,
                self.settings.relevance_max_concurrency,
            )
            relevant: List[Dict[str, Any]] = []
            failure_count = 0
            results: List[Dict[str, Any]] = []
            for candidate, response in zip(reranked, responses):
                if isinstance(response, Exception):
                    failure_count += 1
                    results.append(
                        {
                            **self.candidate_summary(candidate),
                            "relevant": False,
                            "error_type": type(response).__name__,
                            "error": self._truncate(str(response), 300),
                        }
                    )
                    continue
                is_relevant = (
                    response
                    if isinstance(response, bool)
                    else bool(getattr(response, "relevant", False))
                )
                results.append(
                    {
                        **self.candidate_summary(candidate),
                        "relevant": bool(is_relevant),
                        "raw_response": self._truncate(
                            str(getattr(response, "raw_response", "")),
                            300,
                        ),
                        "parsed_score": getattr(response, "parsed_score", None),
                    }
                )
                if is_relevant:
                    relevant.append(candidate)
            if reranked and failure_count == len(reranked):
                self.record_grading_failure()
            else:
                self.reset_grading_failures()
            admitted = relevant[: self.settings.reranker_top_k]
            return reranked, admitted, {
                **base,
                "passed_count": len(admitted),
                "relevant_count": len(admitted),
                "failure_count": failure_count,
                "fallback": "llm",
                "results": results,
            }
        raise RerankerError(
            f"unsupported reranker failure strategy: {strategy}"
        )

    def grading_circuit_is_open(self) -> bool:
        with self._grading_lock:
            return time.monotonic() < self._grading_circuit_open_until

    def record_grading_failure(self) -> None:
        with self._grading_lock:
            self._grading_failures += 1
            if self._grading_failures >= 3:
                self._grading_circuit_open_until = time.monotonic() + 30

    def reset_grading_failures(self) -> None:
        with self._grading_lock:
            self._grading_failures = 0
            self._grading_circuit_open_until = 0.0

    @staticmethod
    def candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return KnowledgeCandidateSummary.from_mapping(candidate)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: limit - 3] + "..."


class KnowledgeCandidateSummary:
    @staticmethod
    def from_mapping(candidate: Dict[str, Any]) -> Dict[str, Any]:
        item = Candidate.from_mapping(candidate)
        return {
            "chunk_id": item.chunk_id,
            "document_uuid": item.document_uuid,
            "version_uuid": item.version_uuid,
            "source_type": item.source_type,
            "score": item.score,
            "dense_score": item.dense_score,
            "sparse_score": item.sparse_score,
            "retrieval_sources": item.retrieval_sources,
            "rerank_score": item.rerank_score,
            "parent_chunk_id": item.parent_chunk_id,
            "section_path": item.section_path,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "chunking_strategy": item.chunking_strategy,
            "chunking_version": item.chunking_version,
        }
