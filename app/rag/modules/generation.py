import time

from app.config import Settings
from app.rag.contracts.protocols import AnswerGenerator
from app.rag.contracts.state import RAGState
from app.rag.utils import timing_with


class LangChainGenerationModule:
    def __init__(self, generator: AnswerGenerator, settings: Settings) -> None:
        self.generator = generator
        self.settings = settings

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        evidence = state.get("evidence", state.get("relevant", []))
        if len(evidence) < self.settings.rag_min_relevant_documents:
            web_enabled = self.settings.web_search_enabled
            detail = refusal_detail(state, web_enabled)
            return {
                **state,
                "answer": (
                    "当前知识库和外部来源中没有足够证据回答这个问题。"
                    if web_enabled
                    else "当前知识库中没有足够且经过授权的证据回答这个问题。"
                ),
                "cited_indices": [],
                "refused": True,
                "refusal_reason": (
                    "insufficient_evidence"
                    if web_enabled
                    else "insufficient_authorized_evidence"
                ),
                "refusal_detail": detail,
                "diagnostics": {
                    **state.get("diagnostics", {}),
                    "refusal_detail": detail,
                },
                "timings": timing_with(
                    state.get("timings", {}),
                    "generation",
                    time.perf_counter() - started,
                ),
            }
        answer = self.generator.generate_answer(state["question"], evidence)
        return {
            **state,
            "answer": answer,
            "refused": False,
            "refusal_reason": None,
            "refusal_detail": None,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "refusal_detail": None,
            },
            "timings": timing_with(
                state.get("timings", {}),
                "generation",
                time.perf_counter() - started,
            ),
        }


def refusal_detail(state: RAGState, web_search_enabled: bool) -> str:
    diagnostics = state.get("diagnostics", {})
    reranking = diagnostics.get("reranking", {})
    knowledge_reranking = reranking.get("knowledge", {})
    grading = diagnostics.get("grading", {})
    knowledge_grading = grading.get("knowledge", {})
    if state.get("evidence_routing_failed"):
        return "evidence_routing_failed"
    if state.get("web_search_failed"):
        return "web_search_failed"
    if (
        knowledge_reranking.get("failed")
        and knowledge_reranking.get("failure_strategy") == "reject"
    ):
        return "reranker_failed"
    if "version_uuids" in state and not state.get("version_uuids"):
        return "no_authorized_documents"
    if "queries" in state and not state.get("queries"):
        return "no_retrieval_queries"
    if "candidates" in state and not state.get("candidates"):
        return "no_retrieval_results"
    if knowledge_grading.get("skipped") == "circuit_open":
        return "relevance_grading_circuit_open"
    if (
        knowledge_grading.get("input_count")
        and knowledge_grading.get("failure_count")
        == knowledge_grading.get("input_count")
    ):
        return "relevance_grading_failed"
    if state.get("candidates") and not state.get("relevant"):
        return "no_relevant_evidence"
    if state.get("evidence_route") == "web":
        if not state.get("web_candidates"):
            return "no_web_results"
        if not state.get("web_relevant"):
            return "no_relevant_web_evidence"
    if state.get("evidence_route") == "hybrid":
        return "insufficient_hybrid_evidence"
    return "insufficient_evidence" if web_search_enabled else "no_relevant_evidence"
