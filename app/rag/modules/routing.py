import time

from app.rag.contracts.protocols import EvidenceRouter
from app.rag.contracts.state import RAGState
from app.rag.utils import timing_with


class LLMEvidenceRoutingModule:
    def __init__(self, router: EvidenceRouter, web_search_enabled: bool) -> None:
        self.router = router
        self.web_search_enabled = web_search_enabled

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        if not self.web_search_enabled:
            return {**state, "evidence_route": "knowledge_base"}
        failed = False
        try:
            route = self.router.route_evidence(
                state["question"],
                state.get("relevant", []),
            )
            if route not in {"knowledge_base", "web", "hybrid"}:
                raise ValueError("invalid evidence route")
        except Exception:
            route = ""
            failed = True
        return {
            **state,
            "evidence_route": route,
            "evidence_routing_failed": failed,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "evidence_route": route,
                "evidence_routing_failed": failed,
            },
            "timings": timing_with(
                state.get("timings", {}),
                "evidence_routing",
                time.perf_counter() - started,
            ),
        }

    @staticmethod
    def next_node(state: RAGState) -> str:
        if state.get("evidence_route") in {"web", "hybrid"}:
            return "web_search"
        return "select_evidence"
