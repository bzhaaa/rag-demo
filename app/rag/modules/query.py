import time
from typing import Any, Dict

from app.config import Settings
from app.rag.contracts.protocols import QueryPreprocessor, RAGModelGateway
from app.rag.contracts.state import RAGState
from app.rag.utils import timing_with


class DefaultQueryModule:
    def __init__(
        self,
        preprocessor: QueryPreprocessor,
        model_gateway: RAGModelGateway,
        settings: Settings,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_gateway = model_gateway
        self.settings = settings

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        queries = self.preprocessor.preprocess(
            state["question"],
            self.model_gateway,
            self.settings.query_rewrite_max_queries,
        )
        diagnostics: Dict[str, Any] = state.get("diagnostics", {})
        return {
            **state,
            "queries": queries,
            "query_rewrite_attempted": True,
            "diagnostics": {
                **diagnostics,
                "queries": queries,
                "query_rewrite_attempted": True,
            },
            "timings": timing_with(
                state.get("timings", {}),
                "query_preprocess",
                time.perf_counter() - started,
            ),
        }
