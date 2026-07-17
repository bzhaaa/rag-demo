import time
from typing import Any, Dict, List

from app.config import Settings
from app.rag.contracts.protocols import WebSearchProvider, WebSearchResponse
from app.rag.contracts.state import RAGState
from app.rag.utils import timing_with
from app.rag.web_search import merge_web_results


class WebSearchModule:
    def __init__(self, provider: WebSearchProvider, settings: Settings) -> None:
        self.provider = provider
        self.settings = settings

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        results: List[Dict[str, Any]] = []
        query_count = 0
        failed = False
        retry_count = 0
        error_type = None
        request_ids: List[str] = []
        provider_response_time = 0.0
        usage_credits = 0.0
        for query in state.get("queries", [])[: self.settings.web_search_max_queries]:
            query_count += 1
            try:
                response = self.provider.search(
                    query,
                    self.settings.web_search_result_count,
                )
            except Exception as exc:
                failed = True
                error_type = getattr(exc, "error_type", type(exc).__name__)
                retry_count += int(getattr(exc, "retry_count", 0))
                break
            if isinstance(response, WebSearchResponse):
                results.extend(response.results)
                item_diagnostics = response.diagnostics
                retry_count += int(item_diagnostics.get("retry_count") or 0)
                if item_diagnostics.get("request_id"):
                    request_ids.append(item_diagnostics["request_id"])
                provider_response_time += float(
                    item_diagnostics.get("provider_response_time") or 0
                )
                usage_credits += float(
                    item_diagnostics.get("usage_credits") or 0
                )
            else:
                results.extend(response)
        merged = [] if failed else merge_web_results(results)
        return {
            **state,
            "web_candidates": merged,
            "web_search_attempted": True,
            "web_search_failed": failed,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "web_search": {
                    "attempted": True,
                    "query_count": query_count,
                    "candidate_count": len(merged),
                    "result_count": len(merged),
                    "provider": self.provider.name,
                    "failed": failed,
                    "retry_count": retry_count,
                    "error_type": error_type,
                    "request_id": (
                        request_ids[0]
                        if len(request_ids) == 1
                        else request_ids or None
                    ),
                    "provider_response_time": provider_response_time or None,
                    "usage_credits": usage_credits or None,
                },
            },
            "timings": timing_with(
                state.get("timings", {}),
                "web_search",
                time.perf_counter() - started,
            ),
        }
