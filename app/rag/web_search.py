import hashlib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests  # type: ignore[import-untyped,unused-ignore]

from app.config import Settings
from app.rag.types import WebSearchProvider, WebSearchResponse


class WebSearchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_count: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.retry_count = retry_count
        self.error_type = error_type or type(self).__name__


class DisabledWebSearchProvider:
    name = "disabled"

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        return []


class MockWebSearchProvider:
    name = "mock"

    _RESULTS: Sequence[Dict[str, Any]] = (
        {
            "title": "Mock: Corrective RAG Overview",
            "url": "https://example.com/mock/crag-overview",
            "content": (
                "Mock data: Corrective RAG evaluates retrieved evidence and can "
                "use an external search source when internal evidence is insufficient."
            ),
            "score": 0.95,
        },
        {
            "title": "Mock: RAG Evidence Routing",
            "url": "https://example.com/mock/rag-evidence-routing",
            "content": (
                "Mock data: Evidence routing can select knowledge-base, web, or "
                "hybrid context before answer generation."
            ),
            "score": 0.9,
        },
        {
            "title": "Mock: Citation Validation",
            "url": "https://example.com/mock/citation-validation",
            "content": (
                "Mock data: Generated answers should cite only evidence that was "
                "actually supplied to the model."
            ),
            "score": 0.85,
        },
    )

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if not query.strip() or limit <= 0:
            return []
        return [dict(item) for item in self._RESULTS[:limit]]


class TavilyWebSearchProvider:
    name = "tavily"

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.tavily.com/search",
        search_depth: str = "basic",
        topic: str = "general",
        timeout_seconds: int = 10,
        max_retries: int = 2,
        post: Optional[Callable[..., Any]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.search_depth = search_depth
        self.topic = topic
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.post = post or requests.post
        self.sleep = sleep or time.sleep

    def search(self, query: str, limit: int) -> WebSearchResponse:
        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return WebSearchResponse(results=[], diagnostics={"retry_count": 0})
        if not self.api_key or not self.endpoint:
            raise WebSearchError(
                "Tavily search is not configured",
                error_type="ConfigurationError",
            )

        payload = {
            "query": normalized_query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise WebSearchError(
                    "Tavily request failed",
                    retry_count=attempt,
                    error_type=type(exc).__name__,
                ) from exc

            status_code = int(getattr(response, "status_code", 0))
            if status_code == 429 or status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise WebSearchError(
                    f"Tavily returned HTTP {status_code}",
                    retry_count=attempt,
                    error_type="HTTPError",
                )
            if status_code >= 400:
                raise WebSearchError(
                    f"Tavily returned HTTP {status_code}",
                    retry_count=attempt,
                    error_type="HTTPError",
                )

            try:
                response.raise_for_status()
                data = response.json()
                results = self._parse_results(data)
            except WebSearchError:
                raise
            except Exception as exc:
                raise WebSearchError(
                    "Tavily returned an invalid response",
                    retry_count=attempt,
                    error_type=type(exc).__name__,
                ) from exc
            return WebSearchResponse(
                results=results,
                diagnostics=self._diagnostics(data, attempt),
            )

        raise WebSearchError("Tavily request failed")

    def _sleep_before_retry(self, attempt: int) -> None:
        self.sleep(0.25 * (2**attempt))

    @staticmethod
    def _parse_results(data: Any) -> List[Dict[str, Any]]:
        if not isinstance(data, dict) or "results" not in data:
            raise WebSearchError(
                "Tavily response must contain results",
                error_type="ResponseValidationError",
            )
        raw_results = data["results"]
        if not isinstance(raw_results, list):
            raise WebSearchError(
                "Tavily results must be a list",
                error_type="ResponseValidationError",
            )
        results: List[Dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                raise WebSearchError(
                    "Tavily result must be an object",
                    error_type="ResponseValidationError",
                )
            try:
                score = float(item.get("score") or 0)
            except (TypeError, ValueError) as exc:
                raise WebSearchError(
                    "Tavily result score must be numeric",
                    error_type="ResponseValidationError",
                ) from exc
            if not math.isfinite(score) or score < 0 or score > 1:
                raise WebSearchError(
                    "Tavily result score must be between 0 and 1",
                    error_type="ResponseValidationError",
                )
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "score": score,
                }
            )
        return results

    @staticmethod
    def _diagnostics(data: Dict[str, Any], retry_count: int) -> Dict[str, Any]:
        usage = data.get("usage")
        usage_credits = (
            usage.get("credits") if isinstance(usage, dict) else None
        )
        return {
            "request_id": data.get("request_id"),
            "provider_response_time": data.get("response_time"),
            "usage_credits": usage_credits,
            "retry_count": retry_count,
        }


def normalize_web_result(item: Dict[str, Any]) -> Dict[str, Any]:
    url = str(item.get("url") or "").strip()
    title = str(item.get("title") or url or "Web source").strip()
    content = str(item.get("content") or item.get("snippet") or "").strip()
    stable_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return {
        "source_type": "web",
        "url": url,
        "title": title,
        "source_name": title,
        "document_uuid": None,
        "version_uuid": None,
        "version_number": None,
        "page_number": None,
        "chunk_id": f"web:{stable_id}",
        "content": content,
        "score": float(item.get("score") or 0),
    }


def merge_web_results(
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw_item in results:
        item = normalize_web_result(raw_item)
        url = item["url"]
        if not url or not item["content"]:
            continue
        existing = merged.get(url)
        if existing is None or item["score"] > existing["score"]:
            merged[url] = item
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )


def create_web_search_provider(settings: Settings) -> WebSearchProvider:
    if not settings.web_search_enabled:
        return DisabledWebSearchProvider()
    return TavilyWebSearchProvider(
        api_key=settings.tavily_api_key,
        endpoint=settings.tavily_endpoint,
        search_depth=settings.tavily_search_depth,
        topic=settings.tavily_topic,
        timeout_seconds=settings.web_search_timeout_seconds,
        max_retries=settings.web_search_max_retries,
    )
