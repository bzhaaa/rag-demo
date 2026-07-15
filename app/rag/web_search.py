import hashlib
from typing import Any, Dict, List, Sequence

from app.config import Settings
from app.rag.types import WebSearchProvider


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
    if settings.web_search_provider == "mock":
        return MockWebSearchProvider()
    return DisabledWebSearchProvider()
