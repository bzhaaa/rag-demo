from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    queries: List[str]
    version_uuids: List[str]
    candidates: List[Dict[str, Any]]
    relevant: List[Dict[str, Any]]
    web_candidates: List[Dict[str, Any]]
    web_relevant: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    evidence_route: str
    evidence_routing_failed: bool
    web_search_attempted: bool
    web_search_failed: bool
    answer: str
    cited_indices: List[int]
    query_rewrite_attempted: bool
    refused: bool
    refusal_reason: Optional[str]
    refusal_detail: Optional[str]
    timings: Dict[str, float]
    diagnostics: Dict[str, Any]
