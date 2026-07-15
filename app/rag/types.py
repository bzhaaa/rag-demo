from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict, Union


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


class RAGModelGateway(Protocol):
    def grade_relevance(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        max_concurrency: int,
    ) -> List[Union[bool, Exception]]:
        ...

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        strict_citations: bool = False,
    ) -> str:
        ...

    def rewrite_queries(self, question: str, count: int = 2) -> List[str]:
        ...

    def rewrite_query(self, question: str) -> str:
        ...

    def generate_hypothetical_document(self, question: str) -> str:
        ...

    def rewrite_step_back_query(self, question: str) -> str:
        ...

    def route_evidence(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        ...


class QueryPreprocessor(Protocol):
    def preprocess(
        self,
        question: str,
        model_gateway: RAGModelGateway,
        max_queries: Optional[int] = None,
    ) -> List[str]:
        ...


class CandidateReranker(Protocol):
    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ...


@dataclass
class WebSearchResponse:
    results: List[Dict[str, Any]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class WebSearchProvider(Protocol):
    name: str

    def search(
        self, query: str, limit: int
    ) -> Union[List[Dict[str, Any]], WebSearchResponse]:
        ...
