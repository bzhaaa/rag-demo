from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union

from app.rag.contracts.models import Candidate


class QueryRewriter(Protocol):
    def rewrite_queries(self, question: str, count: int = 2) -> List[str]: ...

    def rewrite_query(self, question: str) -> str: ...

    def generate_hypothetical_document(self, question: str) -> str: ...

    def rewrite_step_back_query(self, question: str) -> str: ...


class EvidenceRouter(Protocol):
    def route_evidence(
        self, question: str, evidence: Sequence[Dict[str, Any]]
    ) -> str: ...


class AnswerGenerator(Protocol):
    def generate_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        strict_citations: bool = False,
    ) -> str: ...


class FallbackRelevanceGrader(Protocol):
    def grade_relevance(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        max_concurrency: int,
    ) -> List[Any]: ...


class RAGModelGateway(
    QueryRewriter,
    EvidenceRouter,
    AnswerGenerator,
    FallbackRelevanceGrader,
    Protocol,
):
    pass


class QueryPreprocessor(Protocol):
    def preprocess(
        self,
        question: str,
        model_gateway: RAGModelGateway,
        max_queries: Optional[int] = None,
    ) -> List[str]: ...


class KnowledgeRetriever(Protocol):
    def retrieve(
        self, query: str, version_uuids: Sequence[str], limit: int
    ) -> Dict[str, List[Candidate]]: ...


class CandidateFusion(Protocol):
    def fuse(
        self,
        channels: Mapping[str, Sequence[Candidate]],
        limit: int,
    ) -> List[Candidate]: ...

    def merge_queries(
        self, candidates: Sequence[Candidate]
    ) -> List[Candidate]: ...


class CandidateReranker(Protocol):
    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...


class EvidenceSelector(Protocol):
    def select(self, state: Dict[str, Any]) -> List[Candidate]: ...


class CitationValidator(Protocol):
    def validate(self, answer: str, evidence_count: int) -> List[int]: ...


@dataclass
class WebSearchResponse:
    results: List[Dict[str, Any]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class WebSearchProvider(Protocol):
    name: str

    def search(
        self, query: str, limit: int
    ) -> Union[List[Dict[str, Any]], WebSearchResponse]: ...
