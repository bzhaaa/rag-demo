from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict, Union


class RAGState(TypedDict, total=False):
    question: str
    queries: List[str]
    version_uuids: List[str]
    candidates: List[Dict[str, Any]]
    relevant: List[Dict[str, Any]]
    answer: str
    cited_indices: List[int]
    corrective_attempted: bool
    query_rewrite_attempted: bool
    refused: bool
    refusal_reason: Optional[str]
    timings: Dict[str, float]


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
