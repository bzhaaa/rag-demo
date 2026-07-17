from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class Candidate:
    chunk_id: str
    content: str
    score: float = 0.0
    document_uuid: Optional[str] = None
    version_uuid: Optional[str] = None
    version_number: Optional[int] = None
    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: Optional[int] = None
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None
    parent_index: Optional[int] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    chunking_strategy: Optional[str] = None
    chunking_version: Optional[str] = None
    source_name: Optional[str] = None
    source_type: str = "knowledge_base"
    url: Optional[str] = None
    title: Optional[str] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_sources: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    _FIELDS = {
        "chunk_id",
        "content",
        "score",
        "document_uuid",
        "version_uuid",
        "version_number",
        "page_number",
        "page_start",
        "page_end",
        "chunk_index",
        "parent_chunk_id",
        "parent_content",
        "parent_index",
        "section_title",
        "section_path",
        "chunking_strategy",
        "chunking_version",
        "source_name",
        "source_type",
        "url",
        "title",
        "dense_score",
        "sparse_score",
        "rerank_score",
        "retrieval_sources",
    }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Candidate":
        data = dict(value)
        return cls(
            chunk_id=str(data.get("chunk_id") or ""),
            content=str(data.get("content") or ""),
            score=float(data.get("score") or 0),
            document_uuid=data.get("document_uuid"),
            version_uuid=data.get("version_uuid"),
            version_number=data.get("version_number"),
            page_number=data.get("page_number"),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            chunk_index=data.get("chunk_index"),
            parent_chunk_id=data.get("parent_chunk_id"),
            parent_content=data.get("parent_content"),
            parent_index=data.get("parent_index"),
            section_title=data.get("section_title"),
            section_path=data.get("section_path"),
            chunking_strategy=data.get("chunking_strategy"),
            chunking_version=data.get("chunking_version"),
            source_name=data.get("source_name"),
            source_type=str(data.get("source_type") or "knowledge_base"),
            url=data.get("url"),
            title=data.get("title"),
            dense_score=_optional_float(data.get("dense_score")),
            sparse_score=_optional_float(data.get("sparse_score")),
            rerank_score=_optional_float(data.get("rerank_score")),
            retrieval_sources=list(data.get("retrieval_sources") or []),
            extra={key: item for key, item in data.items() if key not in cls._FIELDS},
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            **self.extra,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "document_uuid": self.document_uuid,
            "version_uuid": self.version_uuid,
            "version_number": self.version_number,
            "page_number": self.page_number,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "parent_chunk_id": self.parent_chunk_id,
            "parent_content": self.parent_content,
            "parent_index": self.parent_index,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "chunking_strategy": self.chunking_strategy,
            "chunking_version": self.chunking_version,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "url": self.url,
            "title": self.title,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
            "retrieval_sources": list(self.retrieval_sources),
        }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


@dataclass
class PipelineInput:
    question: str
    version_uuids: List[str]
    authorized_version_count: int


@dataclass
class QueryResult:
    queries: List[str]
    attempted: bool = True


@dataclass
class RetrievalResult:
    candidates: List[Candidate]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    candidates: List[Candidate]
    relevant: List[Candidate]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    route: str
    failed: bool = False


@dataclass
class GenerationResult:
    answer: str
    refused: bool = False
    refusal_reason: Optional[str] = None
    refusal_detail: Optional[str] = None


@dataclass
class ValidationResult:
    answer: str
    cited_indices: List[int]
    refused: bool = False
    refusal_reason: Optional[str] = None
    refusal_detail: Optional[str] = None
