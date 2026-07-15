from typing import Any, Dict, List, Optional, Sequence

import pytest
import requests
from sqlalchemy import select

from app.config import Settings
from app.models import AuditLog, Message
from app.rag import (
    DefaultCandidateReranker,
    DefaultQueryPreprocessor,
    ExternalCandidateReranker,
    IdentityCandidateReranker,
    MockWebSearchProvider,
    RAGService,
    RerankerError,
    TavilyWebSearchProvider,
    WebSearchError,
    create_candidate_reranker,
    create_web_search_provider,
)
from app.schemas import Citation


class FakeVectorStore:
    def __init__(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        self.results = results
        self.calls: List[tuple[str, List[str], int]] = []

    def search(
        self, question: str, version_uuids: Sequence[str], limit: int
    ) -> List[Dict[str, Any]]:
        self.calls.append((question, list(version_uuids), limit))
        return list(self.results.get(question, []))


class FakeModelGateway:
    def __init__(
        self,
        answers: Sequence[str],
        relevance: Sequence[object] = (),
        rewrites: Sequence[str] = (),
        rewrite_batches: Sequence[Sequence[str]] = (),
        direct_rewrite: str = "",
        hyde_document: str = "",
        step_back_query: str = "",
        routes: Sequence[object] = (),
    ) -> None:
        self.answers = list(answers)
        self.relevance = list(relevance)
        self.rewrites = list(rewrites)
        self.rewrite_batches = [list(batch) for batch in rewrite_batches]
        self.direct_rewrite = direct_rewrite
        self.hyde_document = hyde_document
        self.step_back_query = step_back_query
        self.routes = list(routes)
        self.generate_calls: List[bool] = []
        self.rewrite_calls = 0
        self.standalone_calls = 0
        self.hyde_calls = 0
        self.step_back_calls = 0
        self.route_calls = 0
        self.grade_calls = 0

    def grade_relevance(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        max_concurrency: int,
    ) -> List[object]:
        self.grade_calls += 1
        if self.relevance:
            result = self.relevance[: len(candidates)]
            self.relevance = self.relevance[len(candidates) :]
            return result
        return [True] * len(candidates)

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        strict_citations: bool = False,
    ) -> str:
        self.generate_calls.append(strict_citations)
        return self.answers.pop(0)

    def rewrite_queries(self, question: str, count: int = 2) -> List[str]:
        self.rewrite_calls += 1
        if self.rewrite_batches:
            return self.rewrite_batches.pop(0)[:count]
        return list(self.rewrites[:count])

    def rewrite_query(self, question: str) -> str:
        self.standalone_calls += 1
        if self.direct_rewrite:
            return self.direct_rewrite
        return self.rewrites[0] if self.rewrites else ""

    def generate_hypothetical_document(self, question: str) -> str:
        self.hyde_calls += 1
        return self.hyde_document

    def rewrite_step_back_query(self, question: str) -> str:
        self.step_back_calls += 1
        return self.step_back_query

    def route_evidence(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        self.route_calls += 1
        if not self.routes:
            return "knowledge_base"
        route = self.routes.pop(0)
        if isinstance(route, Exception):
            raise route
        return str(route)


class FakeWebSearchProvider:
    name = "fake"

    def __init__(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        self.results = results
        self.calls: List[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        self.calls.append((query, limit))
        return list(self.results.get(query, []))


class FakeScoredReranker:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls = 0

    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        self.calls += 1
        ranked = []
        for candidate, score in zip(candidates, self.scores):
            item = dict(candidate)
            item["rerank_score"] = score
            ranked.append(item)
        return sorted(
            ranked,
            key=lambda item: float(item.get("rerank_score") or 0),
            reverse=True,
        )


class PassThroughReranker:
    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        result = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = 0.9
            result.append(item)
        return result


class FailingReranker:
    def __init__(self, exc: Optional[Exception] = None) -> None:
        self.exc = exc or RuntimeError("reranker unavailable")

    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        raise self.exc


def create_rag_user_and_document(db, models):
    department = models.Department(name="Knowledge")
    user = models.User(
        username="reader",
        email="reader@example.com",
        password_hash="x",
        role=models.Role.viewer.value,
    )
    db.add_all([department, user])
    db.flush()
    db.add(
        models.DepartmentMembership(
            user_id=user.id,
            department_id=department.id,
        )
    )
    document = models.Document(
        title="RAG Policy",
        owner_id=user.id,
        department_id=department.id,
        visibility=models.Visibility.restricted.value,
    )
    db.add(document)
    db.flush()
    version = models.DocumentVersion(
        document_id=document.id,
        version_number=1,
        checksum="a" * 64,
        object_key="documents/policy.md",
        source_name="policy.md",
        mime_type="text/markdown",
        size_bytes=100,
        status=models.VersionStatus.ready.value,
    )
    db.add(version)
    db.flush()
    document.current_version_id = version.id
    db.commit()
    db.refresh(user)
    return user, document, version


def make_chunk(document, version, index: int, content: str, score: float = 0.9):
    return {
        "score": score,
        "chunk_id": f"{document.uuid}:{version.version_number}:{index}",
        "document_uuid": document.uuid,
        "version_uuid": version.uuid,
        "version_number": version.version_number,
        "page_number": index + 1,
        "chunk_index": index,
        "source_name": version.source_name,
        "content": content,
    }


def test_answer_uses_authorized_versions_and_persists_complete_metrics(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "RAG 只能根据授权证据回答。")
    vector_store = FakeVectorStore({"RAG 如何回答？": [chunk]})
    gateway = FakeModelGateway(["RAG 只能根据授权证据回答。[1]"])
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        settings=Settings(web_search_enabled=False),
    )

    result = service.answer(db, user, "RAG 如何回答？")

    assert vector_store.calls[0][1] == [version.uuid]
    assert result["answer"] == "RAG 只能根据授权证据回答。[1]"
    assert [item["chunk_id"] for item in result["citations"]] == [chunk["chunk_id"]]
    assert result["timings"]["total"] >= 0
    messages = list(db.scalars(select(Message).order_by(Message.id)))
    assert messages[-1].metrics["timings"] == result["timings"]
    diagnostics = messages[-1].metrics["rag_diagnostics"]
    assert diagnostics["authorized_version_count"] == 1
    assert diagnostics["queries"]
    assert diagnostics["retrieval"]["candidate_count"] == 1
    assert diagnostics["reranking"]["knowledge"]["relevant_count"] == 1
    assert diagnostics["selected_evidence_count"] == 1


def test_web_route_uses_injected_provider_and_returns_web_citation(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    knowledge_chunk = make_chunk(
        document,
        version,
        0,
        "Partial internal evidence that must be discarded by the web route.",
    )
    vector_store = FakeVectorStore({"CRAG": [knowledge_chunk]})
    web_search = FakeWebSearchProvider(
        {
            "CRAG": [
                {
                    "title": "Mock CRAG Guide",
                    "url": "https://example.com/mock-crag",
                    "content": "CRAG can fall back to external search.",
                    "score": 0.95,
                }
            ]
        }
    )
    gateway = FakeModelGateway(
        answers=["Use external search when internal evidence is insufficient [1]."],
        relevance=[True, True],
        routes=["web"],
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "CRAG")

    assert result["refused"] is False
    assert web_search.calls == [("CRAG", 5)]
    assert knowledge_chunk["chunk_id"] not in {
        item["chunk_id"] for item in result["citations"]
    }
    assert result["citations"] == [
        {
            "source_type": "web",
            "url": "https://example.com/mock-crag",
            "document_uuid": None,
            "document_title": "Mock CRAG Guide",
            "version": None,
            "page_number": None,
            "chunk_id": result["citations"][0]["chunk_id"],
            "excerpt": "CRAG can fall back to external search.",
        }
    ]
    assert "evidence_routing" in result["timings"]
    assert "web_search" in result["timings"]
    assert "web_grading" in result["timings"]


def test_default_reranker_filters_deduplicates_sorts_and_limits_documents():
    reranker = DefaultCandidateReranker(
        min_score=0.5,
        max_chunks_per_document=2,
    )
    candidates = [
        {
            "chunk_id": "a-low",
            "document_uuid": "a",
            "content": "低分内容",
            "score": 0.4,
        },
        {
            "chunk_id": "a-duplicate-low",
            "document_uuid": "a",
            "content": "  重复   内容 ",
            "score": 0.7,
        },
        {
            "chunk_id": "a-duplicate-high",
            "document_uuid": "a",
            "content": "重复 内容",
            "score": 0.9,
        },
        {
            "chunk_id": "a-third",
            "document_uuid": "a",
            "content": "第三段",
            "score": 0.8,
        },
        {
            "chunk_id": "a-fourth",
            "document_uuid": "a",
            "content": "第四段",
            "score": 0.6,
        },
        {
            "chunk_id": "b-one",
            "document_uuid": "b",
            "content": "另一文档",
            "score": 0.85,
        },
        {
            "chunk_id": "empty",
            "document_uuid": "b",
            "content": " ",
            "score": 0.99,
        },
    ]

    result = reranker.rerank("问题", candidates)

    assert [item["chunk_id"] for item in result] == [
        "a-duplicate-high",
        "b-one",
        "a-third",
    ]


def test_reranker_factory_uses_configured_strategy():
    assert isinstance(
        create_candidate_reranker(Settings(reranker_type="default")),
        DefaultCandidateReranker,
    )
    assert isinstance(
        create_candidate_reranker(Settings(reranker_type="identity")),
        IdentityCandidateReranker,
    )
    assert isinstance(
        create_candidate_reranker(
            Settings(
                reranker_type="external",
                reranker_endpoint="https://rerank.example/v1/rerank",
                reranker_model="rerank-v1",
            )
        ),
        ExternalCandidateReranker,
    )


def test_external_reranker_sends_query_and_orders_by_model_scores():
    requests = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.31},
                ]
            }

    def fake_post(url, json, headers, timeout):
        requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    candidates = [
        {"chunk_id": "first", "document_uuid": "a", "content": "第一段", "score": 0.9},
        {"chunk_id": "second", "document_uuid": "b", "content": "第二段", "score": 0.1},
    ]
    reranker = ExternalCandidateReranker(
        endpoint="https://rerank.example/v1/rerank",
        model="rerank-v1",
        api_key="secret",
        post=fake_post,
    )

    result = reranker.rerank("用户问题", candidates)

    assert requests[0]["url"] == "https://rerank.example/v1/rerank"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret"
    assert requests[0]["json"]["query"] == "用户问题"
    assert requests[0]["json"]["documents"] == ["第一段", "第二段"]
    assert [item["chunk_id"] for item in result] == ["second", "first"]
    assert result[0]["rerank_score"] == 0.98


def test_external_reranker_rejects_invalid_score_payloads():
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"results": [{"index": 0, "relevance_score": 1.5}]}

    reranker = ExternalCandidateReranker(
        endpoint="https://rerank.example/v1/rerank",
        model="rerank-v1",
        post=lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(RerankerError, match="relevance_score"):
        reranker.rerank(
            "question",
            [{"chunk_id": "first", "document_uuid": "a", "content": "text"}],
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"results": [{"index": 9, "relevance_score": 0.5}]}, "index"),
        (
            {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]},
            "duplicate",
        ),
        ({"results": [{"index": 0, "relevance_score": "nan"}]}, "between 0 and 1"),
        ({"results": [{"index": 0, "relevance_score": "not-a-number"}]}, "could not convert"),
        ({"data": [{"index": 0, "score": 0.8}]}, ""),
    ],
)
def test_external_reranker_validates_http_contract(payload, message):
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return payload

    reranker = ExternalCandidateReranker(
        endpoint="https://rerank.example/v1/rerank",
        model="rerank-v1",
        post=lambda *args, **kwargs: FakeResponse(),
    )
    candidates = [{"chunk_id": "first", "document_uuid": "a", "content": "text"}]

    if message:
        with pytest.raises(RerankerError, match=message):
            reranker.rerank("question", candidates)
    else:
        assert reranker.rerank("question", candidates)[0]["rerank_score"] == 0.8


def test_external_reranker_rejects_missing_results_field():
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"unexpected": []}

    reranker = ExternalCandidateReranker(
        endpoint="https://rerank.example/v1/rerank",
        model="rerank-v1",
        post=lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(RerankerError, match="results or data"):
        reranker.rerank(
            "question",
            [{"chunk_id": "first", "document_uuid": "a", "content": "text"}],
        )


def test_external_reranker_accepts_empty_results_as_no_evidence():
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"results": []}

    reranker = ExternalCandidateReranker(
        endpoint="https://rerank.example/v1/rerank",
        model="rerank-v1",
        post=lambda *args, **kwargs: FakeResponse(),
    )

    assert (
        reranker.rerank(
            "question",
            [{"chunk_id": "first", "document_uuid": "a", "content": "text"}],
        )
        == []
    )


def test_reranker_admission_filters_by_score_and_skips_llm_grading(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunks = [
        make_chunk(document, version, 0, "best evidence", score=0.7),
        make_chunk(document, version, 1, "weak evidence", score=0.6),
        make_chunk(document, version, 2, "second evidence", score=0.5),
    ]
    gateway = FakeModelGateway(["Answer from reranker evidence [1]."])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": chunks}),
        model_gateway=gateway,
        reranker=FakeScoredReranker([0.91, 0.2, 0.72]),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_min_score=0.5,
            reranker_top_k=1,
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is False
    assert gateway.grade_calls == 0
    assert [item["chunk_id"] for item in result["citations"]] == [
        chunks[0]["chunk_id"]
    ]
    message = db.scalar(select(Message).order_by(Message.id.desc()))
    reranking = message.metrics["rag_diagnostics"]["reranking"]["knowledge"]
    assert reranking["input_count"] == 3
    assert reranking["passed_count"] == 1
    assert reranking["min_score"] == 0.5
    assert reranking["top_k"] == 1
    assert reranking["results"][0]["rerank_score"] == 0.91


def test_reranker_admission_refuses_when_all_scores_are_below_threshold(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "weak evidence", score=0.9)
    gateway = FakeModelGateway([])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": [chunk]}),
        model_gateway=gateway,
        reranker=FakeScoredReranker([0.49]),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_min_score=0.5,
            reranker_top_k=6,
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is True
    assert result["refusal_detail"] == "no_relevant_evidence"
    assert gateway.grade_calls == 0


def test_empty_reranker_result_keeps_candidate_diagnostics(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "candidate evidence", score=0.9)
    gateway = FakeModelGateway([])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": [chunk]}),
        model_gateway=gateway,
        reranker=FakeScoredReranker([]),
        settings=Settings(query_rewrite_types="normalize"),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is True
    assert result["refusal_detail"] == "no_relevant_evidence"
    message = db.scalar(select(Message).order_by(Message.id.desc()))
    reranking = message.metrics["rag_diagnostics"]["reranking"]["knowledge"]
    assert reranking["input_count"] == 1
    assert reranking["passed_count"] == 0
    assert reranking["results"][0]["chunk_id"] == chunk["chunk_id"]
    assert reranking["results"][0]["rerank_score"] is None


def test_reranker_failure_reject_strategy_refuses(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "evidence", score=0.9)
    gateway = FakeModelGateway([])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": [chunk]}),
        model_gateway=gateway,
        reranker=FailingReranker(),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_failure_strategy="reject",
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is True
    assert result["refusal_detail"] == "reranker_failed"
    assert gateway.grade_calls == 0
    message = db.scalar(select(Message).order_by(Message.id.desc()))
    reranking = message.metrics["rag_diagnostics"]["reranking"]["knowledge"]
    assert reranking["failed"] is True
    assert reranking["failure_strategy"] == "reject"


def test_reranker_failure_vector_strategy_uses_vector_score_threshold(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunks = [
        make_chunk(document, version, 0, "vector evidence", score=0.91),
        make_chunk(document, version, 1, "low vector evidence", score=0.2),
    ]
    gateway = FakeModelGateway(["Vector fallback answer [1]."])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": chunks}),
        model_gateway=gateway,
        reranker=FailingReranker(),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_failure_strategy="vector",
            retrieval_min_score=0.5,
            reranker_top_k=6,
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is False
    assert gateway.grade_calls == 0
    assert [item["chunk_id"] for item in result["citations"]] == [
        chunks[0]["chunk_id"]
    ]


def test_reranker_failure_llm_strategy_falls_back_to_relevance_grading(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "llm fallback evidence", score=0.9)
    gateway = FakeModelGateway(["LLM fallback answer [1]."], relevance=[True])
    service = RAGService(
        vector_store=FakeVectorStore({"policy": [chunk]}),
        model_gateway=gateway,
        reranker=FailingReranker(),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_failure_strategy="llm",
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is False
    assert gateway.grade_calls == 1


def test_vector_failure_strategy_requires_retrieval_min_score():
    with pytest.raises(ValueError, match="retrieval_min_score"):
        Settings(reranker_failure_strategy="vector", retrieval_min_score=None)


def test_query_preprocessor_normalizes_and_deduplicates_queries():
    gateway = FakeModelGateway(
        answers=[],
        rewrites=[" 独立 查询 ", "扩展 查询", "独立 查询"],
    )
    preprocessor = DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize", "standalone", "multi_query"],
        max_queries=3,
    )

    result = preprocessor.preprocess("  原始   问题  ", gateway)

    assert result == ["原始 问题", "独立 查询", "扩展 查询"]
    assert gateway.standalone_calls == 1
    assert gateway.rewrite_calls == 1


def test_query_preprocessor_supports_direct_hyde_step_back_and_multi_query():
    gateway = FakeModelGateway(
        answers=[],
        rewrite_batches=[["扩展查询一", "扩展查询二"]],
        direct_rewrite="直接改写查询",
        hyde_document="一段符合知识库文体的假设答案",
        step_back_query="这个问题依赖哪些通用原理",
    )
    preprocessor = DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=[
            "normalize",
            "direct",
            "hyde",
            "step_back",
            "multi_query",
        ],
        max_queries=7,
    )

    result = preprocessor.preprocess(" 原始问题 ", gateway)

    assert result == [
        "原始问题",
        "直接改写查询",
        "一段符合知识库文体的假设答案",
        "这个问题依赖哪些通用原理",
        "扩展查询一",
        "扩展查询二",
    ]
    assert gateway.standalone_calls == 1
    assert gateway.hyde_calls == 1
    assert gateway.step_back_calls == 1
    assert gateway.rewrite_calls == 1


def test_query_preprocessor_respects_configured_types():
    gateway = FakeModelGateway(answers=[], rewrites=["独立查询", "扩展查询"])

    assert DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize"],
        max_queries=3,
    ).preprocess("  原始   问题  ", gateway) == ["原始 问题"]

    assert DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize", "standalone"],
        max_queries=3,
    ).preprocess("  原始   问题  ", gateway) == ["原始 问题", "独立查询"]

    gateway = FakeModelGateway(answers=[], rewrites=["独立查询", "扩展查询"])
    assert DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize", "multi_query"],
        max_queries=3,
    ).preprocess("  原始   问题  ", gateway) == ["原始 问题", "独立查询", "扩展查询"]


def test_query_rewrite_types_parse_from_comma_separated_config():
    settings = Settings(
        query_rewrite_types="normalize, standalone, multi_query"
    )

    assert settings.query_rewrite_types == [
        "normalize",
        "direct",
        "multi_query",
    ]


def test_query_rewrite_types_accept_all_supported_strategies():
    settings = Settings(
        query_rewrite_types=(
            "normalize,direct,hyde,step_back,multi_query"
        )
    )

    assert settings.query_rewrite_types == [
        "normalize",
        "direct",
        "hyde",
        "step_back",
        "multi_query",
    ]


def test_query_preprocessor_disabled_returns_normalized_original_only():
    gateway = FakeModelGateway(answers=[], rewrites=["不应使用"])
    preprocessor = DefaultQueryPreprocessor(
        enabled=False,
        rewrite_types=["normalize", "standalone", "multi_query"],
        max_queries=3,
    )

    result = preprocessor.preprocess("  原始   问题  ", gateway)

    assert result == ["原始 问题"]
    assert gateway.standalone_calls == 0
    assert gateway.rewrite_calls == 0


def test_empty_normalized_query_does_not_reach_vector_store(db_session):
    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    vector_store = FakeVectorStore({})
    gateway = FakeModelGateway(answers=[], rewrites=["不应使用"])
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        settings=Settings(web_search_enabled=False),
    )

    result = service.answer(db, user, "   ")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert result["refusal_detail"] == "no_retrieval_queries"
    assert vector_store.calls == []


def test_invalid_citations_retry_once_then_keep_only_referenced_chunks(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunks = [
        make_chunk(document, version, 0, "第一段证据。", score=0.9),
        make_chunk(document, version, 1, "第二段证据。", score=0.8),
    ]
    vector_store = FakeVectorStore({"引用测试": chunks})
    gateway = FakeModelGateway(
        [
            "第一次回答没有引用。",
            "严格重试后只使用第二段证据。[2]",
        ]
    )
    service = RAGService(vector_store=vector_store, model_gateway=gateway, reranker=PassThroughReranker())

    result = service.answer(db, user, "引用测试")

    assert result["refused"] is False
    assert gateway.generate_calls == [False, True]
    assert [item["chunk_id"] for item in result["citations"]] == [
        chunks[1]["chunk_id"]
    ]
    assert "citation_validation" in result["timings"]


def test_invalid_citations_after_retry_returns_structured_refusal(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "唯一证据。")
    vector_store = FakeVectorStore({"无效引用": [chunk]})
    gateway = FakeModelGateway(["错误引用。[2]", "仍然没有引用。"])
    service = RAGService(vector_store=vector_store, model_gateway=gateway, reranker=PassThroughReranker())

    result = service.answer(db, user, "无效引用")

    assert result["refused"] is True
    assert result["refusal_reason"] == "invalid_citations"
    assert result["citations"] == []


def test_initial_preprocessed_queries_are_the_only_knowledge_retrieval(
    db_session,
):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    first = make_chunk(document, version, 0, "无关内容。", score=0.2)
    vector_store = FakeVectorStore(
        {
            "原始问题": [first],
            "初检无关": [first],
        }
    )
    gateway = FakeModelGateway(
        [],
        relevance=[False],
        rewrite_batches=[["初检无关"], ["改写一", "改写二"]],
    )
    preprocessor = DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize", "multi_query"],
        max_queries=2,
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=FakeScoredReranker([0.1]),
        query_preprocessor=preprocessor,
        settings=Settings(web_search_enabled=False),
    )

    result = service.answer(db, user, "原始问题")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == [
        "原始问题",
        "初检无关",
    ]
    assert all(call[1] == [version.uuid] for call in vector_store.calls)
    assert "query_rewrite" not in result["timings"]
    assert set(result["timings"]).isdisjoint(
        {"query_rewrite", "corrective_retrieval"}
    )


def test_initial_retrieval_uses_preprocessed_queries_with_same_authorization(
    db_session,
):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    original = make_chunk(document, version, 0, "原始查询证据。", score=0.5)
    expanded = make_chunk(document, version, 1, "扩展查询证据。", score=0.95)
    vector_store = FakeVectorStore(
        {
            "原始问题": [original],
            "扩展问题": [expanded],
            "同义问题": [expanded],
        }
    )
    gateway = FakeModelGateway(
        answers=["扩展查询可以回答。[1]"],
        relevance=[True],
        rewrites=["扩展问题", "同义问题"],
    )
    preprocessor = DefaultQueryPreprocessor(
        enabled=True,
        rewrite_types=["normalize", "multi_query"],
        max_queries=3,
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        query_preprocessor=preprocessor,
    )

    result = service.answer(db, user, " 原始问题 ")

    assert [call[0] for call in vector_store.calls] == [
        "原始问题",
        "扩展问题",
        "同义问题",
    ]
    assert all(call[1] == [version.uuid] for call in vector_store.calls)
    assert [item["chunk_id"] for item in result["citations"]] == [
        expanded["chunk_id"]
    ]
    assert "query_preprocess" in result["timings"]


def test_initial_query_rewrite_does_not_repeat_when_evidence_is_sufficient(
    db_session,
):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "初次检索证据足够。")
    vector_store = FakeVectorStore({"原始问题": [chunk]})
    gateway = FakeModelGateway(
        ["初次检索可以回答。[1]"],
        relevance=[True],
        rewrites=["不应使用"],
    )
    service = RAGService(vector_store=vector_store, model_gateway=gateway, reranker=PassThroughReranker())

    result = service.answer(db, user, "原始问题")

    assert result["refused"] is False
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == ["原始问题", "不应使用"]


def test_grading_failures_open_circuit_and_refuse_conservatively(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "看起来相关的证据。")
    service = RAGService(
        vector_store=FakeVectorStore({"question": [chunk]}),
        reranker=FailingReranker(),
        model_gateway=FakeModelGateway(
            answers=[],
            relevance=[Exception("model down") for _ in range(6)],
            rewrites=[],
        ),
        settings=Settings(
            query_rewrite_types="normalize",
            reranker_failure_strategy="llm",
            web_search_enabled=False,
        ),
    )

    for _ in range(3):
        result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert service._grading_circuit_is_open()


def test_initial_query_rewrite_failure_keeps_authorized_evidence_refusal(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "无关内容。")
    vector_store = FakeVectorStore({"问题": [chunk]})
    gateway = FakeModelGateway(
        answers=[],
        relevance=[False],
        rewrites=[],
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=FakeScoredReranker([0.1]),
        settings=Settings(web_search_enabled=False),
    )

    result = service.answer(db, user, "问题")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == ["问题"]


def test_knowledge_base_route_does_not_call_web_search(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "Authorized answer.")
    vector_store = FakeVectorStore({"policy": [chunk]})
    web_search = FakeWebSearchProvider({})
    gateway = FakeModelGateway(
        answers=["Authorized answer [1]."],
        relevance=[True],
        routes=["knowledge_base"],
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is False
    assert web_search.calls == []
    assert result["citations"][0]["source_type"] == "knowledge_base"
    assert result["citations"][0]["url"] is None


def test_hybrid_route_uses_knowledge_first_and_persists_audit_details(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "Internal policy evidence.")
    vector_store = FakeVectorStore({"policy": [chunk]})
    web_search = FakeWebSearchProvider(
        {
            "policy": [
                {
                    "title": "External context",
                    "url": "https://example.com/external-context",
                    "content": "Public supporting evidence.",
                    "score": 0.9,
                }
            ]
        }
    )
    gateway = FakeModelGateway(
        answers=["Internal policy [1] is supplemented by public context [2]."],
        relevance=[True, True],
        routes=["hybrid"],
    )
    service = RAGService(
        vector_store=vector_store,
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is False
    assert [item["source_type"] for item in result["citations"]] == [
        "knowledge_base",
        "web",
    ]
    audit = db.scalar(select(AuditLog).order_by(AuditLog.id.desc()))
    assert audit.details["evidence_route"] == "hybrid"
    assert audit.details["web_search_attempted"] is True
    assert audit.details["knowledge_evidence_count"] == 1
    assert audit.details["web_evidence_count"] == 1
    assert audit.details["web_search_provider"] == "fake"
    messages = list(db.scalars(select(Message).order_by(Message.id)))
    assert messages[-1].citations == result["citations"]


def test_hybrid_route_refuses_when_one_evidence_source_is_missing(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "Internal policy evidence.")
    gateway = FakeModelGateway(
        answers=[],
        relevance=[True],
        routes=["hybrid"],
    )
    service = RAGService(
        vector_store=FakeVectorStore({"policy": [chunk]}),
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=FakeWebSearchProvider({}),
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "policy")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_evidence"
    assert result["citations"] == []


def test_evidence_routing_failure_refuses_without_web_search(db_session):
    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    web_search = FakeWebSearchProvider({})
    gateway = FakeModelGateway(
        answers=[],
        routes=[ValueError("invalid route JSON")],
    )
    service = RAGService(
        vector_store=FakeVectorStore({"question": []}),
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_evidence"
    assert web_search.calls == []


def test_web_results_are_deduplicated_by_url_and_keep_highest_score(db_session):
    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    web_search = FakeWebSearchProvider(
        {
            "question": [
                {
                    "title": "Low score",
                    "url": "https://example.com/result",
                    "content": "Old snippet.",
                    "score": 0.3,
                },
                {
                    "title": "High score",
                    "url": "https://example.com/result",
                    "content": "Best snippet.",
                    "score": 0.9,
                },
            ]
        }
    )
    gateway = FakeModelGateway(
        answers=["Best snippet [1].", "Best snippet [1]."],
        relevance=[True, True],
        routes=["web", "web"],
    )
    service = RAGService(
        vector_store=FakeVectorStore({"question": []}),
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    first = service.answer(db, user, "question")
    second = service.answer(db, user, "question")

    assert first["citations"][0]["document_title"] == "High score"
    assert first["citations"][0]["excerpt"] == "Best snippet."
    assert first["citations"][0]["chunk_id"] == second["citations"][0]["chunk_id"]


def test_mock_web_search_provider_is_disabled_by_default_and_returns_fixed_results():
    disabled = create_web_search_provider(Settings(web_search_enabled=False))
    mock = MockWebSearchProvider()

    assert disabled.name == "disabled"
    assert disabled.search("CRAG", 5) == []
    results = mock.search("CRAG", 5)
    assert len(results) == 3
    assert all(item["url"].startswith("https://example.com/") for item in results)
    assert all("Mock" in item["title"] for item in results)


def test_tavily_web_search_sends_expected_request_and_maps_diagnostics():
    requests = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {
                "request_id": "request-123",
                "response_time": 0.42,
                "usage": {"credits": 1},
                "results": [
                    {
                        "title": "Milvus",
                        "url": "https://example.com/milvus",
                        "content": "Milvus is a vector database.",
                        "score": 0.91,
                    }
                ],
            }

    def fake_post(url, json, headers, timeout):
        requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    provider = TavilyWebSearchProvider(
        api_key="secret",
        endpoint="https://api.tavily.com/search",
        timeout_seconds=10,
        max_retries=2,
        post=fake_post,
        sleep=lambda _: None,
    )

    response = provider.search("Milvus RAG", 5)

    assert requests == [
        {
            "url": "https://api.tavily.com/search",
            "json": {
                "query": "Milvus RAG",
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "timeout": 10,
        }
    ]
    assert response.results[0]["title"] == "Milvus"
    assert response.diagnostics == {
        "request_id": "request-123",
        "provider_response_time": 0.42,
        "usage_credits": 1,
        "retry_count": 0,
    }


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_tavily_web_search_retries_retryable_http_errors(status_code):
    calls = []
    sleeps = []

    class FakeResponse:
        def __init__(self, status: int) -> None:
            self.status_code = status

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> Dict[str, Any]:
            return {"results": []}

    responses = [FakeResponse(status_code), FakeResponse(200)]

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    provider = TavilyWebSearchProvider(
        api_key="secret",
        max_retries=2,
        post=fake_post,
        sleep=sleeps.append,
    )

    response = provider.search("question", 5)

    assert response.results == []
    assert response.diagnostics["retry_count"] == 1
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_tavily_web_search_does_not_retry_non_retryable_http_error():
    calls = 0

    class FakeResponse:
        status_code = 401

        def raise_for_status(self) -> None:
            raise RuntimeError("unauthorized")

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse()

    provider = TavilyWebSearchProvider(
        api_key="secret",
        max_retries=2,
        post=fake_post,
        sleep=lambda _: None,
    )

    with pytest.raises(WebSearchError, match="HTTP 401") as exc_info:
        provider.search("question", 5)

    assert calls == 1
    assert exc_info.value.retry_count == 0
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": []},
        {"results": "invalid"},
        {"results": ["invalid-item"]},
        {
            "results": [
                {
                    "title": "Invalid score",
                    "url": "https://example.com",
                    "content": "content",
                    "score": "not-a-number",
                }
            ]
        },
    ],
)
def test_tavily_web_search_rejects_invalid_response_without_retry(payload):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    provider = TavilyWebSearchProvider(
        api_key="secret",
        max_retries=2,
        post=lambda *args, **kwargs: FakeResponse(),
        sleep=lambda _: None,
    )

    with pytest.raises(WebSearchError) as exc_info:
        provider.search("question", 5)

    assert exc_info.value.retry_count == 0


def test_tavily_web_search_retries_timeout_with_exponential_backoff():
    sleeps = []
    calls = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"results": []}

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise requests.Timeout("timeout")
        return FakeResponse()

    provider = TavilyWebSearchProvider(
        api_key="secret",
        max_retries=2,
        post=fake_post,
        sleep=sleeps.append,
    )

    response = provider.search("question", 5)

    assert response.results == []
    assert response.diagnostics["retry_count"] == 2
    assert sleeps == [0.25, 0.5]


def test_web_search_failure_is_distinct_from_empty_results(db_session):
    class FailingWebSearchProvider:
        name = "tavily"

        def search(self, query: str, limit: int):
            raise WebSearchError("search unavailable", retry_count=2)

    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    service = RAGService(
        vector_store=FakeVectorStore({"question": []}),
        model_gateway=FakeModelGateway(answers=[], routes=["web"]),
        web_search_provider=FailingWebSearchProvider(),
        settings=Settings(
            web_search_enabled=True,
            web_search_provider="tavily",
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_detail"] == "web_search_failed"
    message = db.scalar(select(Message).order_by(Message.id.desc()))
    diagnostics = message.metrics["rag_diagnostics"]["web_search"]
    assert diagnostics["failed"] is True
    assert diagnostics["retry_count"] == 2
    assert diagnostics["error_type"] == "WebSearchError"
    audit = db.scalar(select(AuditLog).order_by(AuditLog.id.desc()))
    assert audit.details["web_search_failed"] is True


def test_hybrid_route_refuses_when_web_search_fails(db_session):
    class FailingWebSearchProvider:
        name = "tavily"

        def search(self, query: str, limit: int):
            raise WebSearchError(
                "Tavily returned HTTP 503",
                retry_count=2,
                error_type="HTTPError",
            )

    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "authorized evidence")
    service = RAGService(
        vector_store=FakeVectorStore({"question": [chunk]}),
        model_gateway=FakeModelGateway(answers=[], routes=["hybrid"]),
        reranker=PassThroughReranker(),
        web_search_provider=FailingWebSearchProvider(),
        settings=Settings(
            web_search_enabled=True,
            web_search_provider="tavily",
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_detail"] == "web_search_failed"
    message = db.scalar(select(Message).order_by(Message.id.desc()))
    diagnostics = message.metrics["rag_diagnostics"]["web_search"]
    assert diagnostics["error_type"] == "HTTPError"
    assert diagnostics["result_count"] == 0


def test_web_search_reuses_preprocessed_queries_up_to_configured_limit(db_session):
    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    web_search = FakeWebSearchProvider(
        {
            "original": [],
            "rewrite-one": [
                {
                    "title": "Result",
                    "url": "https://example.com/result",
                    "content": "Relevant result.",
                    "score": 0.9,
                }
            ],
        }
    )
    gateway = FakeModelGateway(
        answers=["Relevant result [1]."],
        relevance=[True],
        rewrites=["rewrite-one", "rewrite-two"],
        routes=["web"],
    )
    service = RAGService(
        vector_store=FakeVectorStore({}),
        model_gateway=gateway,
        reranker=PassThroughReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            web_search_max_queries=2,
            query_rewrite_types="normalize,multi_query",
            query_rewrite_max_queries=3,
        ),
    )

    result = service.answer(db, user, "original")

    assert result["refused"] is False
    assert web_search.calls == [("original", 5), ("rewrite-one", 5)]


def test_web_grading_failure_refuses_without_unverified_evidence(db_session):
    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    web_search = FakeWebSearchProvider(
        {
            "question": [
                {
                    "title": "Unverified",
                    "url": "https://example.com/unverified",
                    "content": "Unverified content.",
                    "score": 0.9,
                }
            ]
        }
    )
    gateway = FakeModelGateway(
        answers=[],
        relevance=[Exception("grader unavailable")],
        routes=["web"],
    )
    service = RAGService(
        vector_store=FakeVectorStore({"question": []}),
        model_gateway=gateway,
        reranker=FailingReranker(),
        web_search_provider=web_search,
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
            reranker_failure_strategy="llm",
        ),
    )

    result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_evidence"
    assert result["citations"] == []


def test_web_provider_failure_returns_structured_refusal(db_session):
    class FailingWebSearchProvider:
        name = "failing"

        def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
            raise TimeoutError("search unavailable")

    db, models = db_session
    user, _, _ = create_rag_user_and_document(db, models)
    gateway = FakeModelGateway(answers=[], routes=["web"])
    service = RAGService(
        vector_store=FakeVectorStore({"question": []}),
        model_gateway=gateway,
        web_search_provider=FailingWebSearchProvider(),
        settings=Settings(
            web_search_enabled=True,
            query_rewrite_types="normalize",
        ),
    )

    result = service.answer(db, user, "question")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_evidence"
    assert result["citations"] == []


def test_citation_schema_accepts_knowledge_base_and_web_sources():
    knowledge = Citation(
        source_type="knowledge_base",
        url=None,
        document_uuid="document-uuid",
        document_title="Policy",
        version=1,
        page_number=2,
        chunk_id="chunk-1",
        excerpt="Authorized evidence.",
    )
    web = Citation(
        source_type="web",
        url="https://example.com/source",
        document_uuid=None,
        document_title="Public source",
        version=None,
        page_number=None,
        chunk_id="web:source",
        excerpt="Public evidence.",
    )

    assert knowledge.document_uuid == "document-uuid"
    assert knowledge.url is None
    assert web.document_uuid is None
    assert web.url == "https://example.com/source"
