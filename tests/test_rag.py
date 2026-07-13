from typing import Any, Dict, List, Sequence

from sqlalchemy import select

from app.config import Settings
from app.models import Message
from app.rag import (
    DefaultCandidateReranker,
    DefaultQueryPreprocessor,
    ExternalCandidateReranker,
    IdentityCandidateReranker,
    RAGService,
    create_candidate_reranker,
)


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
    ) -> None:
        self.answers = list(answers)
        self.relevance = list(relevance)
        self.rewrites = list(rewrites)
        self.rewrite_batches = [list(batch) for batch in rewrite_batches]
        self.direct_rewrite = direct_rewrite
        self.hyde_document = hyde_document
        self.step_back_query = step_back_query
        self.generate_calls: List[bool] = []
        self.rewrite_calls = 0
        self.standalone_calls = 0
        self.hyde_calls = 0
        self.step_back_calls = 0

    def grade_relevance(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        max_concurrency: int,
    ) -> List[object]:
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
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "RAG 如何回答？")

    assert vector_store.calls[0][1] == [version.uuid]
    assert result["answer"] == "RAG 只能根据授权证据回答。[1]"
    assert [item["chunk_id"] for item in result["citations"]] == [chunk["chunk_id"]]
    assert result["timings"]["total"] >= 0
    messages = list(db.scalars(select(Message).order_by(Message.id)))
    assert messages[-1].metrics == result["timings"]


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
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "   ")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
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
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

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
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "无效引用")

    assert result["refused"] is True
    assert result["refusal_reason"] == "invalid_citations"
    assert result["citations"] == []


def test_corrective_rag_rewrites_once_when_initial_evidence_is_insufficient(
    db_session,
):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    first = make_chunk(document, version, 0, "无关内容。", score=0.2)
    rewritten = make_chunk(document, version, 1, "改写查询找到的授权证据。", score=0.95)
    vector_store = FakeVectorStore(
        {
            "原始问题": [first],
            "初检无关": [first],
            "改写一": [rewritten],
            "改写二": [rewritten],
        }
    )
    gateway = FakeModelGateway(
        ["改写查询找到了答案。[1]"],
        relevance=[False, True],
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
        query_preprocessor=preprocessor,
    )

    result = service.answer(db, user, "原始问题")

    assert result["refused"] is False
    assert gateway.rewrite_calls == 2
    assert [call[0] for call in vector_store.calls] == [
        "原始问题",
        "初检无关",
        "改写一",
        "改写二",
    ]
    assert all(call[1] == [version.uuid] for call in vector_store.calls)
    assert [item["chunk_id"] for item in result["citations"]] == [
        rewritten["chunk_id"]
    ]
    assert "query_rewrite" in result["timings"]
    assert "corrective_retrieval" in result["timings"]


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


def test_corrective_rag_does_not_rewrite_when_initial_evidence_is_sufficient(
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
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "原始问题")

    assert result["refused"] is False
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == ["原始问题", "不应使用"]


def test_grading_failures_open_circuit_and_refuse_conservatively(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "看起来相关的证据。")
    service = RAGService(
        vector_store=FakeVectorStore({"问题": [chunk]}),
        model_gateway=FakeModelGateway(
            answers=[],
            relevance=[Exception("model down") for _ in range(6)],
            rewrites=[],
        ),
    )

    for _ in range(3):
        result = service.answer(db, user, "问题")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert service._grading_circuit_is_open()


def test_failed_query_rewrite_keeps_insufficient_evidence_refusal(db_session):
    db, models = db_session
    user, document, version = create_rag_user_and_document(db, models)
    chunk = make_chunk(document, version, 0, "无关内容。")
    vector_store = FakeVectorStore({"问题": [chunk]})
    gateway = FakeModelGateway(
        answers=[],
        relevance=[False],
        rewrites=[],
    )
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "问题")

    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert gateway.rewrite_calls == 2
    assert [call[0] for call in vector_store.calls] == ["问题"]
