from typing import Any, Dict, List, Sequence

from sqlalchemy import select

from app.models import Message
from app.rag import DefaultCandidateReranker, RAGService


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
    ) -> None:
        self.answers = list(answers)
        self.relevance = list(relevance)
        self.rewrites = list(rewrites)
        self.generate_calls: List[bool] = []
        self.rewrite_calls = 0

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
        return list(self.rewrites[:count])


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

    result = reranker.rerank(candidates)

    assert [item["chunk_id"] for item in result] == [
        "a-duplicate-high",
        "b-one",
        "a-third",
    ]


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
            "改写一": [rewritten],
            "改写二": [rewritten],
        }
    )
    gateway = FakeModelGateway(
        ["改写查询找到了答案。[1]"],
        relevance=[False, True],
        rewrites=["改写一", "改写二"],
    )
    service = RAGService(vector_store=vector_store, model_gateway=gateway)

    result = service.answer(db, user, "原始问题")

    assert result["refused"] is False
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == [
        "原始问题",
        "改写一",
        "改写二",
    ]
    assert all(call[1] == [version.uuid] for call in vector_store.calls)
    assert [item["chunk_id"] for item in result["citations"]] == [
        rewritten["chunk_id"]
    ]
    assert "query_rewrite" in result["timings"]
    assert "corrective_retrieval" in result["timings"]


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
    assert gateway.rewrite_calls == 0
    assert [call[0] for call in vector_store.calls] == ["原始问题"]


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
    assert gateway.rewrite_calls == 1
    assert [call[0] for call in vector_store.calls] == ["问题"]
