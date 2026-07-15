import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    Conversation,
    Document,
    Message,
    MessageRole,
    User,
)
from app.rag.model_gateway import LangChainRAGModelGateway
from app.rag.preprocessors import create_query_preprocessor
from app.rag.rerankers import create_candidate_reranker
from app.rag.types import (
    CandidateReranker,
    QueryPreprocessor,
    RAGModelGateway,
    RAGState,
)
from app.rag.utils import merge_candidates, timing_with, valid_citation_indices
from app.repositories import active_versions_for_user
from app.services import write_audit
from app.vector_store import MilvusChunkStore


class RAGService:
    def __init__(
        self,
        vector_store: Optional[MilvusChunkStore] = None,
        model_gateway: Optional[RAGModelGateway] = None,
        reranker: Optional[CandidateReranker] = None,
        query_preprocessor: Optional[QueryPreprocessor] = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or MilvusChunkStore()
        self.model_gateway = model_gateway or LangChainRAGModelGateway()
        self.reranker = reranker or create_candidate_reranker(self.settings)
        self.query_preprocessor = query_preprocessor or create_query_preprocessor(
            self.settings
        )
        self._grading_lock = threading.Lock()
        self._grading_failures = 0
        self._grading_circuit_open_until = 0.0
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("preprocess_query", self._preprocess_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("rewrite_and_retrieve", self._rewrite_and_retrieve)
        workflow.add_node("generate", self._generate)
        workflow.set_entry_point("preprocess_query")
        workflow.add_edge("preprocess_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._after_grading,
            {
                "rewrite_and_retrieve": "rewrite_and_retrieve",
                "generate": "generate",
            },
        )
        workflow.add_edge("rewrite_and_retrieve", "grade_documents")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def _preprocess_query(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        queries = self.query_preprocessor.preprocess(
            state["question"],
            self.model_gateway,
            self.settings.query_rewrite_max_queries,
        )
        return {
            **state,
            "queries": queries,
            "query_rewrite_attempted": True,
            "timings": timing_with(
                state.get("timings", {}),
                "query_preprocess",
                time.perf_counter() - started,
            ),
        }

    def _retrieve(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates: List[Dict[str, Any]] = []
        for query in state.get("queries", []):
            candidates.extend(
                self.vector_store.search(
                    query,
                    state.get("version_uuids", []),
                    self.settings.retrieval_candidate_count,
                )
            )
        return {
            **state,
            "candidates": merge_candidates(candidates),
            "timings": {
                **state.get("timings", {}),
                "retrieval": time.perf_counter() - started,
            },
        }

    def _grade_documents(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates = self.reranker.rerank(
            state["question"], state.get("candidates", [])
        )
        if self._grading_circuit_is_open():
            return {
                **state,
                "relevant": [],
                "timings": timing_with(
                    state.get("timings", {}),
                    "grading",
                    time.perf_counter() - started,
                ),
            }
        if not candidates:
            return {
                **state,
                "relevant": [],
                "timings": timing_with(
                    state.get("timings", {}),
                    "grading",
                    time.perf_counter() - started,
                ),
            }

        if self.settings.relevance_grading_enabled:
            responses = self.model_gateway.grade_relevance(
                state["question"],
                candidates,
                self.settings.relevance_max_concurrency,
            )
        else:
            responses = [True] * len(candidates)

        relevant: List[Dict[str, Any]] = []
        failure_count = 0
        for candidate, response in zip(candidates, responses):
            if isinstance(response, Exception):
                failure_count += 1
                continue
            if response:
                relevant.append(candidate)
        if failure_count == len(candidates):
            self._record_grading_failure()
        else:
            self._reset_grading_failures()
        grading_timing = time.perf_counter() - started
        timings = state.get("timings", {})
        return {
            **state,
            "candidates": candidates,
            "relevant": relevant[: self.settings.final_context_count],
            "timings": {
                **timings,
                "grading": timings.get("grading", 0) + grading_timing,
            },
        }

    def _after_grading(self, state: RAGState) -> str:
        if len(state.get("relevant", [])) >= self.settings.rag_min_relevant_documents:
            return "generate"
        if state.get("corrective_attempted"):
            return "generate"
        return "rewrite_and_retrieve"

    def _rewrite_and_retrieve(self, state: RAGState) -> RAGState:
        rewrite_started = time.perf_counter()
        rewritten_queries = self.query_preprocessor.preprocess(
            state["question"],
            self.model_gateway,
            self.settings.query_rewrite_max_queries
            + self.settings.query_rewrite_corrective_max_queries,
        )
        previous_queries = set(state.get("queries", []))
        rewrites = [
            query
            for query in rewritten_queries
            if query not in previous_queries
        ][: self.settings.query_rewrite_corrective_max_queries]
        query_rewrite_timing = time.perf_counter() - rewrite_started
        if not rewrites:
            timings = state.get("timings", {})
            return {
                **state,
                "candidates": [],
                "corrective_attempted": True,
                "timings": {
                    **timings,
                    "query_rewrite": timings.get("query_rewrite", 0)
                    + query_rewrite_timing,
                    "corrective_retrieval": timings.get(
                        "corrective_retrieval", 0
                    ),
                },
            }
        retrieval_started = time.perf_counter()
        corrective_candidates: List[Dict[str, Any]] = []
        for query in rewrites:
            corrective_candidates.extend(
                self.vector_store.search(
                    query,
                    state.get("version_uuids", []),
                    self.settings.retrieval_candidate_count,
                )
            )
        merged = merge_candidates(
            [*state.get("candidates", []), *corrective_candidates]
        )
        corrective_retrieval_timing = time.perf_counter() - retrieval_started
        timings = state.get("timings", {})
        return {
            **state,
            "candidates": merged,
            "corrective_attempted": True,
            "timings": {
                **timings,
                "query_rewrite": timings.get("query_rewrite", 0)
                + query_rewrite_timing,
                "corrective_retrieval": timings.get("corrective_retrieval", 0)
                + corrective_retrieval_timing,
            },
        }

    def _generate(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        relevant = state.get("relevant", [])
        if len(relevant) < self.settings.rag_min_relevant_documents:
            return {
                **state,
                "answer": "当前知识库中没有足够且经过授权的证据回答这个问题。",
                "refused": True,
                "refusal_reason": "insufficient_authorized_evidence",
                "timings": {
                    **state.get("timings", {}),
                    "generation": time.perf_counter() - started,
                },
            }

        answer = self.model_gateway.generate_answer(
            state["question"],
            relevant,
        )
        validation_started = time.perf_counter()
        cited_indices = valid_citation_indices(answer, len(relevant))
        retry_count = 0
        while not cited_indices and retry_count < self.settings.rag_citation_retry_count:
            retry_count += 1
            answer = self.model_gateway.generate_answer(
                state["question"],
                relevant,
                strict_citations=True,
            )
            cited_indices = valid_citation_indices(answer, len(relevant))
        validation_timing = time.perf_counter() - validation_started
        if not cited_indices:
            return {
                **state,
                "answer": "生成答案缺少有效引用，无法确认依据来自授权证据。",
                "cited_indices": [],
                "refused": True,
                "refusal_reason": "invalid_citations",
                "timings": {
                    **state.get("timings", {}),
                    "generation": time.perf_counter() - started - validation_timing,
                    "citation_validation": validation_timing,
                },
            }
        return {
            **state,
            "answer": answer,
            "cited_indices": cited_indices,
            "refused": False,
            "refusal_reason": None,
            "timings": {
                **state.get("timings", {}),
                "generation": time.perf_counter() - started - validation_timing,
                "citation_validation": validation_timing,
            },
        }

    def answer(
        self,
        db: Session,
        user: User,
        question: str,
        conversation_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        total_started = time.perf_counter()
        versions = list(active_versions_for_user(db, user))
        version_uuids = [version.uuid for version in versions]
        run_id = uuid.uuid4()
        trace_id = str(run_id)
        result = self.graph.invoke(
            {
                "question": question,
                "version_uuids": version_uuids,
                "timings": {},
            },
            config={
                "run_name": "enterprise-corrective-rag",
                "run_id": run_id,
                "tags": ["enterprise-rag", "authorized-retrieval"],
                "metadata": {
                    "trace_id": trace_id,
                    "user_uuid": user.uuid,
                    "authorized_version_count": len(version_uuids),
                },
            },
        )
        citations = self._citations(
            db,
            self._cited_chunks(
                result.get("relevant", []),
                result.get("cited_indices", []),
            ),
        )
        conversation = self._conversation(
            db, user, conversation_uuid, question
        )
        result["timings"]["total"] = time.perf_counter() - total_started
        db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.user.value,
                content=question,
            )
        )
        db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.assistant.value,
                content=result["answer"],
                citations=citations,
                model_name=self.settings.llm_model,
                trace_id=trace_id,
                metrics=result.get("timings", {}),
            )
        )
        write_audit(
            db,
            "query.execute",
            "conversation",
            user,
            conversation.uuid,
            {
                "trace_id": trace_id,
                "refused": result.get("refused", False),
                "citation_count": len(citations),
            },
        )
        conversation.updated_at = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        db.commit()
        return {
            "conversation_uuid": conversation.uuid,
            "answer": result["answer"],
            "citations": citations,
            "refused": result.get("refused", False),
            "refusal_reason": result.get("refusal_reason"),
            "trace_id": trace_id,
            "timings": result["timings"],
        }

    @staticmethod
    def _conversation(
        db: Session,
        user: User,
        conversation_uuid: Optional[str],
        question: str,
    ) -> Conversation:
        conversation = None
        if conversation_uuid:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.uuid == conversation_uuid,
                    Conversation.user_id == user.id,
                )
            )
        if conversation is None:
            conversation = Conversation(
                user_id=user.id, title=question.strip()[:120]
            )
            db.add(conversation)
            db.flush()
        return conversation

    @staticmethod
    def _citations(
        db: Session, chunks: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        document_uuids = {item["document_uuid"] for item in chunks}
        documents = {
            document.uuid: document
            for document in db.scalars(
                select(Document).where(Document.uuid.in_(document_uuids))
            )
        }
        return [
            {
                "document_uuid": item["document_uuid"],
                "document_title": documents[item["document_uuid"]].title,
                "version": item["version_number"],
                "page_number": item.get("page_number") or None,
                "chunk_id": item["chunk_id"],
                "excerpt": item["content"][:300],
            }
            for item in chunks
            if item["document_uuid"] in documents
        ]

    @staticmethod
    def _cited_chunks(
        chunks: Sequence[Dict[str, Any]],
        cited_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        if not cited_indices:
            return []
        result = []
        for index in cited_indices:
            position = index - 1
            if 0 <= position < len(chunks):
                result.append(chunks[position])
        return result

    def _grading_circuit_is_open(self) -> bool:
        with self._grading_lock:
            return time.monotonic() < self._grading_circuit_open_until

    def _record_grading_failure(self) -> None:
        with self._grading_lock:
            self._grading_failures += 1
            if self._grading_failures >= 3:
                self._grading_circuit_open_until = time.monotonic() + 30

    def _reset_grading_failures(self) -> None:
        with self._grading_lock:
            self._grading_failures = 0
            self._grading_circuit_open_until = 0.0


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()
