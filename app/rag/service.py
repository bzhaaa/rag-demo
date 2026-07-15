import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Conversation, Document, Message, MessageRole, User
from app.rag.model_gateway import LangChainRAGModelGateway
from app.rag.preprocessors import create_query_preprocessor
from app.rag.rerankers import create_candidate_reranker
from app.rag.types import (
    CandidateReranker,
    QueryPreprocessor,
    RAGModelGateway,
    RAGState,
    WebSearchProvider,
)
from app.rag.utils import merge_candidates, timing_with, valid_citation_indices
from app.rag.web_search import create_web_search_provider, merge_web_results
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
        web_search_provider: Optional[WebSearchProvider] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or MilvusChunkStore()
        self.model_gateway = model_gateway or LangChainRAGModelGateway()
        self.reranker = reranker or create_candidate_reranker(self.settings)
        self.query_preprocessor = query_preprocessor or create_query_preprocessor(
            self.settings
        )
        self.web_search_provider = (
            web_search_provider or create_web_search_provider(self.settings)
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
        workflow.add_node("route_evidence", self._route_evidence)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("grade_web_documents", self._grade_web_documents)
        workflow.add_node("select_evidence", self._select_evidence)
        workflow.add_node("generate", self._generate)
        workflow.set_entry_point("preprocess_query")
        workflow.add_edge("preprocess_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._after_knowledge_grading,
            {
                "route_evidence": "route_evidence",
                "select_evidence": "select_evidence",
            },
        )
        workflow.add_conditional_edges(
            "route_evidence",
            self._after_routing,
            {
                "web_search": "web_search",
                "select_evidence": "select_evidence",
            },
        )
        workflow.add_edge("web_search", "grade_web_documents")
        workflow.add_edge("grade_web_documents", "select_evidence")
        workflow.add_edge("select_evidence", "generate")
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
            "timings": timing_with(
                state.get("timings", {}),
                "retrieval",
                time.perf_counter() - started,
            ),
        }

    def _grade_documents(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates, relevant = self._grade_candidates(
            state["question"], state.get("candidates", [])
        )
        return {
            **state,
            "candidates": candidates,
            "relevant": relevant[: self.settings.final_context_count],
            "timings": timing_with(
                state.get("timings", {}),
                "grading",
                time.perf_counter() - started,
            ),
        }

    def _after_knowledge_grading(self, state: RAGState) -> str:
        if self.settings.web_search_enabled:
            return "route_evidence"
        return "select_evidence"

    def _route_evidence(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        failed = False
        try:
            route = self.model_gateway.route_evidence(
                state["question"],
                state.get("relevant", []),
            )
            if route not in {"knowledge_base", "web", "hybrid"}:
                raise ValueError("invalid evidence route")
        except Exception:
            route = ""
            failed = True
        return {
            **state,
            "evidence_route": route,
            "evidence_routing_failed": failed,
            "timings": timing_with(
                state.get("timings", {}),
                "evidence_routing",
                time.perf_counter() - started,
            ),
        }

    @staticmethod
    def _after_routing(state: RAGState) -> str:
        if state.get("evidence_route") in {"web", "hybrid"}:
            return "web_search"
        return "select_evidence"

    def _web_search(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for query in state.get("queries", [])[: self.settings.web_search_max_queries]:
            try:
                results.extend(
                    self.web_search_provider.search(
                        query,
                        self.settings.web_search_result_count,
                    )
                )
            except Exception:
                continue
        return {
            **state,
            "web_candidates": merge_web_results(results),
            "web_search_attempted": True,
            "timings": timing_with(
                state.get("timings", {}),
                "web_search",
                time.perf_counter() - started,
            ),
        }

    def _grade_web_documents(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates, relevant = self._grade_candidates(
            state["question"], state.get("web_candidates", [])
        )
        return {
            **state,
            "web_candidates": candidates,
            "web_relevant": relevant[: self.settings.final_context_count],
            "timings": timing_with(
                state.get("timings", {}),
                "web_grading",
                time.perf_counter() - started,
            ),
        }

    def _grade_candidates(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        reranked = self.reranker.rerank(question, candidates)
        if self._grading_circuit_is_open() or not reranked:
            return reranked, []
        if self.settings.relevance_grading_enabled:
            responses = self.model_gateway.grade_relevance(
                question,
                reranked,
                self.settings.relevance_max_concurrency,
            )
        else:
            responses = [True] * len(reranked)

        relevant: List[Dict[str, Any]] = []
        failure_count = 0
        for candidate, response in zip(reranked, responses):
            if isinstance(response, Exception):
                failure_count += 1
                continue
            if response:
                relevant.append(candidate)
        if reranked and failure_count == len(reranked):
            self._record_grading_failure()
        else:
            self._reset_grading_failures()
        return reranked, relevant

    def _select_evidence(self, state: RAGState) -> RAGState:
        knowledge = state.get("relevant", [])
        web = state.get("web_relevant", [])
        minimum = self.settings.rag_min_relevant_documents

        if not self.settings.web_search_enabled:
            evidence = knowledge if len(knowledge) >= minimum else []
            route = "knowledge_base"
        else:
            route = state.get("evidence_route", "")
            if state.get("evidence_routing_failed"):
                evidence = []
            elif route == "knowledge_base":
                evidence = knowledge if len(knowledge) >= minimum else []
            elif route == "web":
                evidence = web if len(web) >= minimum else []
            elif route == "hybrid" and knowledge and web:
                evidence = [*knowledge, *web][: self.settings.final_context_count]
            else:
                evidence = []
        return {
            **state,
            "evidence_route": route,
            "evidence": evidence,
        }

    def _generate(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        evidence = state.get("evidence", state.get("relevant", []))
        if len(evidence) < self.settings.rag_min_relevant_documents:
            web_enabled = getattr(self.settings, "web_search_enabled", False)
            return {
                **state,
                "answer": (
                    "当前知识库和外部来源中没有足够证据回答这个问题。"
                    if web_enabled
                    else "当前知识库中没有足够且经过授权的证据回答这个问题。"
                ),
                "cited_indices": [],
                "refused": True,
                "refusal_reason": (
                    "insufficient_evidence"
                    if web_enabled
                    else "insufficient_authorized_evidence"
                ),
                "timings": timing_with(
                    state.get("timings", {}),
                    "generation",
                    time.perf_counter() - started,
                ),
            }

        answer = self.model_gateway.generate_answer(state["question"], evidence)
        validation_started = time.perf_counter()
        cited_indices = valid_citation_indices(answer, len(evidence))
        retry_count = 0
        while (
            not cited_indices
            and retry_count < self.settings.rag_citation_retry_count
        ):
            retry_count += 1
            answer = self.model_gateway.generate_answer(
                state["question"],
                evidence,
                strict_citations=True,
            )
            cited_indices = valid_citation_indices(answer, len(evidence))
        validation_timing = time.perf_counter() - validation_started
        generation_timing = time.perf_counter() - started - validation_timing
        if not cited_indices:
            return {
                **state,
                "answer": "生成答案缺少有效引用，无法确认其证据来源。",
                "cited_indices": [],
                "refused": True,
                "refusal_reason": "invalid_citations",
                "timings": {
                    **state.get("timings", {}),
                    "generation": generation_timing,
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
                "generation": generation_timing,
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
                "run_name": "enterprise-crag",
                "run_id": run_id,
                "tags": ["enterprise-rag", "crag", "authorized-retrieval"],
                "metadata": {
                    "trace_id": trace_id,
                    "user_uuid": user.uuid,
                    "authorized_version_count": len(version_uuids),
                },
            },
        )
        cited_evidence = self._cited_chunks(
            result.get("evidence", result.get("relevant", [])),
            result.get("cited_indices", []),
        )
        citations = self._citations(db, cited_evidence)
        conversation = self._conversation(db, user, conversation_uuid, question)
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
                "evidence_route": result.get("evidence_route"),
                "web_search_attempted": result.get(
                    "web_search_attempted", False
                ),
                "knowledge_evidence_count": len(result.get("relevant", [])),
                "web_evidence_count": len(result.get("web_relevant", [])),
                "web_search_provider": self.web_search_provider.name,
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
        document_uuids = {
            item["document_uuid"]
            for item in chunks
            if item.get("source_type") != "web" and item.get("document_uuid")
        }
        documents = (
            {
                document.uuid: document
                for document in db.scalars(
                    select(Document).where(Document.uuid.in_(document_uuids))
                )
            }
            if document_uuids
            else {}
        )
        citations: List[Dict[str, Any]] = []
        for item in chunks:
            if item.get("source_type") == "web":
                citations.append(
                    {
                        "source_type": "web",
                        "url": item.get("url"),
                        "document_uuid": None,
                        "document_title": item.get("source_name")
                        or item.get("title")
                        or "Web source",
                        "version": None,
                        "page_number": None,
                        "chunk_id": item["chunk_id"],
                        "excerpt": item["content"][:300],
                    }
                )
                continue
            document = documents.get(item.get("document_uuid"))
            if document is None:
                continue
            citations.append(
                {
                    "source_type": "knowledge_base",
                    "url": None,
                    "document_uuid": item["document_uuid"],
                    "document_title": document.title,
                    "version": item["version_number"],
                    "page_number": item.get("page_number") or None,
                    "chunk_id": item["chunk_id"],
                    "excerpt": item["content"][:300],
                }
            )
        return citations

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
