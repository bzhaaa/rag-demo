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
from app.rag.rerankers import RerankerError, create_candidate_reranker
from app.rag.types import (
    CandidateReranker,
    QueryPreprocessor,
    RAGModelGateway,
    RAGState,
    WebSearchProvider,
    WebSearchResponse,
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
            "diagnostics": {
                **state.get("diagnostics", {}),
                "queries": queries,
                "query_rewrite_attempted": True,
            },
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
            "diagnostics": {
                **state.get("diagnostics", {}),
                "retrieval": {
                    "candidate_count": len(merge_candidates(candidates)),
                    "candidates": self._candidate_summaries(
                        merge_candidates(candidates)
                    ),
                },
            },
            "timings": timing_with(
                state.get("timings", {}),
                "retrieval",
                time.perf_counter() - started,
            ),
        }

    def _grade_documents(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates, relevant, grading = self._grade_candidates(
            state["question"], state.get("candidates", []), "knowledge"
        )
        diagnostics = state.get("diagnostics", {})
        return {
            **state,
            "candidates": candidates,
            "relevant": relevant[: self.settings.final_context_count],
            "diagnostics": {
                **diagnostics,
                "reranking": {
                    **diagnostics.get("reranking", {}),
                    "knowledge": grading,
                },
            },
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
            "diagnostics": {
                **state.get("diagnostics", {}),
                "evidence_route": route,
                "evidence_routing_failed": failed,
            },
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
        query_count = 0
        failed = False
        retry_count = 0
        error_type = None
        request_ids = []
        provider_response_time = 0.0
        usage_credits = 0.0
        for query in state.get("queries", [])[: self.settings.web_search_max_queries]:
            query_count += 1
            try:
                response = self.web_search_provider.search(
                    query,
                    self.settings.web_search_result_count,
                )
            except Exception as exc:
                failed = True
                error_type = getattr(exc, "error_type", type(exc).__name__)
                retry_count += int(getattr(exc, "retry_count", 0))
                break
            if isinstance(response, WebSearchResponse):
                results.extend(response.results)
                diagnostics = response.diagnostics
                retry_count += int(diagnostics.get("retry_count") or 0)
                if diagnostics.get("request_id"):
                    request_ids.append(diagnostics["request_id"])
                provider_response_time += float(
                    diagnostics.get("provider_response_time") or 0
                )
                usage_credits += float(
                    diagnostics.get("usage_credits") or 0
                )
            else:
                results.extend(response)
        merged_results = [] if failed else merge_web_results(results)
        return {
            **state,
            "web_candidates": merged_results,
            "web_search_attempted": True,
            "web_search_failed": failed,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "web_search": {
                    "attempted": True,
                    "query_count": query_count,
                    "candidate_count": len(merged_results),
                    "result_count": len(merged_results),
                    "provider": self.web_search_provider.name,
                    "failed": failed,
                    "retry_count": retry_count,
                    "error_type": error_type,
                    "request_id": (
                        request_ids[0]
                        if len(request_ids) == 1
                        else request_ids or None
                    ),
                    "provider_response_time": provider_response_time or None,
                    "usage_credits": usage_credits or None,
                },
            },
            "timings": timing_with(
                state.get("timings", {}),
                "web_search",
                time.perf_counter() - started,
            ),
        }

    def _grade_web_documents(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        candidates, relevant, grading = self._grade_candidates(
            state["question"], state.get("web_candidates", []), "web"
        )
        diagnostics = state.get("diagnostics", {})
        return {
            **state,
            "web_candidates": candidates,
            "web_relevant": relevant[: self.settings.final_context_count],
            "diagnostics": {
                **diagnostics,
                "reranking": {
                    **diagnostics.get("reranking", {}),
                    "web": grading,
                },
            },
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
        stage: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], [], {
                "stage": stage,
                "input_count": 0,
                "passed_count": 0,
                "relevant_count": 0,
                "failure_count": 0,
                "skipped": "no_candidates",
                "min_score": self.settings.reranker_min_score,
                "top_k": self.settings.reranker_top_k,
                "results": [],
            }

        try:
            reranked = self.reranker.rerank(question, candidates)
        except Exception as exc:
            return self._handle_reranker_failure(question, candidates, stage, exc)

        return self._admit_reranked_candidates(candidates, reranked, stage)

    def _admit_reranked_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        reranked: Sequence[Dict[str, Any]],
        stage: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        original_candidates = list(candidates)
        ranked = list(reranked)
        relevant = [
            candidate
            for candidate in ranked
            if float(candidate.get("rerank_score") or 0)
            >= self.settings.reranker_min_score
        ][: self.settings.reranker_top_k]
        ranked_by_chunk_id = {
            candidate.get("chunk_id"): candidate
            for candidate in ranked
            if candidate.get("chunk_id")
        }
        passed_chunk_ids = {candidate.get("chunk_id") for candidate in relevant}
        results = []
        for candidate in original_candidates:
            ranked_candidate = ranked_by_chunk_id.get(candidate.get("chunk_id"))
            diagnostic_candidate = ranked_candidate or candidate
            passed = candidate.get("chunk_id") in passed_chunk_ids
            results.append(
                {
                    **self._candidate_summary(diagnostic_candidate),
                    "relevant": bool(passed),
                    "passed": bool(passed),
                }
            )
        self._reset_grading_failures()
        return original_candidates, relevant, {
            "stage": stage,
            "input_count": len(original_candidates),
            "passed_count": len(relevant),
            "relevant_count": len(relevant),
            "failure_count": 0,
            "failed": False,
            "min_score": self.settings.reranker_min_score,
            "top_k": self.settings.reranker_top_k,
            "failure_strategy": None,
            "endpoint": self.settings.reranker_endpoint,
            "model": self.settings.reranker_model,
            "results": results,
        }

    def _handle_reranker_failure(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        stage: str,
        exc: Exception,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        strategy = self.settings.reranker_failure_strategy
        base = {
            "stage": stage,
            "input_count": len(candidates),
            "passed_count": 0,
            "relevant_count": 0,
            "failure_count": len(candidates),
            "failed": True,
            "failure_strategy": strategy,
            "error_type": type(exc).__name__,
            "error": self._truncate(str(exc), 300),
            "min_score": self.settings.reranker_min_score,
            "top_k": self.settings.reranker_top_k,
            "endpoint": self.settings.reranker_endpoint,
            "model": self.settings.reranker_model,
            "results": [],
        }

        if strategy == "reject":
            return list(candidates), [], base
        if strategy == "vector":
            fallback = [
                candidate
                for candidate in sorted(
                    candidates,
                    key=lambda item: float(item.get("score") or 0),
                    reverse=True,
                )
                if float(candidate.get("score") or 0)
                >= float(self.settings.retrieval_min_score or 0)
            ][: self.settings.reranker_top_k]
            results = [
                {
                    **self._candidate_summary(candidate),
                    "relevant": candidate in fallback,
                    "passed": candidate in fallback,
                }
                for candidate in candidates
            ]
            return list(candidates), fallback, {
                **base,
                "passed_count": len(fallback),
                "relevant_count": len(fallback),
                "failure_count": 1,
                "fallback": "vector",
                "results": results,
            }
        if strategy == "llm":
            reranked = list(candidates)
            if self._grading_circuit_is_open():
                return reranked, [], {
                    **base,
                    "skipped": "circuit_open",
                    "fallback": "llm",
                }
            responses = self.model_gateway.grade_relevance(
                question,
                reranked,
                self.settings.relevance_max_concurrency,
            )
            relevant: List[Dict[str, Any]] = []
            failure_count = 0
            grade_results: List[Dict[str, Any]] = []
            for candidate, response in zip(reranked, responses):
                if isinstance(response, Exception):
                    failure_count += 1
                    grade_results.append(
                        {
                            **self._candidate_summary(candidate),
                            "relevant": False,
                            "error_type": type(response).__name__,
                            "error": self._truncate(str(response), 300),
                        }
                    )
                    continue
                is_relevant = (
                    response
                    if isinstance(response, bool)
                    else bool(getattr(response, "relevant", False))
                )
                grade_results.append(
                    {
                        **self._candidate_summary(candidate),
                        "relevant": bool(is_relevant),
                        "raw_response": self._truncate(
                            str(getattr(response, "raw_response", "")),
                            300,
                        ),
                        "parsed_score": getattr(response, "parsed_score", None),
                    }
                )
                if is_relevant:
                    relevant.append(candidate)
            if reranked and failure_count == len(reranked):
                self._record_grading_failure()
            else:
                self._reset_grading_failures()
            return reranked, relevant[: self.settings.reranker_top_k], {
                **base,
                "passed_count": len(relevant[: self.settings.reranker_top_k]),
                "relevant_count": len(relevant[: self.settings.reranker_top_k]),
                "failure_count": failure_count,
                "fallback": "llm",
                "results": grade_results,
            }
        raise RerankerError(f"unsupported reranker failure strategy: {strategy}")

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
            "diagnostics": {
                **state.get("diagnostics", {}),
                "selected_evidence_count": len(evidence),
            },
        }

    def _generate(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        evidence = state.get("evidence", state.get("relevant", []))
        if len(evidence) < self.settings.rag_min_relevant_documents:
            web_enabled = getattr(self.settings, "web_search_enabled", False)
            refusal_detail = self._refusal_detail(state)
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
                "refusal_detail": refusal_detail,
                "diagnostics": {
                    **state.get("diagnostics", {}),
                    "refusal_detail": refusal_detail,
                },
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
                "refusal_detail": "invalid_citations",
                "diagnostics": {
                    **state.get("diagnostics", {}),
                    "refusal_detail": "invalid_citations",
                },
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
            "refusal_detail": None,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "refusal_detail": None,
            },
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
                "diagnostics": {
                    "authorized_version_count": len(version_uuids),
                },
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
                metrics={
                    "timings": result.get("timings", {}),
                    "rag_diagnostics": result.get("diagnostics", {}),
                },
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
                "refusal_detail": result.get("refusal_detail"),
                "citation_count": len(citations),
                "evidence_route": result.get("evidence_route"),
                "web_search_attempted": result.get(
                    "web_search_attempted", False
                ),
                "knowledge_evidence_count": len(result.get("relevant", [])),
                "web_evidence_count": len(result.get("web_relevant", [])),
                "web_search_provider": self.web_search_provider.name,
                "web_search_failed": result.get("web_search_failed", False),
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
            "refusal_detail": result.get("refusal_detail"),
            "trace_id": trace_id,
            "timings": result["timings"],
        }

    def _refusal_detail(self, state: RAGState) -> str:
        diagnostics = state.get("diagnostics", {})
        reranking = diagnostics.get("reranking", {})
        knowledge_reranking = reranking.get("knowledge", {})
        grading = diagnostics.get("grading", {})
        knowledge_grading = grading.get("knowledge", {})

        if state.get("evidence_routing_failed"):
            return "evidence_routing_failed"
        if state.get("web_search_failed"):
            return "web_search_failed"
        if knowledge_reranking.get("failed") and (
            knowledge_reranking.get("failure_strategy") == "reject"
        ):
            return "reranker_failed"
        if "version_uuids" in state and not state.get("version_uuids"):
            return "no_authorized_documents"
        if "queries" in state and not state.get("queries"):
            return "no_retrieval_queries"
        if "candidates" in state and not state.get("candidates"):
            return "no_retrieval_results"
        if knowledge_grading.get("skipped") == "circuit_open":
            return "relevance_grading_circuit_open"
        if (
            knowledge_grading.get("input_count")
            and knowledge_grading.get("failure_count")
            == knowledge_grading.get("input_count")
        ):
            return "relevance_grading_failed"
        if state.get("candidates") and not state.get("relevant"):
            return "no_relevant_evidence"
        if state.get("evidence_route") == "web":
            if not state.get("web_candidates"):
                return "no_web_results"
            if not state.get("web_relevant"):
                return "no_relevant_web_evidence"
        if state.get("evidence_route") == "hybrid":
            return "insufficient_hybrid_evidence"
        if getattr(self.settings, "web_search_enabled", False):
            return "insufficient_evidence"
        return "no_relevant_evidence"

    @classmethod
    def _candidate_summaries(
        cls,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [cls._candidate_summary(candidate) for candidate in candidates]

    @staticmethod
    def _candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chunk_id": candidate.get("chunk_id"),
            "document_uuid": candidate.get("document_uuid"),
            "version_uuid": candidate.get("version_uuid"),
            "source_type": candidate.get("source_type", "knowledge_base"),
            "score": candidate.get("score"),
            "rerank_score": candidate.get("rerank_score"),
        }

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

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
