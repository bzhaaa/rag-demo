from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, cast

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import User
from app.rag.application.citations import assemble_citations, cited_chunks
from app.rag.application.service import RAGApplicationService
from app.rag.contracts.protocols import (
    AnswerGenerator,
    CandidateReranker,
    QueryPreprocessor,
    RAGModelGateway,
    WebSearchProvider,
)
from app.rag.contracts.state import RAGState
from app.rag.model_gateway import LangChainRAGModelGateway
from app.rag.modules.generation import refusal_detail
from app.rag.orchestration.factory import RAGPipelineFactory
from app.rag.preprocessors import create_query_preprocessor
from app.rag.rerankers import create_candidate_reranker
from app.rag.web_search import create_web_search_provider
from app.vector_store import MilvusChunkStore


class RAGService:
    """Compatibility facade around the modular query pipeline."""

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
        self.query_preprocessor = (
            query_preprocessor or create_query_preprocessor(self.settings)
        )
        self.web_search_provider = (
            web_search_provider or create_web_search_provider(self.settings)
        )
        self.pipeline = RAGPipelineFactory(self.settings).create(
            vector_store=self.vector_store,
            model_gateway=self.model_gateway,
            reranker=self.reranker,
            query_preprocessor=self.query_preprocessor,
            web_search_provider=self.web_search_provider,
        )
        self.graph = self.pipeline.graph
        self.application = RAGApplicationService(
            self.pipeline,
            self.settings,
            self.web_search_provider.name,
        )

    def answer(
        self,
        db: Session,
        user: User,
        question: str,
        conversation_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.application.answer(
            db,
            user,
            question,
            conversation_uuid,
        )

    def _generate(self, state: RAGState) -> RAGState:
        """Legacy test hook; new code should use the pipeline modules."""
        if not hasattr(self, "pipeline"):
            gateway = getattr(self, "model_gateway", None)
            if gateway is None and state.get("evidence", state.get("relevant", [])):
                gateway = LangChainRAGModelGateway()
            from app.rag.modules.generation import LangChainGenerationModule
            from app.rag.modules.validation import (
                BracketCitationValidationModule,
            )

            generated = LangChainGenerationModule(
                cast(AnswerGenerator, gateway),
                self.settings,
            ).run(state)
            if generated.get("refused"):
                return generated
            return BracketCitationValidationModule(
                cast(AnswerGenerator, gateway),
                self.settings,
            ).run(generated)
        generated = self.pipeline.generation.run(state)
        if generated.get("refused"):
            return generated
        return self.pipeline.validation.run(generated)

    def _grading_circuit_is_open(self) -> bool:
        return self.pipeline.reranking.grading_circuit_is_open()

    def _record_grading_failure(self) -> None:
        self.pipeline.reranking.record_grading_failure()

    def _reset_grading_failures(self) -> None:
        self.pipeline.reranking.reset_grading_failures()

    def _refusal_detail(self, state: RAGState) -> str:
        return refusal_detail(state, self.settings.web_search_enabled)

    @staticmethod
    def _citations(
        db: Session, chunks: Sequence[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        return assemble_citations(db, chunks)

    @staticmethod
    def _cited_chunks(
        chunks: Sequence[Dict[str, Any]],
        cited_indices: Sequence[int],
    ) -> list[Dict[str, Any]]:
        return cited_chunks(chunks, cited_indices)


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()
