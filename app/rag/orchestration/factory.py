from typing import Any, Optional

from app.config import Settings
from app.rag.contracts.protocols import (
    CandidateReranker,
    QueryPreprocessor,
    RAGModelGateway,
    WebSearchProvider,
)
from app.rag.modules.reranking import RerankingModule
from app.rag.modules.web_search import WebSearchModule
from app.rag.orchestration.pipeline import RAGPipeline
from app.rag.orchestration.registry import ModuleRegistry, create_default_registry
from app.rag.preprocessors import create_query_preprocessor
from app.rag.providers.milvus import create_knowledge_retriever
from app.rag.rerankers import create_candidate_reranker
from app.rag.web_search import create_web_search_provider


class RAGPipelineFactory:
    def __init__(
        self,
        settings: Settings,
        registry: Optional[ModuleRegistry] = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or create_default_registry()

    def validate_module_configuration(self) -> None:
        selections = {
            "query": self.settings.rag_query_module,
            "retriever": self.settings.rag_retriever_module,
            "fusion": self.settings.rag_fusion_module,
            "router": self.settings.rag_router_module,
            "selector": self.settings.rag_selector_module,
            "generator": self.settings.rag_generator_module,
            "validator": self.settings.rag_validator_module,
        }
        for category, name in selections.items():
            self.registry.resolve(category, name)

    def create(
        self,
        *,
        vector_store: Any,
        model_gateway: RAGModelGateway,
        reranker: Optional[CandidateReranker] = None,
        query_preprocessor: Optional[QueryPreprocessor] = None,
        web_search_provider: Optional[WebSearchProvider] = None,
    ) -> RAGPipeline:
        self.validate_module_configuration()
        query_factory = self.registry.resolve(
            "query", self.settings.rag_query_module
        )
        retrieval_factory = self.registry.resolve(
            "retriever", self.settings.rag_retriever_module
        )
        fusion_factory = self.registry.resolve(
            "fusion", self.settings.rag_fusion_module
        )
        routing_factory = self.registry.resolve(
            "router", self.settings.rag_router_module
        )
        selection_factory = self.registry.resolve(
            "selector", self.settings.rag_selector_module
        )
        generation_factory = self.registry.resolve(
            "generator", self.settings.rag_generator_module
        )
        validation_factory = self.registry.resolve(
            "validator", self.settings.rag_validator_module
        )

        preprocessor = query_preprocessor or create_query_preprocessor(
            self.settings
        )
        fusion = fusion_factory(self.settings.retrieval_rrf_k)
        retriever = create_knowledge_retriever(vector_store, self.settings)
        provider = web_search_provider or create_web_search_provider(
            self.settings
        )
        candidate_reranker = reranker or create_candidate_reranker(
            self.settings
        )
        return RAGPipeline(
            query=query_factory(preprocessor, model_gateway, self.settings),
            retrieval=retrieval_factory(
                retriever,
                fusion,
                self.settings,
            ),
            reranking=RerankingModule(
                candidate_reranker,
                model_gateway,
                self.settings,
            ),
            routing=routing_factory(
                model_gateway,
                self.settings.web_search_enabled,
            ),
            web_search=WebSearchModule(provider, self.settings),
            selection=selection_factory(self.settings),
            generation=generation_factory(model_gateway, self.settings),
            validation=validation_factory(model_gateway, self.settings),
        )
