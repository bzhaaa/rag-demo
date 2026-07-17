from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.rag.contracts.models import PipelineInput
from app.rag.contracts.state import RAGState
from app.rag.modules.generation import LangChainGenerationModule
from app.rag.modules.query import DefaultQueryModule
from app.rag.modules.reranking import RerankingModule
from app.rag.modules.retrieval import KnowledgeRetrievalModule
from app.rag.modules.routing import LLMEvidenceRoutingModule
from app.rag.modules.selection import RouteAwareSelectionModule
from app.rag.modules.validation import BracketCitationValidationModule
from app.rag.modules.web_search import WebSearchModule


class RAGPipeline:
    def __init__(
        self,
        query: DefaultQueryModule,
        retrieval: KnowledgeRetrievalModule,
        reranking: RerankingModule,
        routing: LLMEvidenceRoutingModule,
        web_search: WebSearchModule,
        selection: RouteAwareSelectionModule,
        generation: LangChainGenerationModule,
        validation: BracketCitationValidationModule,
    ) -> None:
        self.query = query
        self.retrieval = retrieval
        self.reranking = reranking
        self.routing = routing
        self.web_search = web_search
        self.selection = selection
        self.generation = generation
        self.validation = validation
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("preprocess", self.query.run)
        workflow.add_node("retrieve", self.retrieval.run)
        workflow.add_node("rerank_knowledge", self.reranking.run_knowledge)
        workflow.add_node("route", self.routing.run)
        workflow.add_node("web_search", self.web_search.run)
        workflow.add_node("rerank_web", self.reranking.run_web)
        workflow.add_node("select_evidence", self.selection.run)
        workflow.add_node("generate", self.generation.run)
        workflow.add_node("validate_citations", self.validation.run)
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "retrieve")
        workflow.add_edge("retrieve", "rerank_knowledge")
        workflow.add_edge("rerank_knowledge", "route")
        workflow.add_conditional_edges(
            "route",
            self.routing.next_node,
            {
                "web_search": "web_search",
                "select_evidence": "select_evidence",
            },
        )
        workflow.add_edge("web_search", "rerank_web")
        workflow.add_edge("rerank_web", "select_evidence")
        workflow.add_edge("select_evidence", "generate")
        workflow.add_edge("generate", "validate_citations")
        workflow.add_edge("validate_citations", END)
        return workflow.compile()

    def invoke(
        self,
        pipeline_input: PipelineInput,
        *,
        config: Dict[str, Any],
    ) -> RAGState:
        return self.graph.invoke(
            {
                "question": pipeline_input.question,
                "version_uuids": pipeline_input.version_uuids,
                "timings": {},
                "diagnostics": {
                    "authorized_version_count": (
                        pipeline_input.authorized_version_count
                    ),
                },
            },
            config=config,
        )
