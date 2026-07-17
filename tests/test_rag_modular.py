from typing import Any, Dict, List, Sequence

import pytest

from app.config import Settings
from app.rag.contracts.models import Candidate
from app.rag.modules.fusion import RRFFusionModule
from app.rag.orchestration.factory import RAGPipelineFactory
from app.rag.orchestration.registry import (
    ModuleRegistry,
    UnknownModuleError,
    create_default_registry,
)


class FakeVectorStore:
    def search(
        self, question: str, version_uuids: Sequence[str], limit: int
    ) -> List[Dict[str, Any]]:
        return []


class FakeGateway:
    def rewrite_queries(self, question: str, count: int = 2) -> List[str]:
        return []

    def rewrite_query(self, question: str) -> str:
        return question

    def generate_hypothetical_document(self, question: str) -> str:
        return question

    def rewrite_step_back_query(self, question: str) -> str:
        return question

    def route_evidence(
        self, question: str, evidence: Sequence[Dict[str, Any]]
    ) -> str:
        return "knowledge_base"

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        strict_citations: bool = False,
    ) -> str:
        return "answer [1]"

    def grade_relevance(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        max_concurrency: int,
    ) -> List[bool]:
        return [True] * len(candidates)


class EmptyReranker:
    def rerank(
        self, question: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return []


class DisabledWebSearch:
    name = "disabled"

    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        return []


def test_candidate_mapping_round_trip_preserves_provider_fields():
    candidate = Candidate.from_mapping(
        {
            "chunk_id": "chunk-1",
            "content": "policy",
            "score": 0.8,
            "dense_score": 0.7,
            "sparse_score": 4.2,
            "rerank_score": 0.9,
            "retrieval_sources": ["dense", "sparse"],
            "provider_metadata": {"rank": 1},
        }
    )

    mapping = candidate.to_mapping()

    assert mapping["provider_metadata"] == {"rank": 1}
    assert mapping["retrieval_sources"] == ["dense", "sparse"]
    assert mapping["rerank_score"] == 0.9


def test_rrf_fusion_handles_single_and_dual_retrieval_channels():
    fusion = RRFFusionModule(rrf_k=60)
    dense = Candidate.from_mapping(
        {
            "chunk_id": "dense",
            "content": "semantic",
            "score": 0.9,
            "dense_score": 0.9,
            "retrieval_sources": ["dense"],
        }
    )
    shared_dense = Candidate.from_mapping(
        {
            "chunk_id": "shared",
            "content": "shared",
            "score": 0.7,
            "dense_score": 0.7,
            "retrieval_sources": ["dense"],
        }
    )
    shared_sparse = Candidate.from_mapping(
        {
            "chunk_id": "shared",
            "content": "shared",
            "score": 8.0,
            "sparse_score": 8.0,
            "retrieval_sources": ["sparse"],
        }
    )

    assert fusion.fuse({"dense": [dense]}, 10)[0].score == 0.9
    hybrid = fusion.fuse(
        {"dense": [dense, shared_dense], "sparse": [shared_sparse]},
        10,
    )
    assert hybrid[0].chunk_id == "shared"
    assert hybrid[0].retrieval_sources == ["dense", "sparse"]


def test_registry_rejects_unknown_module_with_category_and_name():
    registry = create_default_registry()

    with pytest.raises(
        UnknownModuleError,
        match="unknown RAG router module 'rules'",
    ):
        registry.resolve("router", "rules")


def test_registry_supports_explicit_whitelist_replacement():
    registry = ModuleRegistry()

    def factory():
        return object()

    registry.register("query", "custom", factory)

    assert registry.resolve("query", "custom") is factory


def test_pipeline_factory_builds_fixed_modular_topology():
    settings = Settings(
        web_search_enabled=False,
        query_rewrite_types="normalize",
    )
    pipeline = RAGPipelineFactory(settings).create(
        vector_store=FakeVectorStore(),
        model_gateway=FakeGateway(),
        reranker=EmptyReranker(),
        web_search_provider=DisabledWebSearch(),
    )

    graph = pipeline.graph.get_graph()
    nodes = set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert {
        "preprocess",
        "retrieve",
        "rerank_knowledge",
        "route",
        "web_search",
        "rerank_web",
        "select_evidence",
        "generate",
        "validate_citations",
    }.issubset(nodes)
    assert ("preprocess", "retrieve") in edges
    assert ("generate", "validate_citations") in edges


def test_pipeline_factory_rejects_unknown_configured_module():
    settings = Settings(
        web_search_enabled=False,
        rag_selector_module="unknown",
    )

    with pytest.raises(UnknownModuleError, match="selector.*unknown"):
        RAGPipelineFactory(settings).create(
            vector_store=FakeVectorStore(),
            model_gateway=FakeGateway(),
            reranker=EmptyReranker(),
            web_search_provider=DisabledWebSearch(),
        )


def test_pipeline_factory_validates_default_module_configuration():
    factory = RAGPipelineFactory(Settings(web_search_enabled=False))

    factory.validate_module_configuration()
