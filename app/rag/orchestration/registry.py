from typing import Any, Callable, Dict


class UnknownModuleError(ValueError):
    pass


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def register(
        self,
        category: str,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        self._modules.setdefault(category, {})[name] = factory

    def resolve(self, category: str, name: str) -> Callable[..., Any]:
        normalized = name.strip().lower()
        factory = self._modules.get(category, {}).get(normalized)
        if factory is None:
            available = ", ".join(sorted(self._modules.get(category, {})))
            raise UnknownModuleError(
                f"unknown RAG {category} module '{name}'; available: {available}"
            )
        return factory


def create_default_registry() -> ModuleRegistry:
    from app.rag.modules import (
        BracketCitationValidationModule,
        DefaultQueryModule,
        KnowledgeRetrievalModule,
        LangChainGenerationModule,
        LLMEvidenceRoutingModule,
        RouteAwareSelectionModule,
        RRFFusionModule,
    )

    registry = ModuleRegistry()
    registry.register("query", "default", DefaultQueryModule)
    registry.register("retriever", "milvus", KnowledgeRetrievalModule)
    registry.register("fusion", "rrf", RRFFusionModule)
    registry.register("router", "llm", LLMEvidenceRoutingModule)
    registry.register("selector", "route_aware", RouteAwareSelectionModule)
    registry.register("generator", "langchain", LangChainGenerationModule)
    registry.register(
        "validator",
        "bracket_citations",
        BracketCitationValidationModule,
    )
    return registry
