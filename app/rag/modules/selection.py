from typing import Any, Dict, List, Sequence, Tuple

from app.config import Settings
from app.rag.contracts.models import Candidate
from app.rag.contracts.state import RAGState


class RouteAwareSelectionModule:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: RAGState) -> RAGState:
        selected, parent_diagnostics = self._select_with_diagnostics(state)
        evidence = [item.to_mapping() for item in selected]
        route = (
            state.get("evidence_route", "")
            if self.settings.web_search_enabled
            else "knowledge_base"
        )
        return {
            **state,
            "evidence_route": route,
            "evidence": evidence,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "selected_evidence_count": len(evidence),
                "parent_expansion": parent_diagnostics,
            },
        }

    def select(self, state: RAGState) -> List[Candidate]:
        selected, _ = self._select_with_diagnostics(state)
        return selected

    def _select_with_diagnostics(
        self, state: RAGState
    ) -> Tuple[List[Candidate], Dict[str, Any]]:
        raw_knowledge = [
            Candidate.from_mapping(item) for item in state.get("relevant", [])
        ]
        knowledge = self._dedupe_and_expand_parents(raw_knowledge)
        web = [
            Candidate.from_mapping(item) for item in state.get("web_relevant", [])
        ]
        diagnostics = self._parent_diagnostics(raw_knowledge, knowledge)
        minimum = self.settings.rag_min_relevant_documents
        if not self.settings.web_search_enabled:
            return (
                knowledge[: self.settings.final_context_count]
                if len(knowledge) >= minimum
                else [],
                diagnostics,
            )
        route = state.get("evidence_route", "")
        if state.get("evidence_routing_failed"):
            return [], diagnostics
        if route == "knowledge_base":
            return (
                knowledge[: self.settings.final_context_count]
                if len(knowledge) >= minimum
                else [],
                diagnostics,
            )
        if route == "web":
            return (
                web[: self.settings.final_context_count]
                if len(web) >= minimum
                else [],
                diagnostics,
            )
        if route == "hybrid" and knowledge and web:
            selected = [*knowledge, *web][: self.settings.final_context_count]
            return (
                selected if len(selected) >= minimum else [],
                diagnostics,
            )
        return [], diagnostics

    @staticmethod
    def _dedupe_and_expand_parents(
        candidates: Sequence[Candidate],
    ) -> List[Candidate]:
        deduped: Dict[str, Candidate] = {}
        for candidate in candidates:
            parent_key = candidate.parent_chunk_id or candidate.chunk_id
            existing = deduped.get(parent_key)
            if existing is None or RouteAwareSelectionModule._rank(
                candidate
            ) > RouteAwareSelectionModule._rank(existing):
                deduped[parent_key] = candidate

        expanded: List[Candidate] = []
        for candidate in sorted(
            deduped.values(),
            key=RouteAwareSelectionModule._rank,
            reverse=True,
        ):
            if candidate.source_type == "web" or not candidate.parent_content:
                expanded.append(candidate)
                continue
            item = Candidate.from_mapping(candidate.to_mapping())
            item.extra.update(
                {
                    "matched_child_id": candidate.chunk_id,
                    "matched_child_content": candidate.content,
                    "citation_content": candidate.content,
                }
            )
            item.content = candidate.parent_content
            expanded.append(item)
        return expanded

    def _parent_diagnostics(
        self,
        before: Sequence[Candidate],
        after: Sequence[Candidate],
    ) -> Dict[str, Any]:
        strategy = next(
            (
                item.chunking_strategy
                for item in before
                if item.chunking_strategy
            ),
            self.settings.chunking_strategy,
        )
        version = next(
            (item.chunking_version for item in before if item.chunking_version),
            self.settings.chunking_version,
        )
        return {
            "knowledge_before_dedup": len(before),
            "knowledge_after_dedup": len(after),
            "strategy": strategy,
            "version": version,
            "evidence": [
                {
                    "matched_child_id": (
                        item.extra.get("matched_child_id") or item.chunk_id
                    ),
                    "parent_chunk_id": item.parent_chunk_id,
                    "section_path": item.section_path,
                    "page_start": item.page_start or item.page_number,
                    "page_end": item.page_end or item.page_number,
                }
                for item in after
            ],
        }

    @staticmethod
    def _rank(candidate: Candidate) -> Tuple[float, float]:
        return (
            float(candidate.rerank_score or 0),
            float(candidate.score or 0),
        )
