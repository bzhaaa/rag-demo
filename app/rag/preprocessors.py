from typing import List, Optional, Sequence

from app.config import Settings
from app.rag.types import QueryPreprocessor, RAGModelGateway
from app.rag.utils import normalize_query


class DefaultQueryPreprocessor:
    def __init__(
        self,
        enabled: bool = True,
        rewrite_types: Optional[Sequence[str]] = None,
        max_queries: int = 3,
    ) -> None:
        self.enabled = enabled
        aliases = {"standalone": "direct"}
        self.rewrite_types = [
            aliases.get(item.strip().lower(), item.strip().lower())
            for item in (rewrite_types or ["normalize"])
        ]
        self.max_queries = max(1, max_queries)

    def preprocess(
        self,
        question: str,
        model_gateway: RAGModelGateway,
        max_queries: Optional[int] = None,
    ) -> List[str]:
        limit = max(1, max_queries or self.max_queries)
        normalized_question = normalize_query(question)
        if not normalized_question:
            return []
        queries = [normalized_question]
        if not self.enabled:
            return queries

        for rewrite_type in self.rewrite_types:
            if len(queries) >= limit:
                break
            if rewrite_type == "normalize":
                continue
            if rewrite_type == "direct":
                try:
                    self._append_query(
                        queries,
                        model_gateway.rewrite_query(normalized_question),
                        limit,
                    )
                except Exception:
                    continue
            elif rewrite_type == "hyde":
                try:
                    self._append_query(
                        queries,
                        model_gateway.generate_hypothetical_document(
                            normalized_question
                        ),
                        limit,
                    )
                except Exception:
                    continue
            elif rewrite_type == "step_back":
                try:
                    self._append_query(
                        queries,
                        model_gateway.rewrite_step_back_query(
                            normalized_question
                        ),
                        limit,
                    )
                except Exception:
                    continue
            elif rewrite_type == "multi_query":
                if len(queries) >= limit:
                    continue
                try:
                    for query in model_gateway.rewrite_queries(
                        normalized_question, count=limit
                    ):
                        self._append_query(queries, query, limit)
                except Exception:
                    continue
        return queries

    @staticmethod
    def _append_query(queries: List[str], query: str, limit: int) -> None:
        normalized = normalize_query(query)
        if normalized and normalized not in queries and len(queries) < limit:
            queries.append(normalized)


def create_query_preprocessor(settings: Settings) -> QueryPreprocessor:
    return DefaultQueryPreprocessor(
        enabled=settings.query_rewrite_enabled,
        rewrite_types=settings.query_rewrite_types,
        max_queries=settings.query_rewrite_max_queries,
    )
