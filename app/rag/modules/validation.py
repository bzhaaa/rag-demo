import time
from typing import List

from app.config import Settings
from app.rag.contracts.protocols import AnswerGenerator
from app.rag.contracts.state import RAGState
from app.rag.utils import valid_citation_indices


class BracketCitationValidationModule:
    def __init__(self, generator: AnswerGenerator, settings: Settings) -> None:
        self.generator = generator
        self.settings = settings

    def validate(self, answer: str, evidence_count: int) -> List[int]:
        return valid_citation_indices(answer, evidence_count)

    def run(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        if state.get("refused"):
            return {**state, "cited_indices": []}
        evidence = state.get("evidence", state.get("relevant", []))
        answer = state.get("answer", "")
        cited_indices = self.validate(answer, len(evidence))
        retry_count = 0
        while (
            not cited_indices
            and retry_count < self.settings.rag_citation_retry_count
        ):
            retry_count += 1
            answer = self.generator.generate_answer(
                state["question"],
                evidence,
                strict_citations=True,
            )
            cited_indices = self.validate(answer, len(evidence))
        elapsed = time.perf_counter() - started
        timings = dict(state.get("timings", {}))
        timings["citation_validation"] = elapsed
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
                "timings": timings,
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
            "timings": timings,
        }
