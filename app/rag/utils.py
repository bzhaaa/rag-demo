import json
import re
from typing import Any, Dict, List, Sequence


def parse_relevance(response: str) -> bool:
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    parsed = json.loads(match.group() if match else response)
    return str(parsed.get("score", "")).lower() == "yes"


def normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().split())


def valid_citation_indices(answer: str, evidence_count: int) -> List[int]:
    indices = [int(match) for match in re.findall(r"\[(\d+)\]", answer)]
    if not indices:
        return []
    if any(index < 1 or index > evidence_count for index in indices):
        return []
    result: List[int] = []
    for index in indices:
        if index not in result:
            result.append(index)
    return result


def merge_candidates(
    candidates: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        chunk_id = str(candidate.get("chunk_id") or "")
        if not chunk_id:
            continue
        existing = merged.get(chunk_id)
        if existing is None or float(candidate.get("score") or 0) > float(
            existing.get("score") or 0
        ):
            merged[chunk_id] = candidate
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )


def timing_with(
    timings: Dict[str, float], key: str, elapsed: float
) -> Dict[str, float]:
    return {**timings, key: timings.get(key, 0) + elapsed}
