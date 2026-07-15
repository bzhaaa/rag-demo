import json
import re
from typing import Any, Dict, List, Sequence

TRUTHY_RELEVANCE = {"yes", "y", "true", "1", "relevant", "是", "相关"}
FALSY_RELEVANCE = {"no", "n", "false", "0", "irrelevant", "否", "不相关"}


def parse_first_json_object(response: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    text = str(response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No JSON object found", text, 0)


def parse_relevance(response: Any) -> bool:
    if isinstance(response, bool):
        return response
    if isinstance(response, dict):
        parsed = response
    else:
        text = str(response or "").strip()
        normalized = text.lower()
        if normalized in TRUTHY_RELEVANCE:
            return True
        if normalized in FALSY_RELEVANCE:
            return False
        parsed = parse_first_json_object(text)

    for field in ("score", "relevant", "is_relevant", "answer"):
        if field not in parsed:
            continue
        value = parsed[field]
        if isinstance(value, bool):
            return value
        normalized_value = str(value).strip().lower()
        if normalized_value in TRUTHY_RELEVANCE:
            return True
        if normalized_value in FALSY_RELEVANCE:
            return False
    return False


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
