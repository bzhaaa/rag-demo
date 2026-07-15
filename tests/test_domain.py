import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.rag import RAGService, parse_relevance
from app.rag.model_gateway import (
    EvidenceRouteDecision,
    RelevanceGrade,
    parse_structured_model_output,
)
from app.services import validate_upload
from app.vector_store import chunk_id


def test_chunk_id_is_deterministic():
    assert chunk_id("doc", 3, 7) == "doc:3:7"


def test_validate_text_upload():
    result = validate_upload("policy.md", "text/markdown", b"# Policy\nAllowed")
    assert result["size_bytes"] > 0
    assert len(result["checksum"]) == 64


def test_reject_binary_text_upload():
    with pytest.raises(HTTPException) as error:
        validate_upload("policy.txt", "text/plain", b"\x00\x01")
    assert error.value.status_code == 415


def test_parse_relevance_json():
    assert parse_relevance('{"score":"yes"}')
    assert not parse_relevance('prefix {"score":"no"} suffix')


def test_parse_relevance_llm_output_variants():
    assert parse_relevance('```json\n{"score":"yes"}\n```')
    assert parse_relevance('{"relevant": true}')
    assert parse_relevance("是")
    assert parse_relevance("YES")
    assert not parse_relevance('{"score":"no"}')
    assert not parse_relevance("否")


def test_structured_model_output_uses_pydantic_schema():
    result = parse_structured_model_output(
        'prefix ```json\n{"score":"yes"}\n``` suffix',
        RelevanceGrade,
    )
    assert result.score == "yes"

    with pytest.raises(ValidationError):
        parse_structured_model_output('{"score":"maybe"}', RelevanceGrade)

    with pytest.raises(ValidationError):
        parse_structured_model_output(
            '{"route":"internet"}',
            EvidenceRouteDecision,
        )


def test_structured_refusal_without_evidence():
    service = RAGService.__new__(RAGService)
    service.settings = Settings(web_search_enabled=False)
    result = service._generate(
        {"question": "unknown", "relevant": [], "timings": {}}
    )
    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
    assert result["refusal_detail"] == "no_relevant_evidence"
