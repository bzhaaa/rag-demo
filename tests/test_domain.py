import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.rag import RAGService, parse_relevance
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


def test_structured_refusal_without_evidence():
    service = RAGService.__new__(RAGService)
    service.settings = get_settings()
    result = service._generate(
        {"question": "unknown", "relevant": [], "timings": {}}
    )
    assert result["refused"] is True
    assert result["refusal_reason"] == "insufficient_authorized_evidence"
