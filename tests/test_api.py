from fastapi.testclient import TestClient

import app.main as main_module
from app.config import get_settings
from app.main import app


def test_live_health_endpoint():
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "dependencies": {}}


def test_ready_health_reports_missing_external_reranker(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "reranker_type", "external")
    monkeypatch.setattr(settings, "reranker_endpoint", "")
    monkeypatch.setattr(settings, "reranker_model", "")
    monkeypatch.setattr(main_module, "_mysql_ready", lambda: True)
    monkeypatch.setattr(
        main_module.Redis,
        "from_url",
        lambda *_: type("R", (), {"ping": lambda self: True})(),
    )
    monkeypatch.setattr(main_module.ObjectStorage, "ready", lambda self: True)
    monkeypatch.setattr(main_module.MilvusChunkStore, "ready", lambda self: True)
    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(settings, "llm_model", "chat")
    monkeypatch.setattr(settings, "embedding_api_key", "x")
    monkeypatch.setattr(settings, "embedding_model", "embed")

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["reranker"] == "unavailable"


def test_ready_health_reports_missing_tavily_api_key(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "web_search_enabled", True)
    monkeypatch.setattr(settings, "web_search_provider", "tavily")
    monkeypatch.setattr(settings, "tavily_endpoint", "https://api.tavily.com/search")
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(main_module, "_mysql_ready", lambda: True)
    monkeypatch.setattr(
        main_module.Redis,
        "from_url",
        lambda *_: type("R", (), {"ping": lambda self: True})(),
    )
    monkeypatch.setattr(main_module.ObjectStorage, "ready", lambda self: True)
    monkeypatch.setattr(main_module.MilvusChunkStore, "ready", lambda self: True)
    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(settings, "llm_model", "chat")
    monkeypatch.setattr(settings, "embedding_api_key", "x")
    monkeypatch.setattr(settings, "embedding_model", "embed")
    monkeypatch.setattr(settings, "reranker_type", "external")
    monkeypatch.setattr(settings, "reranker_endpoint", "https://rerank.example")
    monkeypatch.setattr(settings, "reranker_model", "rerank")

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["web_search"] == "unavailable"
