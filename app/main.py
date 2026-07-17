from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from sqlalchemy import text

from app.api import router
from app.config import get_settings
from app.db import SessionLocal
from app.rag.orchestration.factory import RAGPipelineFactory
from app.storage import ObjectStorage
from app.vector_store import MilvusChunkStore

settings = get_settings()
RAGPipelineFactory(settings).validate_module_configuration()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok", "dependencies": {}}


@app.get("/health/ready")
def ready(response: Response) -> dict:
    dependencies = {}
    checks = {
        "mysql": lambda: _mysql_ready(),
        "redis": lambda: Redis.from_url(settings.redis_url).ping(),
        "minio": lambda: ObjectStorage().ready(),
        "milvus": lambda: MilvusChunkStore().ready(),
        "models": lambda: bool(
            settings.llm_api_key
            and settings.llm_model
            and settings.embedding_api_key
            and settings.embedding_model
        ),
        "reranker": lambda: _reranker_ready(),
        "web_search": lambda: _web_search_ready(),
    }
    for name, check in checks.items():
        try:
            dependencies[name] = "ok" if check() else "unavailable"
        except Exception as exc:
            detail = (
                f": {str(exc)[:240]}"
                if name == "milvus" and str(exc)
                else ""
            )
            dependencies[name] = (
                f"unavailable: {type(exc).__name__}{detail}"
            )
    status = (
        "ok"
        if all(value == "ok" for value in dependencies.values())
        else "degraded"
    )
    if status != "ok":
        response.status_code = 503
    return {"status": status, "dependencies": dependencies}


def _mysql_ready() -> bool:
    with SessionLocal() as db:
        return db.scalar(text("SELECT 1")) == 1


def _reranker_ready() -> bool:
    if settings.reranker_type != "external":
        return False
    return bool(settings.reranker_endpoint and settings.reranker_model)


def _web_search_ready() -> bool:
    if not settings.web_search_enabled:
        return True
    if settings.web_search_provider != "tavily":
        return False
    return bool(settings.tavily_endpoint and settings.tavily_api_key)
