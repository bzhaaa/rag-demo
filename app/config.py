import os
from functools import lru_cache
from typing import Annotated, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Enterprise CRAG"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    database_url: str = (
        "mysql+pymysql://rag:rag_password@localhost:3306/rag?"
        "charset=utf8mb4"
    )
    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "rag_minio"
    minio_secret_key: str = "rag_minio_password"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "enterprise_rag_chunks_hybrid"

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_enable_thinking: bool = False
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 1024

    relevance_max_concurrency: int = 4
    relevance_grading_enabled: bool = True
    query_rewrite_enabled: bool = True
    query_rewrite_types: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "multi_query",
        ]
    )
    query_rewrite_max_queries: int = 3
    web_search_enabled: bool = True
    web_search_provider: str = "tavily"
    web_search_max_queries: int = 1
    web_search_result_count: int = 5
    web_search_timeout_seconds: int = 10
    web_search_max_retries: int = 2
    tavily_api_key: str = ""
    tavily_endpoint: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "basic"
    tavily_topic: str = "general"
    reranker_type: str = "external"
    reranker_endpoint: str = ""
    reranker_api_key: str = ""
    reranker_model: str = ""
    reranker_min_score: float = 0.5
    reranker_top_k: int = 6
    reranker_failure_strategy: str = "reject"
    retrieval_mode: str = "hybrid"
    retrieval_candidate_count: int = 10
    retrieval_dense_limit: int = 10
    retrieval_sparse_limit: int = 10
    retrieval_rrf_k: int = 60
    retrieval_min_score: Optional[float] = None
    retrieval_max_chunks_per_document: int = 3
    final_context_count: int = 6
    max_upload_bytes: int = 25 * 1024 * 1024
    max_pdf_pages: int = 500
    chunk_size: int = 800
    chunk_overlap: int = 120
    model_timeout_seconds: int = 45
    model_max_retries: int = 2
    rag_citation_retry_count: int = 1
    rag_min_relevant_documents: int = 1
    streamlit_api_url: str = "http://localhost:8000"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-crag"
    langsmith_endpoint: str = ""
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True

    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = "ChangeMe123!"
    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_department_name: str = "Default Department"

    @field_validator("llm_base_url", "embedding_base_url")
    @classmethod
    def normalize_openai_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        return normalized

    @field_validator("relevance_max_concurrency")
    @classmethod
    def positive_concurrency(cls, value: int) -> int:
        return max(1, value)

    @field_validator("retrieval_min_score", mode="before")
    @classmethod
    def empty_score_is_disabled(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("reranker_min_score")
    @classmethod
    def valid_reranker_min_score(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("reranker_min_score must be between 0 and 1")
        return value

    @field_validator("reranker_top_k")
    @classmethod
    def positive_reranker_top_k(cls, value: int) -> int:
        return max(1, value)

    @field_validator("reranker_type")
    @classmethod
    def supported_reranker_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"default", "identity", "external"}:
            raise ValueError("reranker_type must be default, identity, or external")
        return normalized

    @field_validator("reranker_failure_strategy")
    @classmethod
    def supported_reranker_failure_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"reject", "vector", "llm"}:
            raise ValueError(
                "reranker_failure_strategy must be reject, vector, or llm"
            )
        return normalized

    @model_validator(mode="after")
    def validate_reranker_fallback(self):
        if (
            self.reranker_failure_strategy == "vector"
            and self.retrieval_min_score is None
        ):
            raise ValueError(
                "retrieval_min_score is required when reranker_failure_strategy=vector"
            )
        return self

    @field_validator("web_search_provider")
    @classmethod
    def supported_web_search_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "tavily":
            raise ValueError("web_search_provider must be tavily")
        return normalized

    @field_validator(
        "web_search_max_queries",
        "web_search_result_count",
        "web_search_timeout_seconds",
        "query_rewrite_max_queries",
    )
    @classmethod
    def positive_counts(cls, value: int) -> int:
        return max(1, value)

    @field_validator("web_search_max_retries")
    @classmethod
    def non_negative_web_search_retries(cls, value: int) -> int:
        return max(0, value)

    @field_validator("retrieval_mode")
    @classmethod
    def supported_retrieval_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"dense", "sparse", "hybrid"}:
            raise ValueError("retrieval_mode must be dense, sparse, or hybrid")
        return normalized

    @field_validator(
        "retrieval_candidate_count",
        "retrieval_dense_limit",
        "retrieval_sparse_limit",
        "retrieval_rrf_k",
    )
    @classmethod
    def positive_retrieval_counts(cls, value: int) -> int:
        return max(1, value)

    @field_validator("tavily_search_depth")
    @classmethod
    def supported_tavily_search_depth(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"basic", "advanced"}:
            raise ValueError("tavily_search_depth must be basic or advanced")
        return normalized

    @field_validator("tavily_topic")
    @classmethod
    def supported_tavily_topic(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"general", "news", "finance"}:
            raise ValueError("tavily_topic must be general, news, or finance")
        return normalized

    @field_validator("query_rewrite_types", mode="before")
    @classmethod
    def split_query_rewrite_types(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("query_rewrite_types")
    @classmethod
    def supported_query_rewrite_types(cls, value: List[str]) -> List[str]:
        supported = {
            "normalize",
            "direct",
            "hyde",
            "step_back",
            "multi_query",
        }
        aliases = {"standalone": "direct"}
        normalized = []
        for item in value:
            rewrite_type = item.strip().lower()
            rewrite_type = aliases.get(rewrite_type, rewrite_type)
            if rewrite_type not in supported:
                raise ValueError(
                    "query_rewrite_types must contain normalize, direct, hyde, "
                    "step_back, or multi_query"
                )
            if rewrite_type not in normalized:
                normalized.append(rewrite_type)
        return normalized or ["normalize"]

    def configure_langsmith(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = (
            "true" if self.langsmith_tracing else "false"
        )
        if not self.langsmith_tracing:
            return
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        if self.langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGCHAIN_HIDE_INPUTS"] = str(self.langsmith_hide_inputs).lower()
        os.environ["LANGCHAIN_HIDE_OUTPUTS"] = str(self.langsmith_hide_outputs).lower()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.configure_langsmith()
    return settings
