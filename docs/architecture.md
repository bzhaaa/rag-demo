# Architecture

The application is split into six runtime services:

- FastAPI owns public APIs, authentication, authorization, audit writes, and RAG orchestration.
- Streamlit is an API-only pilot client and never connects to MySQL, MinIO, Milvus, or model endpoints.
- Celery workers parse, chunk, embed, index, activate, and clean up document versions.
- MySQL 8.0 is the source of truth for users, departments, ACLs, versions, jobs, conversations, and audits.
- MinIO stores original uploads.
- Milvus stores chunk text, embeddings, and retrieval metadata. It does not decide authorization.

Retrieval first resolves the current user's accessible active versions from MySQL. Only those version UUIDs are included in the Milvus filter. Relevance grading remains parallel, with configurable concurrency, and generation refuses when authorized evidence is insufficient.

RAG 的检索、索引、评分、生成、引用和测试覆盖梳理见 [RAG](rag.md)。

Document activation happens only after all chunks have been written to Milvus. The worker locks both the version and document rows before switching `current_version_id`, so an indexing failure leaves the previous active version unchanged.

LangSmith tracing is configured through environment variables. Production defaults hide inputs and outputs.
