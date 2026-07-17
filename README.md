# Enterprise Corrective RAG

面向部门内部知识库的企业化 RAG 示例。项目使用 FastAPI、Streamlit、Celery、MySQL 8.0、Redis、MinIO、Milvus、阿里百炼兼容模型接口和 LangSmith。

## Architecture

- FastAPI：认证、文档、任务、会话、问答、ACL、审计和健康检查。
- Streamlit：只访问 FastAPI，不直接连接数据库、向量库或模型。
- Celery Worker：异步执行解析、切分、Embedding、Milvus 写入、版本激活和向量清理。
- MySQL 8.0：用户、部门、文档、ACL、版本、任务、会话和审计的唯一事实来源。
- MinIO：保存 PDF、TXT 和 Markdown 原始文件。
- Milvus：保存 chunk、Embedding 和检索元数据，查询前由后端生成授权版本过滤条件。
- LangGraph：以固定拓扑编排查询改写、检索、融合、重排、路由、证据选择、生成和引用校验。

查询链路采用 Modular RAG 结构。`RAGService` 只负责授权、会话、持久化和审计，SQLAlchemy 不进入 `RAGPipeline`；各查询阶段通过白名单 Registry 组装，可以在不改变 HTTP API 和 diagnostics 合约的前提下替换实现。

授权知识不足时可通过 Tavily Search API 获取真实外部候选，候选必须经过专业 Reranker 后才能用于答案生成。Tavily 故障时返回结构化拒答，不使用 Mock 或未经验证的替代内容。

## Quick Start

1. 从示例创建本地配置并填写百炼模型参数：

```powershell
Copy-Item .env.example .env
```

至少修改：

```dotenv
SECRET_KEY=replace-with-a-long-random-secret
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
EMBEDDING_DIMENSION=1024
TAVILY_API_KEY=...
```

`TAVILY_API_KEY` 用于默认开启的真实 Web Search。未配置时进程可以启动，但 `/health/ready` 会返回 503。外部搜索会产生费用并发送用户的搜索问题，生产环境应使用 Docker Secret 或企业密钥服务。

`EMBEDDING_DIMENSION` 必须与百炼 Embedding 模型实际输出维度一致。生产环境应使用 Docker Secret 或企业密钥服务，并保持：

```dotenv
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

文档默认采用结构感知 Parent-Child 切分，Child 用于 Dense、BM25 和
Reranker，Parent 用于最终答案生成：

```dotenv
CHUNKING_STRATEGY=structure_parent_child
CHUNK_SIZE=800
CHUNK_OVERLAP=120
CHUNK_PARENT_SIZE=2400
CHUNK_PARENT_OVERLAP=200
CHUNK_MIN_SIZE=120
CHUNK_CONTEXT_HEADER_ENABLED=true
CHUNKING_VERSION=v2
MILVUS_COLLECTION=enterprise_rag_chunks_parent_child_v2
RETRIEVAL_MODE=hybrid
RETRIEVAL_DENSE_LIMIT=10
RETRIEVAL_SPARSE_LIMIT=10
RETRIEVAL_RRF_K=60
```

Milvus 默认使用 Dense 向量检索和原生 BM25 Sparse 检索，并通过 RRF
融合。Markdown 会保留标题层级、代码块、列表和简单表格；TXT 与文本型
PDF 会识别常见章节、条款和编号结构。扫描件和复杂表格暂不支持。

Modular RAG 默认模块：

```dotenv
RAG_QUERY_MODULE=default
RAG_RETRIEVER_MODULE=milvus
RAG_FUSION_MODULE=rrf
RAG_ROUTER_MODULE=llm
RAG_SELECTOR_MODULE=route_aware
RAG_GENERATOR_MODULE=langchain
RAG_VALIDATOR_MODULE=bracket_citations
```

模块名只能从内置 Registry 白名单选择，配置未知名称时应用组装会立即失败，不执行任意动态导入。

从旧 collection 切换到 Parent-Child collection 后，重建所有 active ready
版本。脚本会先按 `version_uuid` 删除新 collection 中的旧结果，支持幂等重建：

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_hybrid_index.py --dry-run
.\.venv\Scripts\python.exe scripts\rebuild_hybrid_index.py
```

切换期间应暂停上传 Worker，同时切换 API 和 Worker 的
`MILVUS_COLLECTION`，并在一个观察周期内保留旧 collection 便于回滚。

2. 确保本机 Milvus 可通过 `localhost:19530` 访问，然后启动应用：

```powershell
docker compose up --build
```

3. 打开：

- Streamlit：`http://localhost:8501`
- FastAPI 文档：`http://localhost:8000/docs`
- MinIO Console：`http://localhost:9001`

默认管理员由 `.env` 中的 `BOOTSTRAP_ADMIN_*` 创建，首次启动后应立即更换密码配置。

## Local Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d mysql redis minio
$env:DATABASE_URL="mysql+pymysql://rag:rag_password@localhost:3307/rag?charset=utf8mb4"
$env:REDIS_URL="redis://localhost:6380/0"
$env:MINIO_ENDPOINT="localhost:9000"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe scripts\bootstrap.py
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

另开终端运行 Worker 和前端：

```powershell
.\.venv\Scripts\celery.exe -A app.celery_app:celery_app worker --loglevel=INFO --pool=solo
.\.venv\Scripts\streamlit.exe run corrective_rag.py
```

Windows 本地 Celery 建议使用 `--pool=solo`；容器中使用默认 prefork。

## Public APIs

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/documents`
- `GET /api/v1/documents`
- `POST /api/v1/documents/{uuid}/versions`
- `GET /api/v1/documents/{uuid}`
- `PUT /api/v1/documents/{uuid}/acl`
- `DELETE /api/v1/documents/{uuid}`
- `GET /api/v1/jobs/{uuid}`
- `POST /api/v1/queries`
- `GET /api/v1/conversations`
- `GET /health/live`
- `GET /health/ready`

客户端不能为查询传入部门、文档或 Milvus 过滤表达式。

## Validation

```powershell
.\.venv\Scripts\ruff.exe check app scripts tests corrective_rag.py
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe -m pip check
docker compose config
```

详细设计见 [架构](docs/architecture.md)、[RAG](docs/rag.md)、[超参数](docs/hyperparameters.md)、[API](docs/api.md)、[数据模型](docs/data-model.md) 和 [测试计划](docs/test-plan.md)。
