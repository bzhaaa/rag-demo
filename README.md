# Enterprise Corrective RAG

面向部门内部知识库的企业化 RAG 示例。项目使用 FastAPI、Streamlit、Celery、MySQL 8.0、Redis、MinIO、Milvus、阿里百炼兼容模型接口和 LangSmith。

## Architecture

- FastAPI：认证、文档、任务、会话、问答、ACL、审计和健康检查。
- Streamlit：只访问 FastAPI，不直接连接数据库、向量库或模型。
- Celery Worker：异步执行解析、切分、Embedding、Milvus 写入、版本激活和向量清理。
- MySQL 8.0：用户、部门、文档、ACL、版本、任务、会话和审计的唯一事实来源。
- MinIO：保存 PDF、TXT 和 Markdown 原始文件。
- Milvus：保存 chunk、Embedding 和检索元数据，查询前由后端生成授权版本过滤条件。
- LangGraph：并行相关性评分、证据判断和答案生成。

正式问答不会使用 Mock Search 内容。授权知识不足时返回结构化拒答。

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
```

`EMBEDDING_DIMENSION` 必须与百炼 Embedding 模型实际输出维度一致。生产环境应使用 Docker Secret 或企业密钥服务，并保持：

```dotenv
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

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

详细设计见 [架构](docs/architecture.md)、[API](docs/api.md)、[数据模型](docs/data-model.md) 和 [测试计划](docs/test-plan.md)。
