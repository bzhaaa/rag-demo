# RAG

本文只整理检索增强生成链路。认证、通用文档管理、UI 行为和部署细节只在它们影响 RAG 边界时提及。

## 范围

RAG 子系统包括：

- 文档校验、解析、切分、Embedding 和 Milvus 索引。
- 基于当前用户授权范围的活跃文档版本检索。
- 通过 LangGraph 编排的 LLM 相关性评分、三档证据路由和答案生成。
- 知识库证据不足时可选的 Mock 网络搜索降级。
- 引用、会话持久化、查询耗时指标和审计日志。

不包括 Streamlit UI 细节、通用账号管理，也不展开非查询业务 API；只有上传和版本接口作为索引入口时会被提到。

## 主要文件

- `app/api.py`：提供上传、版本上传接口并投递索引任务；`POST /api/v1/queries` 调用 `RAGService`。
- `app/services.py`：校验上传文件，创建文档版本和任务，写审计日志，并在索引成功后激活版本。
- `app/tasks.py`：Celery 文档入库和向量删除任务。
- `app/ingestion.py`：抽取文本并构建 chunk 元数据。
- `app/vector_store.py`：负责 Milvus collection schema、Embedding 调用、带过滤条件的检索、向量写入/删除和元数据同步。
- `app/repositories.py`：解析当前用户可访问的活跃文档版本。
- `app/rag/service.py`：RAGService 和 LangGraph 编排，负责授权检索、三档证据路由、网络降级、生成、引用和持久化。
- `app/rag/model_gateway.py`：LLM 调用入口，负责相关性评分、证据路由、答案生成、Direct、HyDE、Step-back 和 Multi-query。
- `app/rag/preprocessors.py`：Query 预处理协议实现和策略组合。
- `app/rag/rerankers.py`：本地、透传和外部 HTTP reranker 实现。
- `app/rag/web_search.py`：网络搜索协议实现、Mock provider 和网络结果标准化。
- `app/rag/types.py`：RAG 状态和组件 Protocol。
- `app/rag/utils.py`：查询规范化、候选合并、引用解析等无状态工具。
- `app/rag/__init__.py`：兼容导出层，保留 `from app.rag import RAGService` 等旧导入方式。
- `app/models.py`：存储文档、版本、入库任务、会话、消息、引用、指标和审计日志。
- `app/config.py`：RAG、模型和索引相关配置。
- `tests/test_domain.py`：当前覆盖 chunk ID、相关性解析、上传校验和结构化拒答的单元测试。

## 数据流

```mermaid
flowchart LR
    Upload["上传或新版本"] --> Validate["validate_upload"]
    Validate --> Persist["Document + Version + IngestionJob"]
    Persist --> MinIO["原始文件写入 MinIO"]
    Persist --> Queue["投递 Celery ingest_document"]
    Queue --> Parse["parse_document"]
    Parse --> Chunk["build_chunks"]
    Chunk --> Embed["兼容 OpenAI 的 Embedding"]
    Embed --> Milvus["Milvus insert_chunks"]
    Milvus --> Activate["activate_version"]
    Activate --> Ready["current_version_id 指向 ready 版本"]

    Query["POST /api/v1/queries"] --> Versions["active_versions_for_user"]
    Versions --> Preprocess["LangGraph: Query 预处理"]
    Preprocess --> Search["多查询初次授权检索"]
    Search --> Grade["LangGraph: 重排与相关性评分"]
    Grade --> WebEnabled{"启用网络降级？"}
    WebEnabled -->|否| SelectKB["选择知识库证据或拒答"]
    WebEnabled -->|是| Route{"LLM 三档证据路由"}
    Route -->|knowledge_base| SelectKB
    Route -->|web| WebSearch["Mock 网络搜索"]
    Route -->|hybrid| WebSearch
    WebSearch --> WebGrade["网络结果重排与相关性评分"]
    WebGrade --> Select["按路由选择最终证据"]
    SelectKB --> Generate["LangGraph: 生成与引用校验"]
    Select --> Generate
    Generate --> Save["消息、引用、指标、审计"]
    Save --> Response["QueryResponse"]
```

## 入库路径

1. `app/api.py` 中的上传接口读取文件，校验归属和部门边界，创建文档版本与入库任务，上传原始文件到 MinIO，然后投递 `ingest_document`。
2. `ingest_document` 依次更新任务阶段：`parsing`、`embedding`、`indexing`、`activating`、`ready`。
3. `parse_document` 支持通过 `pypdf` 抽取 PDF 文本，也支持 UTF-8 文本和 Markdown。
4. `build_chunks` 使用 `RecursiveCharacterTextSplitter`，切分参数来自配置中的 `chunk_size` 和 `chunk_overlap`。
5. Chunk ID 是确定性的：`document_uuid:version_number:chunk_index`。
6. `MilvusChunkStore.insert_chunks` 对 chunk 文本做 Embedding，并写入向量和检索元数据。
7. `activate_version` 只在 Milvus 写入成功后设置 `Document.current_version_id`，因此索引失败不会覆盖旧的活跃版本。

## 检索和生成路径

1. `POST /api/v1/queries` 只接收 `question` 和可选的 `conversation_uuid`；客户端不能传部门 ID、文档 ID 或 Milvus 过滤表达式。
2. `RAGService.answer` 通过 `active_versions_for_user` 解析当前用户可访问的活跃版本。MySQL 是授权事实来源。
3. `preprocess_query` 在初次检索前按配置执行 Query 预处理，生成去重后的检索查询列表；用户原始问题保持不变，仍用于最终答案生成。
4. `retrieve` 对预处理后的查询逐个执行 Milvus 检索。`MilvusChunkStore.search` 对查询做 Embedding，并应用表达式：`version_uuid in [...]`。多查询结果按 `chunk_id` 合并，同一 chunk 保留最高分。
5. `grade_documents` 先按 `RERANKER_TYPE` 选择候选重排策略，然后按配置决定是否执行 LLM 相关性评分。内置默认重排会过滤空内容和可选低分候选、按标准化内容去重、按分数排序、限制单文档 chunk 数。
6. 网络搜索关闭时，知识库证据达到最低数量后直接生成，否则返回 `insufficient_authorized_evidence`。
7. 网络搜索开启时，`route_evidence` 让 LLM 在 `knowledge_base`、`web` 和 `hybrid` 中选择。非法 JSON、超时或异常会保守拒答，不触发网络搜索。
8. `web` 和 `hybrid` 路由复用预处理查询，最多使用 `WEB_SEARCH_MAX_QUERIES` 个查询。首版 provider 是进程内 Mock，不发起真实 HTTP 请求；结果按 URL 去重并保留最高分。
9. 网络结果使用与知识库相同的 reranker、LLM 相关性评分和评分熔断。`web` 只使用网络证据；`hybrid` 必须同时有知识库和网络证据，并按知识库优先排列。
10. `generate` 根据最终证据生成中文答案，并校验答案中的 `[n]` 引用。无引用或越界引用会严格重试一次；仍失败则返回 `invalid_citations`。
11. `RAGService._citations` 只返回答案实际引用的证据。网络引用包含 `source_type = web` 和 URL，知识库引用包含原有文档和版本信息。
12. 用户消息和助手消息会持久化；审计日志额外记录最终路由、是否尝试网络搜索、两类证据数量和 provider。

## 授权不变量

- Milvus 检索前，先由 MySQL 解析出可访问的 version UUID，检索范围被这些 UUID 限定。
- 用户只能检索自己可访问的活跃文档版本。
- 管理员可访问所有未删除文档；其他用户可访问自己拥有的、部门可见的、用户 ACL 授权的或部门 ACL 授权的文档。
- 查询客户端无法扩大检索范围。
- 网络搜索关闭且授权证据不足时，返回 `insufficient_authorized_evidence`。
- 网络搜索开启后，路由失败、搜索无结果、相关性评分失败或最终证据不足时，返回 `insufficient_evidence`。
- 查询改写不会访问外部网络，也不会改变授权版本集合。
- 网络来源不具有企业授权证据语义，必须在引用中标记为 `web`。
- 答案缺少有效引用时返回结构化拒答，`refusal_reason = invalid_citations`。

## Query 预处理

`QUERY_REWRITE_TYPES` 支持按顺序组合以下类型：

| 策略 | 解决的核心问题 | 额外开销 | 适合场景 |
| --- | --- | --- | --- |
| `direct` | 口语化、指代不清、问题表达不完整 | 1 次 LLM 调用 | 用户问题较短、口语化或上下文表达不清 |
| `hyde` | 用户问题与知识库文档的表达风格差异较大 | 1 次 LLM 调用 | 专业知识库、文档语言与提问语言差异明显 |
| `step_back` | 具体问题需要背景知识、概念或原理支撑 | 1 次 LLM 调用 | 技术文档、规范、原理性问题 |
| `multi_query` | 单一查询角度覆盖不全 | 1 次 LLM 调用生成多个查询 | 答案涉及多个维度的复杂问题 |

此外还支持：

- `normalize`：本地规则清洗，去除首尾空白并合并连续空白，不调用 LLM。
- `standalone`：`direct` 的兼容别名，旧配置可继续使用。

预处理结果遵循以下规则：

- 第一项始终是规范化后的用户原问题。
- 查询按规范化文本去重。
- 初次检索最多使用 `QUERY_REWRITE_MAX_QUERIES` 个查询。默认启用 `normalize,direct,multi_query`，HyDE 和 Step-back 建议按知识库场景显式开启。
- LLM 改写失败时保留规范化原问题，不中断 RAG 请求。
- 所有查询只用于检索；最终生成仍使用用户原始问题。

## CRAG 三档路由与网络降级

`WEB_SEARCH_ENABLED=false` 是默认值。关闭时不会调用证据路由模型或网络 provider，行为保持为严格的授权知识库 RAG。

开启后，初次知识库检索和评分完成时，模型必须返回：

```json
{"route": "knowledge_base|web|hybrid"}
```

- `knowledge_base`：知识库证据足以独立回答，不调用网络搜索。
- `web`：放弃知识库弱证据，只使用通过重排和评分的网络结果。
- `hybrid`：同时使用知识库和网络证据。任一侧没有相关证据时拒答。

首版 `WEB_SEARCH_PROVIDER=mock` 内置 3 条明确标记为 Mock 的 RAG/CRAG 示例结果，URL 使用 `example.com`。它只用于测试和开发演示，不能作为真实事实来源。`create_web_search_provider` 和 `WebSearchProvider` 已预留后续 Tavily、Bing 或自建 HTTP 适配器的接入位置。

新增耗时项：

- `evidence_routing`
- `web_search`
- `web_grading`

## Reranker 策略

`RERANKER_TYPE` 控制候选重排实现：

- `default`：默认策略。使用本地规则过滤、去重、按向量分数排序，并限制单文档 chunk 数。
- `identity`：不做重排，按检索返回顺序继续后续评分和生成，主要用于排障或对比实验。
- `external`：调用外部 reranker HTTP 接口，根据模型返回分数重排候选；接口失败或配置为空时回退到 `default` 策略。

外部 reranker 请求格式：

```json
{
  "model": "optional-reranker-model",
  "query": "用户问题",
  "documents": ["候选 chunk 文本 1", "候选 chunk 文本 2"]
}
```

响应支持 `results` 或 `data` 数组，每项至少包含候选下标和分数：

```json
{
  "results": [
    {"index": 1, "relevance_score": 0.98},
    {"index": 0, "relevance_score": 0.31}
  ]
}
```

鉴权使用 `RERANKER_API_KEY`，以 `Authorization: Bearer ...` 发送。

## Milvus Chunk 结构

每个已索引 chunk 包含：

- `chunk_id`
- `document_uuid`
- `version_uuid`
- `version_number`
- `department_uuid`
- `visibility`
- `page_number`
- `chunk_index`
- `source_name`
- `content`
- `embedding`

`department_uuid` 和 `visibility` 用于元数据同步和检查；查询检索的授权判断在 Milvus 搜索前由 MySQL 完成。

## 关键配置

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `RELEVANCE_MAX_CONCURRENCY`
- `RELEVANCE_GRADING_ENABLED`
- `QUERY_REWRITE_ENABLED`
- `QUERY_REWRITE_TYPES`
- `QUERY_REWRITE_MAX_QUERIES`
- `WEB_SEARCH_ENABLED`
- `WEB_SEARCH_PROVIDER`
- `WEB_SEARCH_MAX_QUERIES`
- `WEB_SEARCH_RESULT_COUNT`
- `RERANKER_TYPE`
- `RERANKER_ENDPOINT`
- `RERANKER_API_KEY`
- `RERANKER_MODEL`
- `RETRIEVAL_CANDIDATE_COUNT`
- `RETRIEVAL_MIN_SCORE`
- `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT`
- `FINAL_CONTEXT_COUNT`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MODEL_TIMEOUT_SECONDS`
- `MODEL_MAX_RETRIES`
- `RAG_CITATION_RETRY_COUNT`
- `RAG_MIN_RELEVANT_DOCUMENTS`
- `LANGSMITH_TRACING`, `LANGSMITH_HIDE_INPUTS`, `LANGSMITH_HIDE_OUTPUTS`

## 当前测试和缺口

当前直接覆盖 RAG 相关行为的测试包括：

- 确定性的 chunk ID。
- 上传校验。
- JSON 相关性解析。
- 无证据时的结构化拒答。
- `RAGService.answer` 在 fake vector store 和 fake model gateway 下的主链路。
- 授权版本 UUID 传递、消息持久化、完整 timings 持久化。
- 候选过滤、去重、排序和单文档 chunk 限额。
- Reranker 配置选择和外部 reranker 请求/排序。
- Query normalize、direct、HyDE、Step-back、多查询扩展、配置组合和结果去重。
- 初检前多查询授权检索，且不会执行第二轮知识库检索。
- `knowledge_base`、`web`、`hybrid` 三档路由和路由失败时的保守拒答。
- Mock 网络搜索默认关闭、查询数量限制、URL 去重、稳定 chunk ID。
- 网络结果重排和相关性评分、web 评分失败拒答、混合证据来源完整性。
- 引用校验、严格重试、只返回实际引用 chunks 和 `invalid_citations` 拒答。
- 知识库与网络引用结构、消息指标和路由审计信息持久化。

建议补充：

- 覆盖 owner、department、显式 ACL、admin 路径下的授权版本过滤。
- 覆盖 chunks 指向已删除或不可访问文档时的引用组装。
- 接入真实网络搜索 provider 后增加 HTTP 超时、重试、速率限制和域名策略测试。
- 增加带 opt-in 标记的 Milvus insert/search/delete 集成测试。
- 增加离线评估集，跟踪 Recall@K、引用准确率和事实一致性。
