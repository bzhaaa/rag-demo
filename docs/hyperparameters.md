# 项目超参数说明

本文整理当前项目中会明显影响 RAG 效果、延迟、成本、稳定性和安全边界的配置项。配置统一由 `app/config.py` 的 `Settings` 读取，默认从项目根目录 `.env` 加载；`.env.example` 是可复制的配置模板。

严格来说，数据库连接、API Key、服务地址不是模型超参数，但它们决定运行环境和依赖可用性，因此本文单独列为“运行配置”。生产环境不要把真实密钥提交到仓库。

## 调参优先级

建议按下面顺序调参：

1. 文档切分：`CHUNKING_STRATEGY`、Parent/Child size、overlap 和 `CHUNK_MIN_SIZE`
2. 召回规模：`RETRIEVAL_MODE`、`RETRIEVAL_DENSE_LIMIT`、`RETRIEVAL_SPARSE_LIMIT`、`RETRIEVAL_CANDIDATE_COUNT`
3. Reranker 准入：`RERANKER_MIN_SCORE`、`RERANKER_TOP_K`
4. 最终上下文：`FINAL_CONTEXT_COUNT`、`RAG_MIN_RELEVANT_DOCUMENTS`
5. 查询改写：`QUERY_REWRITE_TYPES`、`QUERY_REWRITE_MAX_QUERIES`
6. Web Search：`WEB_SEARCH_MAX_QUERIES`、`WEB_SEARCH_RESULT_COUNT`
7. 稳定性：`MODEL_TIMEOUT_SECONDS`、`MODEL_MAX_RETRIES`、`RERANKER_FAILURE_STRATEGY`

不要一开始同时调整太多项。每次只改一组参数，并用固定问题集观察 Recall、引用准确率、忠实度、p95 延迟和外部调用成本。

## 模型与 Embedding

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `LLM_MODEL` | 空 | 问答、查询改写、证据路由使用的聊天模型。 | 选择稳定、支持结构化输出较好的模型。模型越强，路由和答案质量通常越好，但成本和延迟更高。 |
| `LLM_BASE_URL` | 空 | OpenAI 兼容 Chat API 地址。 | 百炼 DashScope 兼容接口填写到 `/v1` 层级即可，代码会清理 `/chat/completions` 后缀。 |
| `LLM_ENABLE_THINKING` | `false` | Qwen/DashScope 场景下控制 `enable_thinking`。 | 复杂推理可尝试开启；企业知识库问答通常先保持关闭以降低延迟和输出不确定性。 |
| `EMBEDDING_MODEL` | 空 | 文档和查询向量模型。 | 与业务语言匹配。中文制度、客服知识库优先选中文或多语表现好的 embedding。 |
| `EMBEDDING_DIMENSION` | `1024` | Milvus dense vector 维度。 | 必须与实际 embedding 模型输出维度一致；修改后通常需要新 collection 或重建索引。 |
| `MODEL_TIMEOUT_SECONDS` | `45` | LLM、Embedding、Reranker 等模型调用超时。 | p95 慢但可接受时可增大；交互式问答建议控制在 30-60 秒。 |
| `MODEL_MAX_RETRIES` | `2` | 模型调用重试次数。 | 外部模型偶发失败时保留 1-2 次；高并发场景过高会放大拥塞。 |

## 文档入库与切分

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `MAX_UPLOAD_BYTES` | `25 * 1024 * 1024` | 单文件最大上传大小。 | 由后端配置，当前 `.env.example` 未显式列出。企业内网可按文件类型和扫描能力放宽。 |
| `MAX_PDF_PAGES` | `500` | PDF 最大页数。 | 防止超大 PDF 拖垮解析和入库任务。制度类文档通常 200-500 页足够。 |
| `CHUNKING_STRATEGY` | `structure_parent_child` | 切分策略。可选 `structure_parent_child`、`legacy_recursive`。 | 新文档使用 Parent-Child；仅在回滚或对比基线时使用旧版递归切分。 |
| `CHUNK_SIZE` | `800` | Child 正文目标长度。Child 参与 Dense、BM25 和 Reranker。 | 过小会割裂语义，过大会降低召回精度。中文制度/FAQ 可从 600-1000 试起。 |
| `CHUNK_OVERLAP` | `120` | Child 相邻重叠长度。 | 一般取 Child size 的 10%-20%；增加会提高跨段召回，也会增加索引和 Reranker 成本。 |
| `CHUNK_PARENT_SIZE` | `2400` | Parent 正文目标长度。Parent 只用于答案生成。 | 应能容纳完整章节或条款上下文，但需控制最终 prompt 成本。可从 1800-3200 校准。 |
| `CHUNK_PARENT_OVERLAP` | `200` | 超长 Parent 递归拆分时的重叠长度。 | 用于保留 Parent 边界语义，通常小于 Child size。 |
| `CHUNK_MIN_SIZE` | `120` | 碎片合并阈值。 | 太小会产生弱证据，太大可能错误合并独立条款；只在同一标题层级内合并。 |
| `CHUNK_CONTEXT_HEADER_ENABLED` | `true` | 是否把标题路径加入 Child 检索文本。 | 制度、手册和技术文档建议开启，可提升章节名和条款名召回。 |
| `CHUNKING_VERSION` | `v2` | 切分实现版本标识。 | 改变解析或切分语义时升级，用于诊断、重建和评测追踪。 |

当前支持文本型 PDF、TXT 和 Markdown，不支持 OCR 和复杂 PDF 表格。Child ID 使用 `document_uuid:version_number:chunk_index`，Parent ID 使用 `document_uuid:version_number:parent:parent_index`。Milvus 中每个 Child 冗余保存 Parent 正文，MySQL 版本 metadata 保存切分策略和参数。

## 查询改写

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `QUERY_REWRITE_ENABLED` | `true` | 是否启用查询预处理和改写。 | 生产建议开启；如果模型调用成本敏感或问题很短，可仅保留 `normalize`。 |
| `QUERY_REWRITE_TYPES` | `multi_query` | 改写策略列表。支持 `normalize`、`direct`、`hyde`、`step_back`、`multi_query`。 | 默认模板建议 `normalize,direct,multi_query`。知识库表达和用户表达差距大时加入 `hyde`。 |
| `QUERY_REWRITE_MAX_QUERIES` | `3` | 单次问答最多用于检索的查询数。 | 越大召回越强，但检索、Reranker 和 Web Search 成本越高。100 人内部系统建议 2-4。 |

策略说明：

- `normalize`：本地去除多余空白，不调用 LLM。
- `direct`：把口语化问题改成独立检索问题。
- `hyde`：生成假想文档用于向量检索。
- `step_back`：生成更抽象的背景问题。
- `multi_query`：生成多个检索角度。

## Milvus Dense + Sparse 混合检索

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `MILVUS_COLLECTION` | `enterprise_rag_chunks_parent_child_v2` | Milvus collection 名称。 | Collection 除 Dense/BM25 字段外，还必须包含 Parent 正文、Parent ID、标题路径、页码范围和切分版本字段。 |
| `RETRIEVAL_MODE` | `hybrid` | 检索模式：`dense`、`sparse`、`hybrid`。 | 企业知识库建议默认 `hybrid`。旧 Milvus 不支持 BM25 function 时临时用 `dense`。 |
| `RETRIEVAL_DENSE_LIMIT` | `10` | 每个查询 dense 召回数量。 | 语义问题多时增大；过大会增加 Reranker 成本。 |
| `RETRIEVAL_SPARSE_LIMIT` | `10` | 每个查询 sparse/BM25 召回数量。 | 编号、条款、专有名词查询多时增大。 |
| `RETRIEVAL_CANDIDATE_COUNT` | `10` | 每个查询融合后候选上限。 | 通常不小于 dense/sparse 的单路候选期望值。 |
| `RETRIEVAL_RRF_K` | `60` | RRF 融合公式中的平滑参数。 | 值越小越强调排名靠前的结果；常用 60，先不建议频繁调整。 |
| `RETRIEVAL_MIN_SCORE` | 空 | Reranker 故障且 `RERANKER_FAILURE_STRATEGY=vector` 时的向量分数阈值。 | 只有选择 vector 回退才必须配置。Hybrid RRF 分数和 cosine/BM25 不同，需用评测集校准。 |
| `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT` | `3` | 默认本地 reranker 每个文档最多保留 chunk 数。 | 防止单文档刷屏。外部 Reranker 正常路径主要由 `RERANKER_TOP_K` 控制。 |

授权过滤在检索前完成，同一个 `version_uuid in [...]` 过滤条件必须同时作用于 dense 和 sparse 两路检索。

## Reranker 准入

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `RERANKER_TYPE` | `external` | Reranker 类型：`external`、`identity`、`default`。 | 生产要求 `external`。`identity` 和 `default` 主要用于测试或离线调试。 |
| `RERANKER_MODEL` | 空 | 外部 Reranker 模型名。 | 与服务商接口一致。未配置时 `/health/ready` 返回 503。 |
| `RERANKER_MIN_SCORE` | `0.5` | 外部 Reranker 最低通过分。 | 过高会频繁拒答，过低会引入噪声。用评测集按忠实度和召回率校准。 |
| `RERANKER_TOP_K` | `6` | 阈值过滤后最多进入最终证据池的 chunk 数。 | 增大会提升覆盖面，但增加生成上下文和引用噪声。常见范围 4-8。 |
| `RERANKER_FAILURE_STRATEGY` | `reject` | Reranker 故障策略：`reject`、`vector`、`llm`。 | 生产默认 `reject` 最保守。`vector` 需配置 `RETRIEVAL_MIN_SCORE`。`llm` 只作为降级兜底。 |
| `RELEVANCE_MAX_CONCURRENCY` | `4` | LLM 相关性评分故障回退时的并发数。 | 正常 Reranker 成功路径不会使用。 |
| `RELEVANCE_GRADING_ENABLED` | `true` | 历史兼容开关。 | 已废弃，正常查询链路不再由它控制。 |

Reranker 成功时不会逐 chunk 调用 LLM 评分。所有候选低于阈值时系统会结构化拒答，不强行选择 Top-K。

## 最终证据与答案生成

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `FINAL_CONTEXT_COUNT` | `6` | 进入答案生成的不同 Parent 上限。 | Parent 通常比 Child 长，提高该值会明显增加 prompt token；建议先保持 4-6。 |
| `RAG_MIN_RELEVANT_DOCUMENTS` | `1` | Parent 去重后生成答案所需的最少证据数。 | 高风险制度/合规问答可提高到 2，减少同一 Parent 多个 Child 造成的虚假证据数量。 |
| `RAG_CITATION_RETRY_COUNT` | `1` | 答案缺少有效 `[n]` 引用时的重试次数。 | 增大能提高引用合规率，但会增加 LLM 调用成本和延迟。 |

生成答案必须引用实际提供给模型的证据编号。无引用、引用越界或重试后仍无有效引用时，返回 `invalid_citations`。

## Web Search

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `WEB_SEARCH_ENABLED` | `true` | 是否启用外部搜索。 | 涉及最新信息或知识库不足时开启；强内网场景可关闭。 |
| `WEB_SEARCH_PROVIDER` | `tavily` | 外部搜索 provider。 | 当前生产只支持 Tavily。 |
| `WEB_SEARCH_MAX_QUERIES` | `1` | 单次问答最多搜索查询数。 | 默认 1 用于控制费用。若 `QUERY_REWRITE_MAX_QUERIES` 较大，不建议同步放大搜索次数。 |
| `WEB_SEARCH_RESULT_COUNT` | `5` | 每次搜索最多返回结果数。 | 结果越多 Reranker 成本越高。常用 3-8。 |
| `WEB_SEARCH_TIMEOUT_SECONDS` | `10` | Tavily 请求超时。 | 网络波动较大可提高到 15，但会影响用户等待时间。 |
| `WEB_SEARCH_MAX_RETRIES` | `2` | Tavily 429、连接错误、超时和 5xx 重试次数。 | 过高会拖慢失败请求。 |
| `TAVILY_SEARCH_DEPTH` | `basic` | Tavily 搜索深度：`basic` 或 `advanced`。 | `basic` 成本低、延迟低；需要更强覆盖时再试 `advanced`。 |
| `TAVILY_TOPIC` | `general` | Tavily 主题：`general`、`news`、`finance`。 | 内部知识库外部补充通常用 `general`。 |

Web Search 只发送搜索问题，不发送知识库 chunk、用户身份、ACL 或会话历史。搜索结果仍必须经过同一套 Reranker 准入，不能直接进入生成模型。

## Modular RAG 模块选择

| 配置项 | 默认值 | 作用 | 可选值 |
| --- | --- | --- | --- |
| `RAG_QUERY_MODULE` | `default` | 查询预处理模块。 | `default` |
| `RAG_RETRIEVER_MODULE` | `milvus` | 授权知识库召回模块。 | `milvus` |
| `RAG_FUSION_MODULE` | `rrf` | Dense/Sparse 与多查询融合模块。 | `rrf` |
| `RAG_ROUTER_MODULE` | `llm` | 证据路由模块。 | `llm` |
| `RAG_SELECTOR_MODULE` | `route_aware` | 最终证据选择模块。 | `route_aware` |
| `RAG_GENERATOR_MODULE` | `langchain` | 答案生成模块。 | `langchain` |
| `RAG_VALIDATOR_MODULE` | `bracket_citations` | 引用校验模块。 | `bracket_citations` |

这些配置目前用于固定拓扑下的白名单替换。未知模块名会在应用启动或 Pipeline 组装时失败，并提示模块类别和名称。系统不支持通过配置动态导入任意 Python 路径。

## LangSmith 可观测性

| 配置项 | 默认值 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `LANGSMITH_TRACING` | `false` | 是否启用 LangSmith trace。 | 调试链路时开启，生产按合规要求开启。 |
| `LANGSMITH_PROJECT` | `enterprise-crag` | LangSmith 项目名。 | 可按环境拆分，例如 `rag-dev`、`rag-prod`。 |
| `LANGSMITH_ENDPOINT` | 空 | LangSmith API 地址。 | 私有化或代理部署时配置。 |
| `LANGSMITH_HIDE_INPUTS` | `true` | 是否隐藏输入。 | 生产建议保持 `true`。 |
| `LANGSMITH_HIDE_OUTPUTS` | `true` | 是否隐藏输出。 | 生产建议保持 `true`。 |

如果需要排查具体 RAG 问题，可以临时在受控环境关闭隐藏输入输出；排查结束后应恢复。

## 认证、安全与运行配置

| 配置项 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT 有效期。 | 内部系统一般 15-60 分钟。 |
| `SECRET_KEY` | `change-me-in-production` | JWT 签名密钥。 | 生产必须替换为高强度随机值。 |
| `JWT_ALGORITHM` | `HS256` | JWT 签名算法。 | 当前默认即可，若接入 OIDC 再统一调整。 |
| `CORS_ORIGINS` | `http://localhost:8501` | 允许访问 FastAPI 的前端源。 | 生产只放真实域名。 |
| `DATABASE_URL` | MySQL 默认连接串 | MySQL 连接。 | 生产使用 Secret，不提交 `.env`。 |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker/result 或短期状态依赖。 | 单机可用 Redis；生产需配置持久化和监控。 |
| `MINIO_BUCKET` | `rag-documents` | 原始文档对象桶。 | 生产开启备份、生命周期和访问控制。 |
| `MINIO_SECURE` | `false` | MinIO 是否使用 HTTPS。 | 生产建议 `true` 或走内网 TLS 代理。 |
| `STREAMLIT_API_URL` | `http://localhost:8000` | Streamlit 前端访问后端地址。 | 容器、内网和反代部署时按实际地址修改。 |

## 推荐基线

### 低成本试点

```dotenv
QUERY_REWRITE_TYPES=normalize,multi_query
QUERY_REWRITE_MAX_QUERIES=2
RETRIEVAL_MODE=dense
RETRIEVAL_CANDIDATE_COUNT=8
RERANKER_MIN_SCORE=0.5
RERANKER_TOP_K=4
FINAL_CONTEXT_COUNT=4
WEB_SEARCH_MAX_QUERIES=1
WEB_SEARCH_RESULT_COUNT=3
```

适合数据量小、Milvus 暂不支持 BM25 function 或优先控制成本的环境。

### 企业知识库默认

```dotenv
QUERY_REWRITE_TYPES=normalize,direct,multi_query
QUERY_REWRITE_MAX_QUERIES=3
RETRIEVAL_MODE=hybrid
RETRIEVAL_DENSE_LIMIT=10
RETRIEVAL_SPARSE_LIMIT=10
RETRIEVAL_RRF_K=60
RERANKER_MIN_SCORE=0.5
RERANKER_TOP_K=6
FINAL_CONTEXT_COUNT=6
RAG_MIN_RELEVANT_DOCUMENTS=1
WEB_SEARCH_MAX_QUERIES=1
WEB_SEARCH_RESULT_COUNT=5
```

适合部门知识库、制度问答、客服知识库等常规场景。

### 高准确率保守模式

```dotenv
QUERY_REWRITE_TYPES=normalize,direct,hyde,multi_query
QUERY_REWRITE_MAX_QUERIES=4
RETRIEVAL_MODE=hybrid
RETRIEVAL_DENSE_LIMIT=15
RETRIEVAL_SPARSE_LIMIT=15
RETRIEVAL_CANDIDATE_COUNT=15
RERANKER_MIN_SCORE=0.65
RERANKER_TOP_K=8
FINAL_CONTEXT_COUNT=8
RAG_MIN_RELEVANT_DOCUMENTS=2
RAG_CITATION_RETRY_COUNT=2
RERANKER_FAILURE_STRATEGY=reject
```

适合合规、财务、人事制度等不能轻易误答的场景。代价是拒答率、延迟和外部模型调用成本都会上升。

## 调参观察指标

每次调参后至少观察：

- Recall@10：是否能把正确 chunk 召回到候选池。
- Reranker 通过率：`rag_diagnostics.reranking.*.passed_count / input_count`。
- 拒答分布：`refusal_detail` 中 `no_relevant_evidence`、`reranker_failed`、`invalid_citations` 的比例。
- 引用准确率：引用片段是否真实支撑答案。
- 忠实度：答案是否只依据引用证据。
- p95 延迟：特别关注 query rewrite、retrieval、reranking、web_search、generation。
- 成本：LLM 改写次数、Reranker 文档数、Tavily 搜索次数。

这些指标会持久化在 `messages.metrics` 中，其中 `timings` 保存阶段耗时，`rag_diagnostics` 保存查询、检索候选、Reranker 分数、Web Search 状态、最终证据数量和拒答详情。
