# API

所有业务 API 都位于 `/api/v1` 下，并使用 Bearer JWT。

- `POST /auth/login`：登录本地账号。
- `GET /auth/me`：返回当前用户和所属部门。
- `POST /documents`：上传 PDF、TXT 或 Markdown 文件。
- `GET /documents`：列出当前用户有权访问的文档。
- `POST /documents/{uuid}/versions`：上传新版本；索引成功前不会替换活跃版本。
- `GET /documents/{uuid}`：返回文档详情、版本和 ACL。
- `PUT /documents/{uuid}/acl`：替换可见性和显式 ACL。
- `DELETE /documents/{uuid}`：软删除文档并投递向量清理任务。
- `GET /jobs/{uuid}`：返回当前用户有权查看的入库任务。
- `POST /queries`：执行 RAG/CRAG 查询，只返回答案实际引用的证据。
- `GET /conversations`：列出当前用户的会话。

健康检查位于 `/health/live` 和 `/health/ready`。

上传时客户端可以指定目标部门，但后端会校验成员关系。查询客户端不能提交部门 ID、文档 ID 或 Milvus 过滤表达式。

## 查询引用

`POST /queries` 的顶层响应字段保持不变。每条 `citation` 包含：

- `source_type`：`knowledge_base` 或 `web`。
- `url`：网络证据返回 URL，知识库证据为 `null`。
- `document_uuid` 和 `version`：知识库证据返回对应值，网络证据为 `null`。
- `document_title`、`page_number`、`chunk_id` 和 `excerpt`。

网络降级默认开启，使用 Tavily Search API。知识库证据可由三档路由选择为纯知识库、纯网络或混合证据；Web 候选仍须通过专业 Reranker 后才能用于生成。Tavily 故障时返回结构化拒答 `web_search_failed`，不会使用 Mock 或未经验证的替代内容。外部搜索会产生费用并发送用户搜索问题，生产环境应配置密钥管理、出域控制和审计策略。
