# API

All business APIs are under `/api/v1` and use a bearer JWT.

- `POST /auth/login`: authenticate a mock local account.
- `GET /auth/me`: return the current user and departments.
- `POST /documents`: upload a local PDF, TXT, or Markdown file.
- `GET /documents`: list documents authorized for the current user.
- `POST /documents/{uuid}/versions`: upload a new version without replacing the active version until indexing succeeds.
- `GET /documents/{uuid}`: return document details, versions, and ACL.
- `PUT /documents/{uuid}/acl`: replace visibility and explicit ACL entries.
- `DELETE /documents/{uuid}`: soft-delete and enqueue vector cleanup.
- `GET /jobs/{uuid}`: return an authorized ingestion job.
- `POST /queries`: answer from authorized active versions and return citations.
- `GET /conversations`: list the current user's conversations.

Health checks are exposed at `/health/live` and `/health/ready`.

Clients may request a target department during upload, but the backend verifies membership. Query clients cannot submit department IDs, document IDs, or Milvus filters.
