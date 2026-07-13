# Data Model

MySQL uses InnoDB, `utf8mb4`, UTC timestamps, internal numeric keys, and public UUIDs.

- `users`, `departments`, and `department_memberships` define identities and organization.
- `documents` owns visibility, owner, department, soft-delete state, and the active version pointer.
- `document_acl` grants read access to exactly one user or one department per row.
- `document_versions` stores checksums, MinIO object keys, processing state, and chunk counts.
- `ingestion_jobs` stores asynchronous stage, progress, retry count, task ID, and errors.
- `conversations` and `messages` store questions, answers, citations, timings, models, and trace IDs.
- `audit_logs` is append-only from business APIs.

The SHA-256 checksum is globally unique in the first release to prevent duplicate storage. Vector chunk IDs follow `document_uuid:version_number:chunk_index`.
