# Test Plan

Unit tests cover password hashing, JWT validation, ACL predicates, deterministic chunk IDs, upload validation, citation assembly, and structured refusal.

API tests use dependency overrides and infrastructure fakes. MySQL and Milvus integration tests are opt-in and run against Docker services.

Release validation includes:

- Alembic migration on MySQL 8.0.
- Duplicate upload and version activation transaction checks.
- Department, explicit-user, owner, and administrator access combinations.
- Worker retry, model timeout, and Milvus failure behavior.
- Real MinIO upload and Milvus insert, filtered search, and deletion.
- Recall@10, citation accuracy, and faithfulness evaluation datasets.
- Twenty-user load tests with query p95 at most 15 seconds and upload acknowledgement p95 at most 2 seconds.
- Backup and restore exercises using `mysqldump` or Percona XtraBackup.
