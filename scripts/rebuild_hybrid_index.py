import argparse
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.ingestion import build_chunks, parse_document
from app.models import Document, DocumentVersion, VersionStatus
from app.storage import ObjectStorage
from app.vector_store import MilvusChunkStore


def version_statement(version_uuid: str = ""):
    statement = (
        select(DocumentVersion)
        .join(Document, Document.current_version_id == DocumentVersion.id)
        .where(
            Document.deleted_at.is_(None),
            DocumentVersion.status == VersionStatus.ready.value,
        )
        .options(
            selectinload(DocumentVersion.document).selectinload(
                Document.department
            )
        )
        .order_by(DocumentVersion.id)
    )
    if version_uuid:
        statement = statement.where(DocumentVersion.uuid == version_uuid)
    return statement


def rebuild_version(version: DocumentVersion) -> int:
    storage = ObjectStorage()
    vector_store = MilvusChunkStore()
    data = storage.download(version.object_key)
    pages = parse_document(data, version.mime_type)
    if not pages:
        raise ValueError(f"No readable text was extracted: {version.uuid}")
    chunks = build_chunks(version.document, version, pages)
    vector_store.delete_version(version.uuid)
    return vector_store.insert_chunks(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ready active document versions into the configured Milvus hybrid collection."
    )
    parser.add_argument(
        "--version-uuid",
        default="",
        help="Rebuild only one active ready document version.",
    )
    args = parser.parse_args()

    rebuilt = 0
    with SessionLocal() as db:
        versions = list(db.scalars(version_statement(args.version_uuid)).all())
        for version in versions:
            inserted = rebuild_version(version)
            rebuilt += 1
            print(f"Rebuilt {version.uuid}: {inserted} chunks")
    print(f"Hybrid index rebuild complete. Versions rebuilt: {rebuilt}")


if __name__ == "__main__":
    main()
