import argparse
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.ingestion import build_chunks, chunking_metadata, parse_document
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


def rebuild_version(version: DocumentVersion, dry_run: bool = False) -> int:
    storage = ObjectStorage()
    vector_store = MilvusChunkStore()
    data = storage.download(version.object_key)
    pages = parse_document(data, version.mime_type)
    if not pages:
        raise ValueError(f"No readable text was extracted: {version.uuid}")
    chunks = build_chunks(version.document, version, pages)
    if dry_run:
        return len(chunks)
    vector_store.delete_version(version.uuid)
    return vector_store.insert_chunks(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild ready active document versions into the configured "
            "Milvus parent-child hybrid collection."
        )
    )
    parser.add_argument(
        "--version-uuid",
        default="",
        help="Rebuild only one active ready document version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count chunks without deleting or writing vectors.",
    )
    args = parser.parse_args()

    rebuilt = 0
    with SessionLocal() as db:
        versions = list(db.scalars(version_statement(args.version_uuid)).all())
        for version in versions:
            inserted = rebuild_version(version, dry_run=args.dry_run)
            rebuilt += 1
            action = "Planned" if args.dry_run else "Rebuilt"
            print(f"{action} {version.uuid}: {inserted} child chunks")
            if not args.dry_run:
                version.metadata_json = {
                    **(version.metadata_json or {}),
                    "chunking": chunking_metadata(),
                }
                db.commit()
    print(
        "Parent-child index rebuild complete. "
        f"Versions processed: {rebuilt}"
    )


if __name__ == "__main__":
    main()
