from typing import Any, Dict, List, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document


def cited_chunks(
    chunks: Sequence[Dict[str, Any]],
    cited_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    result = []
    for index in cited_indices:
        position = index - 1
        if 0 <= position < len(chunks):
            result.append(chunks[position])
    return result


def assemble_citations(
    db: Session, chunks: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    document_uuids = {
        item["document_uuid"]
        for item in chunks
        if item.get("source_type") != "web" and item.get("document_uuid")
    }
    documents = (
        {
            document.uuid: document
            for document in db.scalars(
                select(Document).where(Document.uuid.in_(document_uuids))
            )
        }
        if document_uuids
        else {}
    )
    citations: List[Dict[str, Any]] = []
    for item in chunks:
        if item.get("source_type") == "web":
            citations.append(
                {
                    "source_type": "web",
                    "url": item.get("url"),
                    "document_uuid": None,
                    "document_title": (
                        item.get("source_name")
                        or item.get("title")
                        or "Web source"
                    ),
                    "version": None,
                    "page_number": None,
                    "chunk_id": item["chunk_id"],
                    "excerpt": item["content"][:300],
                }
            )
            continue
        document_uuid = str(item.get("document_uuid") or "")
        document = documents.get(document_uuid)
        if document is None:
            continue
        citations.append(
            {
                "source_type": "knowledge_base",
                "url": None,
                "document_uuid": item["document_uuid"],
                "document_title": document.title,
                "version": item["version_number"],
                "page_number": item.get("page_number") or None,
                "chunk_id": item["chunk_id"],
                "excerpt": (
                    item.get("citation_content") or item["content"]
                )[:300],
            }
        )
    return citations
