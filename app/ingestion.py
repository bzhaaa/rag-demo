from typing import Any, Dict, List

from app.chunking import (
    ParsedBlock,
    StructureAwareDocumentParser,
    create_chunking_strategy,
)
from app.config import get_settings
from app.models import Document, DocumentVersion
from app.vector_store import chunk_id, parent_chunk_id


def parse_document(data: bytes, mime_type: str) -> List[Dict[str, Any]]:
    """Compatibility entry point returning serializable parsed blocks."""
    return [
        block.to_mapping()
        for block in StructureAwareDocumentParser().parse(data, mime_type)
    ]


def build_chunks(
    document: Document,
    version: DocumentVersion,
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compatibility entry point backed by the configured chunking strategy."""
    settings = get_settings()
    records = create_chunking_strategy(settings).chunk(
        [ParsedBlock.from_mapping(page) for page in pages]
    )
    chunks: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        parent_id = parent_chunk_id(
            document.uuid,
            version.version_number,
            record.parent_index,
        )
        chunks.append(
            {
                "chunk_id": chunk_id(
                    document.uuid,
                    version.version_number,
                    index,
                ),
                "document_uuid": document.uuid,
                "version_uuid": version.uuid,
                "version_number": version.version_number,
                "department_uuid": document.department.uuid,
                "visibility": document.visibility,
                "page_number": record.page_start,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "chunk_index": index,
                "parent_chunk_id": parent_id,
                "parent_content": record.parent_content,
                "parent_index": record.parent_index,
                "section_title": record.section_title,
                "section_path": " > ".join(record.section_path),
                "chunking_strategy": record.chunking_strategy,
                "chunking_version": record.chunking_version,
                "source_name": version.source_name,
                "content": record.content,
            }
        )
    return chunks


def chunking_metadata() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "strategy": settings.chunking_strategy,
        "version": settings.chunking_version,
        "child_size": settings.chunk_size,
        "child_overlap": settings.chunk_overlap,
        "parent_size": settings.chunk_parent_size,
        "parent_overlap": settings.chunk_parent_overlap,
        "min_size": settings.chunk_min_size,
        "context_header_enabled": settings.chunk_context_header_enabled,
    }
