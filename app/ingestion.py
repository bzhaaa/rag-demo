from io import BytesIO
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import get_settings
from app.models import Document, DocumentVersion
from app.vector_store import chunk_id


def parse_document(data: bytes, mime_type: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    if mime_type == "application/pdf":
        reader = PdfReader(BytesIO(data))
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page_number": index})
    else:
        text = data.decode("utf-8-sig")
        if text.strip():
            pages.append({"text": text, "page_number": None})
    return pages


def build_chunks(
    document: Document,
    version: DocumentVersion,
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        for text in splitter.split_text(page["text"]):
            index = len(chunks)
            chunks.append(
                {
                    "chunk_id": chunk_id(
                        document.uuid, version.version_number, index
                    ),
                    "document_uuid": document.uuid,
                    "version_uuid": version.uuid,
                    "version_number": version.version_number,
                    "department_uuid": document.department.uuid,
                    "visibility": document.visibility,
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "source_name": version.source_name,
                    "content": text,
                }
            )
    return chunks
