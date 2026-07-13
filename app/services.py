import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException, status
from pypdf import PdfReader
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    AuditLog,
    Department,
    Document,
    DocumentACL,
    DocumentVersion,
    IngestionJob,
    JobStatus,
    Role,
    User,
    VersionStatus,
    Visibility,
)
from app.repositories import get_user_by_username, user_department_ids
from app.security import create_access_token, verify_password

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    },
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def authenticate(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def login_token(user: User) -> str:
    return create_access_token(user.uuid, {"role": user.role})


def write_audit(
    db: Session,
    action: str,
    resource_type: str,
    actor: Optional[User] = None,
    resource_uuid: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    entry = AuditLog(
        actor_user_id=actor.id if actor else None,
        action=action,
        resource_type=resource_type,
        resource_uuid=resource_uuid,
        details=details or {},
        ip_address=ip_address,
    )
    db.add(entry)
    return entry


def validate_upload(filename: str, content_type: str, data: bytes) -> Dict[str, Any]:
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported file type")
    if content_type not in ALLOWED_MIME_TYPES[suffix]:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unexpected MIME type")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File is too large")

    page_count = None
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Invalid PDF header")
        try:
            reader = PdfReader(__import__("io").BytesIO(data))
        except Exception as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid PDF file") from exc
        if reader.is_encrypted:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Encrypted PDFs are not supported")
        page_count = len(reader.pages)
        if page_count > settings.max_pdf_pages:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "PDF has too many pages")
    elif b"\x00" in data[:4096]:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Binary text files are not supported")

    return {
        "suffix": suffix,
        "checksum": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "page_count": page_count,
    }


def resolve_upload_department(
    db: Session, user: User, department_uuid: Optional[str]
) -> Department:
    if department_uuid:
        department = db.scalar(
            select(Department).where(Department.uuid == department_uuid)
        )
        if department is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Department not found")
        if user.role != Role.admin.value and department.id not in user_department_ids(user):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Department is not available")
        return department
    if not user.memberships:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User has no department")
    return user.memberships[0].department


def resolve_acl_subjects(
    db: Session,
    user_uuids_json: str,
    department_uuids_json: str,
) -> Tuple[List[User], List[Department]]:
    try:
        user_uuids = json.loads(user_uuids_json or "[]")
        department_uuids = json.loads(department_uuids_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "ACL must be JSON arrays") from exc
    if not isinstance(user_uuids, list) or not isinstance(department_uuids, list):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "ACL must be JSON arrays")
    users = list(db.scalars(select(User).where(User.uuid.in_(user_uuids))).all()) if user_uuids else []
    departments = (
        list(db.scalars(select(Department).where(Department.uuid.in_(department_uuids))).all())
        if department_uuids
        else []
    )
    if len(users) != len(set(user_uuids)) or len(departments) != len(set(department_uuids)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown ACL subject")
    return users, departments


def create_document_upload(
    db: Session,
    actor: User,
    title: str,
    filename: str,
    content_type: str,
    data: bytes,
    department: Department,
    visibility: str,
    acl_users: Iterable[User],
    acl_departments: Iterable[Department],
    object_key: str,
) -> Tuple[Document, DocumentVersion, IngestionJob]:
    if visibility not in {item.value for item in Visibility}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid visibility")
    file_info = validate_upload(filename, content_type, data)

    existing = db.scalar(
        select(DocumentVersion).where(DocumentVersion.checksum == file_info["checksum"])
    )
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This file already exists")

    document = Document(
        title=title.strip() or filename,
        owner_id=actor.id,
        department_id=department.id,
        visibility=visibility,
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        checksum=file_info["checksum"],
        object_key=object_key,
        source_name=filename,
        mime_type=content_type,
        size_bytes=file_info["size_bytes"],
        page_count=file_info["page_count"],
        status=VersionStatus.pending.value,
    )
    db.add(version)
    db.flush()
    for acl_user in acl_users:
        db.add(DocumentACL(document_id=document.id, user_id=acl_user.id))
    for acl_department in acl_departments:
        db.add(
            DocumentACL(
                document_id=document.id, department_id=acl_department.id
            )
        )
    job = IngestionJob(
        document_version_id=version.id,
        requested_by_id=actor.id,
        idempotency_key=f"ingest:{file_info['checksum']}",
        status=JobStatus.queued.value,
        stage="queued",
        progress=0,
    )
    db.add(job)
    write_audit(
        db,
        "document.upload",
        "document",
        actor,
        document.uuid,
        {
            "version_uuid": version.uuid,
            "checksum": version.checksum,
            "department_uuid": department.uuid,
        },
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Duplicate upload request")
    db.refresh(document)
    db.refresh(version)
    db.refresh(job)
    return document, version, job


def create_document_version_upload(
    db: Session,
    actor: User,
    document: Document,
    filename: str,
    content_type: str,
    data: bytes,
    object_key: str,
) -> Tuple[DocumentVersion, IngestionJob]:
    file_info = validate_upload(filename, content_type, data)
    if document.deleted_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    if db.scalar(
        select(DocumentVersion).where(
            DocumentVersion.checksum == file_info["checksum"]
        )
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "This file already exists")

    document_row = db.scalar(
        select(Document)
        .where(Document.id == document.id)
        .with_for_update()
    )
    version = DocumentVersion(
        document_id=document_row.id,
        version_number=next_version_number(db, document_row.id),
        checksum=file_info["checksum"],
        object_key=object_key,
        source_name=filename,
        mime_type=content_type,
        size_bytes=file_info["size_bytes"],
        page_count=file_info["page_count"],
        status=VersionStatus.pending.value,
    )
    db.add(version)
    db.flush()
    job = IngestionJob(
        document_version_id=version.id,
        requested_by_id=actor.id,
        idempotency_key=f"ingest:{file_info['checksum']}",
        status=JobStatus.queued.value,
        stage="queued",
        progress=0,
    )
    db.add(job)
    write_audit(
        db,
        "document.version_upload",
        "document",
        actor,
        document.uuid,
        {"version_uuid": version.uuid, "version": version.version_number},
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Duplicate upload request")
    db.refresh(version)
    db.refresh(job)
    return version, job


def activate_version(db: Session, version_id: int, chunk_count: int) -> None:
    version = db.scalar(
        select(DocumentVersion)
        .where(DocumentVersion.id == version_id)
        .with_for_update()
    )
    if version is None:
        raise ValueError("Document version not found")
    document = db.scalar(
        select(Document).where(Document.id == version.document_id).with_for_update()
    )
    if document is None or document.deleted_at is not None:
        raise ValueError("Document is unavailable")
    version.status = VersionStatus.ready.value
    version.chunk_count = chunk_count
    version.error_message = None
    document.current_version_id = version.id
    document.updated_at = utcnow()
    db.commit()


def create_delete_job(db: Session, actor: User, document: Document) -> IngestionJob:
    if document.current_version_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document has no active version")
    document.deleted_at = utcnow()
    job = IngestionJob(
        document_version_id=document.current_version_id,
        requested_by_id=actor.id,
        idempotency_key=f"delete:{document.uuid}:{int(utcnow().timestamp())}",
        status=JobStatus.deleting.value,
        stage="deleting",
        progress=0,
    )
    db.add(job)
    write_audit(db, "document.delete", "document", actor, document.uuid)
    db.commit()
    db.refresh(job)
    return job


def replace_document_acl(
    db: Session,
    actor: User,
    document: Document,
    visibility: str,
    users: Iterable[User],
    departments: Iterable[Department],
) -> None:
    if visibility not in {item.value for item in Visibility}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid visibility")
    document.visibility = visibility
    for entry in list(document.acl_entries):
        db.delete(entry)
    db.flush()
    for user in users:
        db.add(DocumentACL(document_id=document.id, user_id=user.id))
    for department in departments:
        db.add(
            DocumentACL(
                document_id=document.id, department_id=department.id
            )
        )
    write_audit(
        db,
        "document.acl_update",
        "document",
        actor,
        document.uuid,
        {
            "visibility": visibility,
            "user_uuids": [user.uuid for user in users],
            "department_uuids": [
                department.uuid for department in departments
            ],
        },
    )
    db.commit()


def next_version_number(db: Session, document_id: int) -> int:
    return int(
        db.scalar(
            select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(
                DocumentVersion.document_id == document_id
            )
        )
        or 0
    ) + 1
