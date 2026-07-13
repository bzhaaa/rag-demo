from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db import get_db
from app.dependencies import get_current_user, require_roles
from app.models import (
    Conversation,
    Document,
    Role,
    User,
)
from app.rag import RAGService, get_rag_service
from app.repositories import (
    get_accessible_document,
    get_job_for_user,
    get_owned_document,
    list_accessible_documents,
)
from app.schemas import (
    ConversationResponse,
    DocumentACLUpdateRequest,
    DocumentResponse,
    DocumentUploadResponse,
    JobResponse,
    LoginRequest,
    QueryRequest,
    QueryResponse,
    TokenResponse,
    UserSummary,
)
from app.services import (
    authenticate,
    create_delete_job,
    create_document_upload,
    create_document_version_upload,
    login_token,
    replace_document_acl,
    resolve_acl_subjects,
    resolve_upload_department,
    validate_upload,
    write_audit,
)
from app.storage import ObjectStorage
from app.tasks import delete_document_vectors, ingest_document
from app.vector_store import MilvusChunkStore

settings = get_settings()
router = APIRouter(prefix=settings.api_prefix)


def user_payload(user: User) -> dict:
    return {
        "uuid": user.uuid,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "departments": [
            {"uuid": item.department.uuid, "name": item.department.name}
            for item in user.memberships
        ],
    }


def document_payload(document: Document) -> dict:
    return {
        "uuid": document.uuid,
        "title": document.title,
        "visibility": document.visibility,
        "department": {
            "uuid": document.department.uuid,
            "name": document.department.name,
        },
        "owner": {
            "uuid": document.owner.uuid,
            "username": document.owner.username,
            "email": document.owner.email,
            "role": document.owner.role,
            "departments": [],
        },
        "current_version": document.current_version,
        "versions": sorted(
            document.versions, key=lambda item: item.version_number, reverse=True
        ),
        "acl_users": [
            {
                "uuid": entry.user.uuid,
                "username": entry.user.username,
                "email": entry.user.email,
                "role": entry.user.role,
                "departments": [],
            }
            for entry in document.acl_entries
            if entry.user is not None
        ],
        "acl_departments": [
            {"uuid": entry.department.uuid, "name": entry.department.name}
            for entry in document.acl_entries
            if entry.department is not None
        ],
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


@router.post("/auth/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = authenticate(db, payload.username, payload.password)
    if user is None:
        write_audit(
            db,
            "auth.login_failed",
            "user",
            details={"username": payload.username},
            ip_address=request.client.host if request.client else None,
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    write_audit(
        db,
        "auth.login",
        "user",
        user,
        user.uuid,
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    return TokenResponse(access_token=login_token(user))


@router.get("/auth/me", response_model=UserSummary)
def me(user: User = Depends(get_current_user)) -> dict:
    return user_payload(user)


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    department_uuid: Optional[str] = Form(None),
    visibility: str = Form("department"),
    acl_user_uuids: str = Form("[]"),
    acl_department_uuids: str = Form("[]"),
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles(Role.admin, Role.editor)),
) -> DocumentUploadResponse:
    data = await file.read()
    file_info = validate_upload(file.filename or "upload", file.content_type or "", data)
    department = resolve_upload_department(db, actor, department_uuid)
    acl_users, acl_departments = resolve_acl_subjects(
        db, acl_user_uuids, acl_department_uuids
    )
    safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    object_key = f"uploads/{file_info['checksum']}/{safe_name}"
    document, version, job = create_document_upload(
        db,
        actor,
        title,
        safe_name,
        file.content_type or "",
        data,
        department,
        visibility,
        acl_users,
        acl_departments,
        object_key,
    )
    try:
        ObjectStorage().upload(
            object_key, data, file.content_type or "application/octet-stream"
        )
    except Exception:
        db.delete(job)
        db.delete(version)
        db.delete(document)
        db.commit()
        raise
    result = ingest_document.delay(job.id)
    job.task_id = result.id
    write_audit(
        db,
        "document.ingestion_enqueued",
        "ingestion_job",
        actor,
        job.uuid,
        {"document_uuid": document.uuid, "task_id": result.id},
    )
    db.commit()
    return DocumentUploadResponse(
        document_uuid=document.uuid,
        version_uuid=version.uuid,
        job_uuid=job.uuid,
        status=job.status,
    )


@router.get("/documents", response_model=List[DocumentResponse])
def documents(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> List[dict]:
    return [document_payload(item) for item in list_accessible_documents(db, user)]


@router.post(
    "/documents/{document_uuid}/versions",
    response_model=DocumentUploadResponse,
)
async def upload_document_version(
    document_uuid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles(Role.admin, Role.editor)),
) -> DocumentUploadResponse:
    document = get_owned_document(db, actor, document_uuid)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    data = await file.read()
    file_info = validate_upload(
        file.filename or "upload", file.content_type or "", data
    )
    safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    object_key = f"uploads/{file_info['checksum']}/{safe_name}"
    version, job = create_document_version_upload(
        db,
        actor,
        document,
        safe_name,
        file.content_type or "",
        data,
        object_key,
    )
    try:
        ObjectStorage().upload(
            object_key, data, file.content_type or "application/octet-stream"
        )
    except Exception:
        db.delete(job)
        db.delete(version)
        db.commit()
        raise
    result = ingest_document.delay(job.id)
    job.task_id = result.id
    write_audit(
        db,
        "document.ingestion_enqueued",
        "ingestion_job",
        actor,
        job.uuid,
        {"document_uuid": document.uuid, "task_id": result.id},
    )
    db.commit()
    return DocumentUploadResponse(
        document_uuid=document.uuid,
        version_uuid=version.uuid,
        job_uuid=job.uuid,
        status=job.status,
    )


@router.get("/documents/{document_uuid}", response_model=DocumentResponse)
def document_detail(
    document_uuid: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    document = get_accessible_document(db, user, document_uuid)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document_payload(document)


@router.put("/documents/{document_uuid}/acl", response_model=DocumentResponse)
def update_document_acl(
    document_uuid: str,
    payload: DocumentACLUpdateRequest,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles(Role.admin, Role.editor)),
) -> dict:
    document = get_owned_document(db, actor, document_uuid)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    users, departments = resolve_acl_subjects(
        db,
        __import__("json").dumps(payload.user_uuids),
        __import__("json").dumps(payload.department_uuids),
    )
    replace_document_acl(
        db, actor, document, payload.visibility, users, departments
    )
    refreshed = get_owned_document(db, actor, document_uuid)
    try:
        MilvusChunkStore().update_document_metadata(
            refreshed.uuid,
            refreshed.department.uuid,
            refreshed.visibility,
        )
    except Exception as exc:
        write_audit(
            db,
            "document.vector_metadata_sync_failed",
            "document",
            actor,
            refreshed.uuid,
            {"error": str(exc)[:1000]},
        )
        db.commit()
    return document_payload(refreshed)


@router.delete("/documents/{document_uuid}", response_model=JobResponse)
def delete_document(
    document_uuid: str,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles(Role.admin, Role.editor)),
) -> object:
    document = get_owned_document(db, actor, document_uuid)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    job = create_delete_job(db, actor, document)
    result = delete_document_vectors.delay(job.id)
    job.task_id = result.id
    write_audit(
        db,
        "document.cleanup_enqueued",
        "ingestion_job",
        actor,
        job.uuid,
        {"document_uuid": document.uuid, "task_id": result.id},
    )
    db.commit()
    return job


@router.get("/jobs/{job_uuid}", response_model=JobResponse)
def job_status(
    job_uuid: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> object:
    job = get_job_for_user(db, user, job_uuid)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.post("/queries", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        return rag_service.answer(
            db, user, payload.question, payload.conversation_uuid
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"RAG service unavailable: {exc}",
        ) from exc


@router.get("/conversations", response_model=List[ConversationResponse])
def conversations(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> object:
    return (
        db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.updated_at.desc())
        )
        .unique()
        .all()
    )
