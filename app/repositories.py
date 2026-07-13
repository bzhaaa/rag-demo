from typing import Iterable, List, Optional, Sequence

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models import (
    DepartmentMembership,
    Document,
    DocumentACL,
    DocumentVersion,
    IngestionJob,
    Role,
    User,
    Visibility,
)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.scalar(
        select(User)
        .where(User.username == username)
        .options(
            selectinload(User.memberships).selectinload(
                DepartmentMembership.department
            )
        )
    )


def get_user_by_uuid(db: Session, user_uuid: str) -> Optional[User]:
    return db.scalar(
        select(User)
        .where(User.uuid == user_uuid)
        .options(
            selectinload(User.memberships).selectinload(
                DepartmentMembership.department
            )
        )
    )


def user_department_ids(user: User) -> List[int]:
    return [membership.department_id for membership in user.memberships]


def accessible_document_filter(user: User):
    if user.role == Role.admin.value:
        return Document.deleted_at.is_(None)

    department_ids = user_department_ids(user)
    conditions = [Document.owner_id == user.id]
    if department_ids:
        conditions.append(
            and_(
                Document.visibility == Visibility.department.value,
                Document.department_id.in_(department_ids),
            )
        )
        conditions.append(
            Document.acl_entries.any(
                DocumentACL.department_id.in_(department_ids)
            )
        )
    conditions.append(Document.acl_entries.any(DocumentACL.user_id == user.id))
    return and_(Document.deleted_at.is_(None), or_(*conditions))


def document_load_options():
    return (
        selectinload(Document.department),
        selectinload(Document.owner),
        selectinload(Document.current_version),
        selectinload(Document.versions),
        selectinload(Document.acl_entries).selectinload(DocumentACL.user),
        selectinload(Document.acl_entries).selectinload(DocumentACL.department),
    )


def list_accessible_documents(db: Session, user: User) -> Sequence[Document]:
    return db.scalars(
        select(Document)
        .where(accessible_document_filter(user))
        .options(*document_load_options())
        .order_by(Document.updated_at.desc())
    ).unique().all()


def get_accessible_document(
    db: Session, user: User, document_uuid: str
) -> Optional[Document]:
    return db.scalar(
        select(Document)
        .where(
            Document.uuid == document_uuid,
            accessible_document_filter(user),
        )
        .options(*document_load_options())
    )


def get_owned_document(
    db: Session, user: User, document_uuid: str
) -> Optional[Document]:
    statement = select(Document).where(
        Document.uuid == document_uuid, Document.deleted_at.is_(None)
    )
    if user.role != Role.admin.value:
        statement = statement.where(Document.owner_id == user.id)
    return db.scalar(statement.options(*document_load_options()))


def get_job_for_user(
    db: Session, user: User, job_uuid: str
) -> Optional[IngestionJob]:
    statement = select(IngestionJob).where(IngestionJob.uuid == job_uuid)
    if user.role != Role.admin.value:
        statement = statement.where(IngestionJob.requested_by_id == user.id)
    return db.scalar(statement)


def active_versions_for_user(
    db: Session, user: User
) -> Iterable[DocumentVersion]:
    return db.scalars(
        select(DocumentVersion)
        .join(Document, Document.current_version_id == DocumentVersion.id)
        .where(accessible_document_filter(user))
    ).all()
