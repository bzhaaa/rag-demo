import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

PRIMARY_KEY_TYPE = BigInteger().with_variant(Integer, "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Role(str, enum.Enum):
    admin = "admin"
    editor = "editor"
    viewer = "viewer"


class Visibility(str, enum.Enum):
    department = "department"
    restricted = "restricted"


class VersionStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"
    deleted = "deleted"


class JobStatus(str, enum.Enum):
    queued = "queued"
    parsing = "parsing"
    embedding = "embedding"
    indexing = "indexing"
    activating = "activating"
    ready = "ready"
    failed = "failed"
    deleting = "deleting"


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UUIDMixin:
    id: Mapped[int] = mapped_column(
        PRIMARY_KEY_TYPE, primary_key=True, autoincrement=True
    )
    uuid: Mapped[str] = mapped_column(
        String(36), default=new_uuid, unique=True, nullable=False, index=True
    )


class Department(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("departments.id", ondelete="RESTRICT"), nullable=True
    )

    parent: Mapped[Optional["Department"]] = relationship(remote_side="Department.id")
    memberships: Mapped[list["DepartmentMembership"]] = relationship(
        back_populates="department", cascade="all, delete-orphan"
    )


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default=Role.viewer.value, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    memberships: Mapped[list["DepartmentMembership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class DepartmentMembership(TimestampMixin, Base):
    __tablename__ = "department_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "department_id", name="uq_membership_user_department"),
        Index("ix_membership_department_user", "department_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        PRIMARY_KEY_TYPE, primary_key=True, autoincrement=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="memberships")
    department: Mapped[Department] = relationship(back_populates="memberships")


class Document(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_department_visibility", "department_id", "visibility"),
        Index("ix_documents_owner_deleted", "owner_id", "deleted_at"),
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="RESTRICT"), nullable=False
    )
    visibility: Mapped[str] = mapped_column(
        String(20), default=Visibility.department.value, nullable=False
    )
    current_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(
            "document_versions.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_documents_current_version",
        ),
        nullable=True,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    owner: Mapped[User] = relationship(foreign_keys=[owner_id])
    department: Mapped[Department] = relationship()
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        cascade="all, delete-orphan",
    )
    current_version: Mapped[Optional["DocumentVersion"]] = relationship(
        foreign_keys=[current_version_id], post_update=True
    )
    acl_entries: Mapped[list["DocumentACL"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentACL(TimestampMixin, Base):
    __tablename__ = "document_acl"
    __table_args__ = (
        UniqueConstraint("document_id", "user_id", name="uq_document_acl_user"),
        UniqueConstraint(
            "document_id", "department_id", name="uq_document_acl_department"
        ),
        CheckConstraint(
            "(user_id IS NOT NULL AND department_id IS NULL) OR "
            "(user_id IS NULL AND department_id IS NOT NULL)",
            name="ck_document_acl_one_subject",
        ),
        Index("ix_document_acl_user_document", "user_id", "document_id"),
        Index("ix_document_acl_department_document", "department_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(
        PRIMARY_KEY_TYPE, primary_key=True, autoincrement=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    department_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=True
    )
    permission: Mapped[str] = mapped_column(String(20), default="read", nullable=False)

    document: Mapped[Document] = relationship(back_populates="acl_entries")
    user: Mapped[Optional[User]] = relationship()
    department: Mapped[Optional[Department]] = relationship()


class DocumentVersion(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_version"),
        UniqueConstraint("checksum", name="uq_document_checksum"),
        Index("ix_version_status_created", "status", "created_at"),
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_name: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=VersionStatus.pending.value, nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    document: Mapped[Document] = relationship(
        back_populates="versions", foreign_keys=[document_id]
    )


class IngestionJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_idempotency"),
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_version_status", "document_version_id", "status"),
    )

    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    requested_by_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.queued.value, nullable=False
    )
    stage: Mapped[str] = mapped_column(String(50), default="queued", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    document_version: Mapped[DocumentVersion] = relationship()
    requested_by: Mapped[User] = relationship()


class Conversation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)

    user: Mapped[User] = relationship()
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_actor_created", "actor_user_id", "created_at"),
        Index("ix_audit_action_created", "action", "created_at"),
    )

    id: Mapped[int] = mapped_column(
        PRIMARY_KEY_TYPE, primary_key=True, autoincrement=True
    )
    uuid: Mapped[str] = mapped_column(
        String(36), default=new_uuid, unique=True, nullable=False
    )
    actor_user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_uuid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    actor: Mapped[Optional[User]] = relationship()
