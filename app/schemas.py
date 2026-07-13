from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserSummary(ORMModel):
    uuid: str
    username: str
    email: str
    role: str
    departments: list["DepartmentSummary"] = Field(default_factory=list)


class DepartmentSummary(ORMModel):
    uuid: str
    name: str


class DocumentVersionResponse(ORMModel):
    uuid: str
    version_number: int
    checksum: str
    source_name: str
    mime_type: str
    size_bytes: int
    page_count: Optional[int]
    chunk_count: Optional[int]
    status: str
    error_message: Optional[str]
    created_at: datetime


class DocumentResponse(ORMModel):
    uuid: str
    title: str
    visibility: str
    department: DepartmentSummary
    owner: UserSummary
    current_version: Optional[DocumentVersionResponse]
    created_at: datetime
    updated_at: datetime
    versions: list[DocumentVersionResponse] = Field(default_factory=list)
    acl_users: list[UserSummary] = Field(default_factory=list)
    acl_departments: list[DepartmentSummary] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_uuid: str
    version_uuid: str
    job_uuid: str
    status: str


class DocumentACLUpdateRequest(BaseModel):
    visibility: str
    user_uuids: list[str] = Field(default_factory=list)
    department_uuids: list[str] = Field(default_factory=list)


class JobResponse(ORMModel):
    uuid: str
    status: str
    stage: str
    progress: int
    retry_count: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class Citation(BaseModel):
    document_uuid: str
    document_title: str
    version: int
    page_number: Optional[int]
    chunk_id: str
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_uuid: Optional[str] = None


class QueryResponse(BaseModel):
    conversation_uuid: str
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: Optional[str] = None
    trace_id: Optional[str] = None
    timings: dict[str, float] = Field(default_factory=dict)


class MessageResponse(ORMModel):
    uuid: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime


class ConversationResponse(ORMModel):
    uuid: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, str] = Field(default_factory=dict)
