from typing import Optional

from celery import Task
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.celery_app import celery_app
from app.db import SessionLocal
from app.ingestion import build_chunks, chunking_metadata, parse_document
from app.models import (
    Document,
    DocumentVersion,
    IngestionJob,
    JobStatus,
    VersionStatus,
)
from app.services import activate_version
from app.storage import ObjectStorage
from app.vector_store import MilvusChunkStore


def update_job(
    job_id: int,
    status: str,
    stage: str,
    progress: int,
    error_message: Optional[str] = None,
) -> None:
    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = status
        job.stage = stage
        job.progress = progress
        job.error_message = error_message
        db.commit()


class RetriableTask(Task):
    autoretry_for = (ConnectionError, TimeoutError)
    retry_backoff = True
    retry_backoff_max = 60
    retry_jitter = True
    max_retries = 3


@celery_app.task(bind=True, base=RetriableTask, name="documents.ingest")
def ingest_document(self, job_id: int) -> None:
    try:
        update_job(job_id, JobStatus.parsing.value, "parsing", 10)
        with SessionLocal() as db:
            job = db.get(IngestionJob, job_id)
            if job is None:
                raise ValueError("Ingestion job not found")
            job.task_id = self.request.id
            job.retry_count = self.request.retries
            version = db.scalar(
                select(DocumentVersion)
                .where(DocumentVersion.id == job.document_version_id)
                .options(
                    selectinload(DocumentVersion.document).selectinload(
                        Document.department
                    )
                )
            )
            if version is None:
                raise ValueError("Document version not found")
            version.status = VersionStatus.processing.value
            db.commit()
            data = ObjectStorage().download(version.object_key)
            pages = parse_document(data, version.mime_type)
            if not pages:
                raise ValueError("No readable text was extracted")
            chunks = build_chunks(version.document, version, pages)
            version_id = version.id

        update_job(job_id, JobStatus.embedding.value, "embedding", 40)
        update_job(job_id, JobStatus.indexing.value, "indexing", 70)
        vector_store = MilvusChunkStore()
        vector_store.delete_version(version.uuid)
        inserted = vector_store.insert_chunks(chunks)
        update_job(job_id, JobStatus.activating.value, "activating", 90)
        with SessionLocal() as db:
            version = db.get(DocumentVersion, version_id)
            if version is None:
                raise ValueError("Document version not found")
            version.metadata_json = {
                **(version.metadata_json or {}),
                "chunking": chunking_metadata(),
            }
            db.flush()
            activate_version(db, version_id, inserted)
        update_job(job_id, JobStatus.ready.value, "ready", 100)
    except Exception as exc:
        with SessionLocal() as db:
            job = db.get(IngestionJob, job_id)
            if job is not None:
                job.status = JobStatus.failed.value
                job.stage = "failed"
                job.error_message = str(exc)[:4000]
                job.retry_count = self.request.retries
                version = db.get(DocumentVersion, job.document_version_id)
                if version is not None:
                    version.status = VersionStatus.failed.value
                    version.error_message = str(exc)[:4000]
                db.commit()
        raise


@celery_app.task(bind=True, base=RetriableTask, name="documents.delete_vectors")
def delete_document_vectors(self, job_id: int) -> None:
    update_job(job_id, JobStatus.deleting.value, "deleting", 20)
    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        if job is None:
            raise ValueError("Deletion job not found")
        version = db.get(DocumentVersion, job.document_version_id)
        if version is None:
            raise ValueError("Document version not found")
        version_uuid = version.uuid
    MilvusChunkStore().delete_version(version_uuid)
    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        version = db.get(DocumentVersion, job.document_version_id)
        version.status = VersionStatus.deleted.value
        job.status = JobStatus.ready.value
        job.stage = "deleted"
        job.progress = 100
        db.commit()
