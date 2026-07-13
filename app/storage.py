from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config import get_settings


class ObjectStorage:
    def __init__(self, client: Optional[Minio] = None) -> None:
        settings = get_settings()
        self.bucket = settings.minio_bucket
        self.client = client or Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, object_key: str, data: bytes, content_type: str) -> None:
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            object_key,
            BytesIO(data),
            len(data),
            content_type=content_type,
        )

    def exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except S3Error:
            return False

    def download(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, object_key: str) -> None:
        try:
            self.client.remove_object(self.bucket, object_key)
        except S3Error:
            return

    def ready(self) -> bool:
        self.ensure_bucket()
        return True
