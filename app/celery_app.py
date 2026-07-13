from celery import Celery

from app.config import get_settings

settings = get_settings()
celery_app = Celery(
    "enterprise_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    timezone="UTC",
    enable_utc=True,
)
