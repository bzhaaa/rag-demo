import datetime
import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, Message, MessageRole, User
from app.rag.application.citations import assemble_citations, cited_chunks
from app.rag.contracts.models import PipelineInput
from app.rag.orchestration.pipeline import RAGPipeline
from app.repositories import active_versions_for_user
from app.services import write_audit


class RAGApplicationService:
    def __init__(
        self,
        pipeline: RAGPipeline,
        settings: Settings,
        web_search_provider_name: str,
    ) -> None:
        self.pipeline = pipeline
        self.settings = settings
        self.web_search_provider_name = web_search_provider_name

    def answer(
        self,
        db: Session,
        user: User,
        question: str,
        conversation_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        total_started = time.perf_counter()
        versions = list(active_versions_for_user(db, user))
        version_uuids = [version.uuid for version in versions]
        run_id = uuid.uuid4()
        trace_id = str(run_id)
        result = self.pipeline.invoke(
            PipelineInput(
                question=question,
                version_uuids=version_uuids,
                authorized_version_count=len(version_uuids),
            ),
            config={
                "run_name": "enterprise-crag",
                "run_id": run_id,
                "tags": ["enterprise-rag", "crag", "authorized-retrieval"],
                "metadata": {
                    "trace_id": trace_id,
                    "user_uuid": user.uuid,
                    "authorized_version_count": len(version_uuids),
                },
            },
        )
        chunks = cited_chunks(
            result.get("evidence", result.get("relevant", [])),
            result.get("cited_indices", []),
        )
        citations = assemble_citations(db, chunks)
        conversation = self._conversation(
            db,
            user,
            conversation_uuid,
            question,
        )
        result["timings"]["total"] = time.perf_counter() - total_started
        db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.user.value,
                content=question,
            )
        )
        db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.assistant.value,
                content=result["answer"],
                citations=citations,
                model_name=self.settings.llm_model,
                trace_id=trace_id,
                metrics={
                    "timings": result.get("timings", {}),
                    "rag_diagnostics": result.get("diagnostics", {}),
                },
            )
        )
        write_audit(
            db,
            "query.execute",
            "conversation",
            user,
            conversation.uuid,
            {
                "trace_id": trace_id,
                "refused": result.get("refused", False),
                "refusal_detail": result.get("refusal_detail"),
                "citation_count": len(citations),
                "evidence_route": result.get("evidence_route"),
                "web_search_attempted": result.get(
                    "web_search_attempted", False
                ),
                "knowledge_evidence_count": len(result.get("relevant", [])),
                "web_evidence_count": len(result.get("web_relevant", [])),
                "web_search_provider": self.web_search_provider_name,
                "web_search_failed": result.get("web_search_failed", False),
            },
        )
        conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
        db.commit()
        return {
            "conversation_uuid": conversation.uuid,
            "answer": result["answer"],
            "citations": citations,
            "refused": result.get("refused", False),
            "refusal_reason": result.get("refusal_reason"),
            "refusal_detail": result.get("refusal_detail"),
            "trace_id": trace_id,
            "timings": result["timings"],
        }

    @staticmethod
    def _conversation(
        db: Session,
        user: User,
        conversation_uuid: Optional[str],
        question: str,
    ) -> Conversation:
        conversation = None
        if conversation_uuid:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.uuid == conversation_uuid,
                    Conversation.user_id == user.id,
                )
            )
        if conversation is None:
            conversation = Conversation(
                user_id=user.id,
                title=question.strip()[:120],
            )
            db.add(conversation)
            db.flush()
        return conversation
