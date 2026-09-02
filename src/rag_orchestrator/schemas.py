from __future__ import annotations
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Финальный retrieval передаёт pipeline не более пяти чанков.
SourceId = Literal["S1", "S2", "S3", "S4", "S5"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievedChunk(BaseModel):
    """Контракт между retrieval-модулем и LLM-частью."""

    doc_id: str
    chunk_id: str
    title: str
    text: str
    url: str
    section: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None


class ContextDecision(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ContextGrade(StrictModel):
    """Результат legacy LLM-1 для corrective и резервного naive-пути."""

    decision: ContextDecision
    relevant_source_ids: list[SourceId]
    missing_information: list[str]
    rewritten_queries: list[str]
    reason: str

    @field_validator("rewritten_queries")
    @classmethod
    def normalize_queries(cls, value: list[str]) -> list[str]:
        clean: list[str] = []
        for item in value:
            item = item.strip()
            if item and item not in clean:
                clean.append(item)
        return clean[:3]


SOURCE_MARKER_PATTERN = re.compile(
    r"\[\s*[SsСс]\s*\d+\s*\]"
)


class GeneratedAnswer(StrictModel):
    """Результат legacy LLM-2."""

    answer: str
    used_source_ids: list[SourceId]
    insufficient_context: bool

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        return value.strip()

    @field_validator("used_source_ids")
    @classmethod
    def unique_used_source_ids(
        cls,
        value: list[SourceId],
    ) -> list[SourceId]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_answer_contract(self) -> "GeneratedAnswer":
        if not self.answer:
            raise ValueError(
                "Поле answer не может быть пустым."
            )

        if self.insufficient_context:
            if self.used_source_ids:
                raise ValueError(
                    "При insufficient_context=true "
                    "used_source_ids должен быть пустым."
                )

            if SOURCE_MARKER_PATTERN.search(self.answer):
                raise ValueError(
                    "При insufficient_context=true "
                    "answer не должен содержать S-метки."
                )

        elif not self.used_source_ids:
            raise ValueError(
                "При insufficient_context=false должен быть "
                "указан хотя бы один источник."
            )

        return self


class FastNaiveDecision(str, Enum):
    ANSWER = "answer"
    CORRECTIVE = "corrective"


class FastNaiveResponse(StrictModel):
    """Единый ответ быстрого naive-маршрута.

    За один вызов модель либо отвечает, либо отправляет запрос в corrective.
    """

    decision: FastNaiveDecision
    answer: str
    used_source_ids: list[SourceId]
    rewritten_queries: list[str]
    reason: str

    @field_validator("answer", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("used_source_ids")
    @classmethod
    def unique_source_ids(cls, value: list[SourceId]) -> list[SourceId]:
        return list(dict.fromkeys(value))

    @field_validator("rewritten_queries")
    @classmethod
    def normalize_fast_queries(cls, value: list[str]) -> list[str]:
        clean: list[str] = []
        for item in value:
            item = item.strip()
            if item and item not in clean:
                clean.append(item)
        return clean[:2]

    @model_validator(mode="after")
    def validate_decision_contract(self) -> "FastNaiveResponse":
        if self.decision == FastNaiveDecision.ANSWER:
            if not self.answer:
                raise ValueError("Для decision=answer поле answer не может быть пустым.")
            if not self.used_source_ids:
                raise ValueError(
                    "Для decision=answer должен быть указан хотя бы один источник."
                )
            if self.rewritten_queries:
                raise ValueError(
                    "Для decision=answer rewritten_queries должен быть пустым."
                )

        elif self.decision == FastNaiveDecision.CORRECTIVE:
            if self.answer or self.used_source_ids:
                raise ValueError(
                    "Для decision=corrective ответ и "
                    "used_source_ids должны быть пустыми."
                )

            if not self.rewritten_queries:
                raise ValueError(
                    "Для decision=corrective нужен хотя бы "
                    "один rewritten_query."
                )

        return self


class AnswerEvaluation(StrictModel):
    """Смысловая оценка сформированного ответа."""

    valid: bool
    answers_question: bool
    grounded_in_context: bool
    unsupported_claims: list[str]
    missing_citations: list[str]
    reason: str


class BasicValidation(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    source_id: str
    doc_id: str
    chunk_id: str
    title: str
    section: str | None = None
    url: str
    score: float


class RagStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    GENERATION_ERROR = "generation_error"
    VALIDATION_ERROR = "validation_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RETRIEVAL_ERROR = "retrieval_error"


class RouteType(str, Enum):
    NAIVE = "naive"
    CORRECTIVE = "corrective"


class RagResult(BaseModel):
    """Структурированный результат работы RAG-pipeline."""

    trace_id: str
    answer: str
    sources: list[SourceReference]
    status: RagStatus
    route_type: RouteType
    retrieval_attempts: int
    generation_attempts: int
    rewritten_queries: list[str] = Field(default_factory=list)
    latency_ms: int
    timings: dict[str, int] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)
