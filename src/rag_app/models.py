from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class KnowledgeDocument(BaseModel):
    id: str
    version: str = "1"
    title: str
    section: str | None = None
    content: str
    url: str
    updated_at: datetime
    is_actual: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "version", "title", "content", "url")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Поле не может быть пустым.")
        return value


class ChunkRecord(BaseModel):
    doc_id: str
    chunk_id: str
    version: str
    title: str
    section: str | None = None
    text: str
    url: str
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexManifest(BaseModel):
    format_version: int = 3
    source_key: str = "unknown"
    chunking_version: str = "legacy-v2"
    chunk_size_chars: int | None = None
    chunk_overlap_chars: int | None = None
    created_at: datetime
    document_count: int
    chunk_count: int
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int
    sparse_library: str = "scikit-learn"
    sparse_library_version: str | None = None
    corpus_hash: str
