from __future__ import annotations

from typing import Protocol

from .models import KnowledgeDocument


class KnowledgeBaseSource(Protocol):
    @property
    def source_key(self) -> str: ...

    def list_documents(self) -> list[KnowledgeDocument]: ...
