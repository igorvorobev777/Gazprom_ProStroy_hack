from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from rag_orchestrator import (
    LocalLLMClient,
    RagPipeline,
    Settings,
    build_rag_pipeline,
    get_settings,
)

from .embeddings import Embedder, build_embedder
from .index_store import (
    CHUNKING_VERSION,
    SPARSE_LIBRARY_VERSION,
    LocalIndex,
    build_index,
    load_manifest,
)
from .hihub_source import HihubKnowledgeBaseSource
from .knowledge_base import KnowledgeBaseSource
from .retrieval import HybridRetriever
from rag_orchestrator.fast_answer import FAST_ANSWER_SYSTEM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_paths(settings: Settings) -> Settings:
    return settings.model_copy(update={"index_dir": _absolute(settings.index_dir)})


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    embedder: Embedder
    index: LocalIndex
    retriever: HybridRetriever
    pipeline: RagPipeline


def build_knowledge_source(settings: Settings) -> KnowledgeBaseSource:
    return HihubKnowledgeBaseSource(
        base_url=settings.hihub_base_url,
        email=settings.hihub_email,
        password=settings.hihub_password,
        token_name=settings.hihub_token_name,
        section_id=settings.hihub_section_id,
        timeout_seconds=settings.hihub_timeout_seconds,
        per_page=settings.hihub_per_page,
        max_articles=settings.hihub_max_articles,
    )


def build_container(
    settings: Settings | None = None,
    *,
    force_index: bool = False,
) -> ApplicationContainer:
    settings = resolve_paths(settings or get_settings())
    source = build_knowledge_source(settings)
    embedder = build_embedder(settings)

    manifest_path = settings.index_dir / "manifest.json"
    must_build = force_index or not manifest_path.exists()
    if not must_build:
        manifest = load_manifest(settings.index_dir)
        must_build = (
            manifest.source_key != source.source_key
            or manifest.embedding_backend != embedder.backend_name
            or manifest.embedding_dimension != embedder.dimension
            or manifest.chunking_version != CHUNKING_VERSION
            or manifest.chunk_size_chars != settings.chunk_size_chars
            or manifest.chunk_overlap_chars != settings.chunk_overlap_chars
            or manifest.sparse_library != "scikit-learn"
            or manifest.sparse_library_version != SPARSE_LIBRARY_VERSION
        )

    if must_build:
        if not settings.auto_build_index and not force_index:
            raise FileNotFoundError(
                "Индекс не готов. Выполни: python -m scripts.build_index --force"
            )
        build_index(
            source=source,
            settings=settings,
            embedder=embedder,
            force=True,
        )

    index = LocalIndex(settings.index_dir)
    retriever = HybridRetriever(index=index, embedder=embedder, settings=settings)

    llm = LocalLLMClient(settings)
    if settings.llm_warmup_enabled:
        warmed = llm.warmup(
            settings.llm_warmup_timeout_seconds,
            system_prompt=FAST_ANSWER_SYSTEM,
        )
        if not warmed:
            logger.warning(
                "LLM warmup did not complete; the first query may use fallback."
            )

    pipeline = build_rag_pipeline(
        retriever=retriever,
        settings=settings,
        llm=llm,
    )
    return ApplicationContainer(
        settings=settings,
        embedder=embedder,
        index=index,
        retriever=retriever,
        pipeline=pipeline,
    )
