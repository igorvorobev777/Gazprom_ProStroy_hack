from __future__ import annotations

import argparse

from rag_app.composition import build_knowledge_source, resolve_paths
from rag_app.embeddings import build_embedder
from rag_app.index_store import build_index
from rag_orchestrator import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить локальный индекс из HiHub")
    parser.add_argument("--force", action="store_true", help="Перестроить индекс")
    args = parser.parse_args()

    settings = resolve_paths(get_settings())
    source = build_knowledge_source(settings)
    embedder = build_embedder(settings)
    manifest = build_index(
        source=source,
        settings=settings,
        embedder=embedder,
        force=args.force,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
