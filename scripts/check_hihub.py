from rag_orchestrator import get_settings
from rag_app.composition import build_knowledge_source, resolve_paths


def main() -> None:
    settings = resolve_paths(get_settings())
    probe_settings = settings.model_copy(update={"hihub_max_articles": 3})
    source = build_knowledge_source(probe_settings)
    docs = source.list_documents()
    print(f"OK: HiHub API доступен. Получено тестовых статей: {len(docs)}")
    for doc in docs[:3]:
        print(f"- [{doc.id}] {doc.title}")


if __name__ == "__main__":
    main()
