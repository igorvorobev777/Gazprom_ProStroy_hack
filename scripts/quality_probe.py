from __future__ import annotations

import argparse
import json

from rag_app.composition import resolve_paths
from rag_app.embeddings import build_embedder
from rag_app.index_store import LocalIndex
from rag_app.retrieval import HybridRetriever
from rag_orchestrator.config import get_settings
from rag_orchestrator.quality_gate import assess_evidence
from rag_orchestrator.query_planner import (
    build_search_queries,
    clarification_for_underspecified,
    clarification_from_retrieval,
    direct_response,
    external_query_reason,
    normalize_search_query,
)


def inspect_query(query: str, retriever: HybridRetriever, settings) -> dict:
    direct = direct_response(query) if settings.quality_gate_direct_queries_enabled else None
    if direct is not None:
        return {"query": query, "route": "direct", "reason": direct.reason}

    external = (
        external_query_reason(query)
        if settings.quality_gate_external_intents_enabled
        else None
    )
    if external is not None:
        return {"query": query, "route": "reject", "reason": external}

    clarification = (
        clarification_for_underspecified(query)
        if settings.quality_gate_clarify_generic_enabled
        else None
    )
    if clarification is not None:
        return {"query": query, "route": "clarify", "reason": "underspecified"}

    retrieval_query = normalize_search_query(query)
    search_queries = build_search_queries(retrieval_query)
    chunks = retriever.search(search_queries[0], top_k=max(5, settings.optimized_retrieval_top_k))
    evidence = assess_evidence(
        retrieval_query,
        chunks,
        weak_threshold=settings.quality_gate_weak_threshold,
        strong_threshold=settings.quality_gate_strong_threshold,
    )
    if evidence.level == "weak":
        route = "reject"
    elif clarification_from_retrieval(retrieval_query, chunks) is not None:
        route = "clarify"
    else:
        route = "answer"

    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "route": route,
        "evidence_level": evidence.level,
        "evidence_score": round(evidence.score, 3),
        "evidence_reason": evidence.reason,
        "exact_coverage": round(evidence.exact_coverage, 3),
        "stem_coverage": round(evidence.stem_coverage, 3),
        "title_coverage": round(evidence.title_coverage, 3),
        "title_support": round(evidence.title_support, 3),
        "lexical_support": round(evidence.lexical_support, 3),
        "top_document": chunks[0].title if chunks else None,
        "top_score": round(chunks[0].score, 3) if chunks else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect quality routing without calling the LLM.")
    parser.add_argument("queries", nargs="+", help="Questions to inspect")
    args = parser.parse_args()

    settings = resolve_paths(get_settings())
    index = LocalIndex(settings.index_dir)
    retriever = HybridRetriever(index=index, embedder=build_embedder(settings), settings=settings)
    for query in args.queries:
        print(json.dumps(inspect_query(query, retriever, settings), ensure_ascii=False))


if __name__ == "__main__":
    main()
