from .context_builder import build_context
from .llm_client import StructuredLLM
from .prompts import CONTEXT_GRADER_SYSTEM, context_grader_prompt
from .schemas import ContextDecision, ContextGrade, RetrievedChunk


def grade_context(
    query: str,
    chunks: list[RetrievedChunk],
    llm: StructuredLLM,
) -> ContextGrade:
    available_source_ids = [
        chunk.source_id for chunk in chunks if chunk.source_id
    ]
    result = llm.structured(
        response_model=ContextGrade,
        system_prompt=CONTEXT_GRADER_SYSTEM,
        user_prompt=context_grader_prompt(
            query,
            build_context(chunks),
            available_source_ids,
        ),
        max_tokens=260,
    )

    allowed = set(available_source_ids)
    unknown = set(result.relevant_source_ids) - allowed
    if unknown:
        raise ValueError(
            f"LLM-1 указала несуществующие источники: {sorted(unknown)}"
        )

    if result.decision == ContextDecision.SUFFICIENT:
        if not result.relevant_source_ids:
            raise ValueError(
                "LLM-1 признала контекст достаточным, но не указала источники."
            )
        result.rewritten_queries = []
        result.missing_information = []

    return result
