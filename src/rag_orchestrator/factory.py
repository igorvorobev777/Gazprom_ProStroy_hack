from .config import Settings
from .llm_client import LocalLLMClient, StructuredLLM
from .passage_selector import CrossEncoderPassageSelector
from .pipeline import RagPipeline
from .retriever_contract import Retriever


def build_rag_pipeline(
    *,
    retriever: Retriever,
    settings: Settings,
    llm: StructuredLLM | None = None,
) -> RagPipeline:
    passage_selector = None

    if settings.fast_naive_enabled and settings.naive_passage_selection_enabled:
        passage_selector = CrossEncoderPassageSelector(
            model_name=settings.naive_passage_model,
            device=settings.naive_passage_device,
            model_max_length=settings.naive_passage_model_max_length,
            window_chars=settings.naive_passage_window_chars,
            overlap_chars=settings.naive_passage_overlap_chars,
            batch_size=settings.naive_passage_batch_size,
        )

    return RagPipeline(
        retriever=retriever,
        llm=llm or LocalLLMClient(settings),
        settings=settings,
        passage_selector=passage_selector,
    )
