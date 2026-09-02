from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified application, retrieval and latency-budget configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        case_sensitive=False,
        extra="ignore",
    )

    # RAG gRPC service
    grpc_host: str = "0.0.0.0"
    grpc_port: int = Field(default=50052, ge=1, le=65535)
    grpc_max_concurrent_queries: int = Field(default=1, ge=1, le=32)
    grpc_stream_tokens: bool = True
    grpc_token_chunk_chars: int = Field(default=48, ge=8, le=512)
    grpc_health_llm_timeout_seconds: float = Field(default=1.5, gt=0.1, le=10.0)
    grpc_max_sync_jobs: int = Field(default=100, ge=10, le=1000)
    auto_build_index: bool = True

    # Knowledge source / index
    hihub_base_url: str = "https://hihub.ru"
    hihub_email: str = ""
    hihub_password: str = ""
    hihub_token_name: str = ""
    hihub_section_id: int = Field(default=0, ge=0)
    hihub_timeout_seconds: float = 30.0
    hihub_per_page: int = Field(default=200, ge=1, le=700)
    hihub_max_articles: int = Field(default=0, ge=0)

    index_dir: Path = Path("data/index")
    chunk_size_chars: int = Field(default=1400, ge=400, le=5000)
    chunk_overlap_chars: int = Field(default=180, ge=0, le=1000)

    # Retrieval
    embedding_backend: Literal["hashing", "sentence_transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"
    embedding_dimension: int = Field(default=768, ge=128, le=4096)

    dense_top_k: int = Field(default=60, ge=1, le=200)
    sparse_top_k: int = Field(default=80, ge=1, le=250)
    rerank_candidate_k: int = Field(default=80, ge=5, le=250)
    reranker_backend: Literal["rrf", "cross_encoder"] = "rrf"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=8, ge=1, le=64)
    reranker_max_length: int = Field(default=384, ge=128, le=1024)
    retrieval_min_score: float = Field(default=0.0, ge=-100.0, le=100.0)
    max_chunks_per_document: int = Field(default=3, ge=1, le=8)
    optimized_retrieval_top_k: int = Field(default=10, ge=5, le=20)

    # Cheap quality boosts for the 978-chunk corpus.
    lexical_rerank_enabled: bool = True
    lexical_candidate_k: int = Field(default=80, ge=5, le=300)
    lexical_coverage_weight: float = Field(default=0.42, ge=0.0, le=1.5)
    lexical_title_weight: float = Field(default=0.16, ge=0.0, le=1.0)
    lexical_proximity_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    lexical_document_weight: float = Field(default=0.10, ge=0.0, le=0.5)
    neighbor_expansion_enabled: bool = True
    neighbor_distance: int = Field(default=1, ge=0, le=2)
    neighbor_anchor_k: int = Field(default=3, ge=1, le=10)

    # Fast lexical passage compression before the LLM.
    focused_passages_enabled: bool = True
    focused_passage_chars: int = Field(default=700, ge=300, le=1800)
    focused_passage_neighbor_sentences: int = Field(default=1, ge=0, le=2)
    max_context_chars: int = Field(default=4200, ge=1500, le=12000)
    context_max_chunks: int = Field(default=5, ge=2, le=8)
    context_duplicate_jaccard: float = Field(default=0.88, ge=0.5, le=1.0)

    # LLM
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3-4b-rag"
    # Qwen3 non-thinking defaults recommended by the model family.
    llm_temperature: float = 0.7
    llm_top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    llm_top_k: int = Field(default=20, ge=0, le=200)
    llm_min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    llm_repeat_penalty: float = Field(default=1.05, ge=0.8, le=1.5)
    # llama.cpp-specific request controls. These are ignored by other providers.
    llm_cache_prompt: bool = True
    llm_cache_reuse: int = Field(default=64, ge=0, le=4096)
    llm_server_predict_budget_ms: int = Field(default=17_500, ge=1000, le=45_000)
    llm_timeout_seconds: float = 20.0
    llm_connect_timeout_seconds: float = 2.0
    llm_max_retries: int = Field(default=0, ge=0, le=2)
    llm_max_tokens: int = Field(default=900, ge=40, le=700)
    naive_max_tokens: int = Field(default=160, ge=60, le=500)
    # Adaptive answer length keeps CPU inference inside the SLA without forcing
    # every factual answer to reserve the full procedure-sized output budget.
    llm_adaptive_tokens: bool = True
    llm_fact_max_tokens: int = Field(default=300, ge=32, le=240)
    llm_default_max_tokens: int = Field(default=550, ge=40, le=320)
    llm_procedure_max_tokens: int = Field(default=600, ge=48, le=400)
    llm_min_answer_tokens: int = Field(default=160, ge=24, le=160)
    llm_decode_margin_seconds: float = Field(default=1.5, ge=0.5, le=5.0)
    llm_perf_ema_alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    # Conservative seed measured on the target CPU. Live llama.cpp timings replace
    # these values via EMA after successful requests.
    llm_initial_prompt_tps: float = Field(default=60.0, gt=1.0, le=5000.0)
    llm_initial_decode_tps: float = Field(default=10.0, gt=1.0, le=500.0)
    llm_input_token_count_enabled: bool = True
    llm_prompt_budget_safety: float = Field(default=0.90, ge=0.5, le=1.0)
    qwen_no_think: bool = True

    # Portable quality gate. These values are evidence-score boundaries, not
    # probabilities. Borderline retrieval may call Qwen but cannot use a blind
    # extractive fallback or auto-created citations.
    quality_gate_enabled: bool = True
    quality_gate_direct_queries_enabled: bool = True
    quality_gate_external_intents_enabled: bool = True
    quality_gate_clarify_generic_enabled: bool = True
    quality_gate_weak_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    quality_gate_strong_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    quality_grounding_guard_enabled: bool = True
    quality_answer_relevance_guard_enabled: bool = True
    quality_grounding_min_claim_coverage: float = Field(default=0.18, ge=0.0, le=1.0)
    quality_procedure_source_coherence: bool = True

    # Hard SLA mode: retrieval + one model call + deterministic fallback.
    latency_optimized_mode: bool = False
    request_deadline_seconds: float = Field(default=23.5, ge=5.0, le=60.0)
    llm_response_budget_seconds: float = Field(default=20.0, ge=3.0, le=45.0)
    extractive_fallback_enabled: bool = True
    extractive_min_score: float = Field(default=0.42, ge=0.0, le=1.0)
    answer_cache_size: int = Field(default=128, ge=0, le=2048)
    llm_warmup_enabled: bool = True
    llm_warmup_timeout_seconds: float = Field(default=8.0, ge=1.0, le=20.0)

    # Legacy pipeline knobs kept for backwards compatibility.
    max_retrieval_attempts: int = 2
    max_generation_attempts: int = 2
    validation_mode: Literal["off", "corrective", "always"] = "off"
    fast_naive_enabled: bool = True
    legacy_naive_fallback_enabled: bool = True
    naive_context_top_k: int = Field(default=5, ge=1, le=5)
    naive_passage_selection_enabled: bool = False
    naive_passage_model: str = "BAAI/bge-reranker-v2-m3"
    naive_passage_device: str = "cpu"
    naive_passage_model_max_length: int = Field(default=512, ge=128, le=1024)
    naive_passage_window_chars: int = Field(default=500, ge=300, le=2000)
    naive_passage_overlap_chars: int = Field(default=150, ge=0, le=1000)
    naive_passage_batch_size: int = Field(default=8, ge=1, le=128)

    @model_validator(mode="after")
    def validate_windows(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.")
        if self.naive_passage_overlap_chars >= self.naive_passage_window_chars:
            raise ValueError(
                "NAIVE_PASSAGE_OVERLAP_CHARS must be smaller than NAIVE_PASSAGE_WINDOW_CHARS."
            )
        if self.rerank_candidate_k < self.naive_context_top_k:
            raise ValueError("RERANK_CANDIDATE_K must be >= NAIVE_CONTEXT_TOP_K.")
        if self.quality_gate_weak_threshold >= self.quality_gate_strong_threshold:
            raise ValueError(
                "QUALITY_GATE_WEAK_THRESHOLD must be smaller than QUALITY_GATE_STRONG_THRESHOLD."
            )
        if self.llm_response_budget_seconds >= self.request_deadline_seconds:
            raise ValueError(
                "LLM_RESPONSE_BUDGET_SECONDS must be smaller than REQUEST_DEADLINE_SECONDS."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
