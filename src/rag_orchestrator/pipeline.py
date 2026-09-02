from __future__ import annotations

import logging
import re
from collections import OrderedDict
from time import perf_counter
from threading import Lock
from uuid import uuid4

from .config import Settings
from .context_builder import assign_source_ids
from .context_selector import select_naive_context
from .generator import generate_answer
from .grader import grade_context
from .llm_client import (
    LLMInvalidOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    StructuredLLM,
)
from .fast_answer import FAST_ANSWER_SYSTEM, build_fast_answer_prompt, normalize_plain_answer
from .fast_context import extractive_answer, prepare_context
from .naive_responder import run_fast_naive
from .passage_selector import PassageSelector
from .quality_gate import assess_evidence
from .query_planner import (
    build_search_queries,
    clarification_for_underspecified,
    clarification_from_retrieval,
    direct_response,
    external_query_reason,
    is_comparison_query,
    normalize_search_query,
)
from .grounding_guard import assess_answer_grounding
from .answer_relevance import assess_answer_relevance
from .retriever_contract import Retriever
from .schemas import (
    ContextDecision,
    FastNaiveDecision,
    GeneratedAnswer,
    RagResult,
    RagStatus,
    RetrievedChunk,
    RouteType,
)
from .validator import (
    collect_semantic_issues,
    resolve_sources,
    validate_answer_format,
    validate_answer_with_llm,
)


MIN_FINAL_TOP_K = 1
MAX_FINAL_TOP_K = 5

logger = logging.getLogger(__name__)


def _llm_explicitly_abstained(text: str) -> bool:
    normalized = " ".join(text.casefold().replace("ё", "е").split())
    if "недостаточно данных" in normalized:
        return True
    return bool(
        re.search(
            r"(?:в (?:источниках|source)|источники).*?(?:нет|не содержат|недостаточно).*?информац",
            normalized,
        )
        or re.search(r"недостаточно (?:информации|данных).*?(?:ответ|источник)", normalized)
    )


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _latency_answer_profile(query: str, settings: Settings) -> tuple[str, int, int, int, int]:
    """Return profile, token budget, passage chars, context chars and max chunks.

    Short factual questions need less decode and less prompt than procedures.
    Multi-part/action questions keep the full evidence budget so latency tuning does
    not reduce recall or omit conditions.
    """
    q = " ".join(query.casefold().replace("ё", "е").split())
    words = re.findall(r"[0-9a-zа-я]+", q)
    multipart = any(marker in q for marker in (" и " , ";", ", а "))
    procedure = multipart or any(
        marker in q
        for marker in (
            "что делать", "порядок", "порядок действий", "какие действия", "действия при", "этап", "шаг",
            "как оформ", "как провод", "как выполня", "в каких случаях",
            "какие требования", "какие меры", "перечисл", "опиши",
        )
    )
    factual = not procedure and (
        len(words) <= 10
        or any(
            marker in q
            for marker in (
                "срок", "сколько", "кто ", "кем ", "когда",
                "что такое", "что означает", "определение",
                "какой", "какая", "какое",
            )
        )
    )

    if procedure:
        return (
            "procedure",
            min(settings.naive_max_tokens, settings.llm_procedure_max_tokens),
            min(settings.focused_passage_chars, 420),
            min(settings.max_context_chars, 1400),
            min(settings.context_max_chunks, 3),
        )
    if factual:
        return (
            "fact",
            min(settings.naive_max_tokens, settings.llm_fact_max_tokens),
            min(settings.focused_passage_chars, 340),
            min(settings.max_context_chars, 760),
            min(settings.context_max_chunks, 2),
        )
    return (
        "default",
        min(settings.naive_max_tokens, settings.llm_default_max_tokens),
        min(settings.focused_passage_chars, 380),
        min(settings.max_context_chars, 1100),
        min(settings.context_max_chunks, 3),
    )


_GENERIC_QUERY_ROOTS = {
    "оформ", "измен", "согла", "делат", "дейст", "поряд", "требо",
    "прово", "выпол", "нужн", "можно", "должн", "какие", "какой",
}


def _needs_clarification(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Detect short anaphoric questions whose retrieval fans out across topics.

    The API is stateless, so phrases like "кто их согласовывает?" may have no
    antecedent. If the query contains no concrete entity and the top results come
    from several documents, asking one short clarification is safer than picking a
    random procedure.
    """
    q = " ".join(query.casefold().replace("ё", "е").split())
    if not re.search(r"\b(их|это|эти|этого|такое|такие)\b", q):
        return False
    terms = _search_terms(q)
    anchors = {term for term in terms if term not in _GENERIC_QUERY_ROOTS}
    # Concrete acronyms/numbers and hyphenated named entities are good anchors.
    if anchors or re.search(r"\b[А-ЯA-Z]{2,}\b|\d|[а-яa-z]+-[а-яa-z]+", query):
        return False
    top = chunks[:4]
    docs = {chunk.doc_id for chunk in top}
    if len(docs) < 2:
        return False
    if len(top) >= 2 and top[0].score > 0:
        return top[1].score >= top[0].score * 0.72
    return True


def _coherent_context_order(
    query: str,
    chunks: list[RetrievedChunk],
    profile_name: str,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Prefer one internally consistent document for operational procedures.

    The HiHub corpus contains several site-specific fire-safety instructions.
    Mixing them can combine incompatible phone numbers or local responsibilities.
    Retrieval stays broad; only the evidence passed to Qwen is reordered.
    """
    if (
        not settings.quality_procedure_source_coherence
        or profile_name != "procedure"
        or is_comparison_query(query)
        or len(chunks) < 2
    ):
        return chunks

    anchor = chunks[0]
    same_doc = [chunk for chunk in chunks if chunk.doc_id == anchor.doc_id]
    if len(same_doc) < 2:
        return chunks

    anchor_number = anchor.metadata.get("chunk_number")
    def same_doc_key(chunk: RetrievedChunk) -> tuple[int, float]:
        number = chunk.metadata.get("chunk_number")
        distance = 9999
        if isinstance(anchor_number, int) and isinstance(number, int):
            distance = abs(number - anchor_number)
        return distance, -chunk.score

    ordered_same = [anchor] + sorted(
        [chunk for chunk in same_doc if chunk.chunk_id != anchor.chunk_id],
        key=same_doc_key,
    )
    rest = [chunk for chunk in chunks if chunk.doc_id != anchor.doc_id]
    return [*ordered_same, *rest]


def _intent_context_order(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder evidence for intents where a specific structural block matters.

    Retrieval answers "which document?" well, but a high-scoring chunk from that
    document may still describe its general purpose instead of the requested
    composition block. Keep the winning document, then prefer explicit
    ``В состав ... входят`` / role-list chunks for membership questions.
    """
    if len(chunks) < 2:
        return chunks
    q = " ".join(query.casefold().replace("ё", "е").split())
    membership = bool(re.search(r"\bкто\s+(?:входит|состоит)\b", q)) or "состав комис" in q
    if not membership:
        return chunks

    anchor_doc = chunks[0].doc_id
    same_doc = [chunk for chunk in chunks if chunk.doc_id == anchor_doc]
    if len(same_doc) < 2:
        return chunks

    def key(chunk: RetrievedChunk) -> tuple[float, float]:
        text = chunk.text.casefold().replace("ё", "е")
        explicit = 1.0 if ("в состав" in text and re.search(r"\bвход\w*\b", text)) else 0.0
        roles = sum(
            marker in text
            for marker in ("председател", "заместител", "секретар", "старший групп", "член групп")
        )
        return (explicit * 3.0 + min(5, roles) * 0.45 + chunk.score * 0.15, chunk.score)

    ordered = sorted(same_doc, key=key, reverse=True)
    rest = [chunk for chunk in chunks if chunk.doc_id != anchor_doc]
    return [*ordered, *rest]


def _filter_relevant_chunks(
    chunks: list[RetrievedChunk],
    source_ids: list[str],
) -> list[RetrievedChunk]:
    relevant = set(source_ids)
    return [chunk for chunk in chunks if chunk.source_id in relevant]


_QUERY_STOP_WORDS = {
    "а", "без", "бы", "в", "во", "для", "до", "его", "ее", "её",
    "и", "или", "их", "как", "к", "кто", "ли", "на", "над", "не",
    "но", "о", "об", "от", "по", "под", "при", "про", "с", "со",
    "то", "у", "что", "это", "этот", "эта", "эти", "они", "он",
    "она", "оно", "их", "им", "ими", "их",
}


def _search_terms(text: str) -> set[str]:
    """Возвращает устойчивые к русским окончаниям поисковые основы.

    Это не полноценный морфологический анализатор: для быстрой защиты
    corrective-запросов достаточно сравнивать первые 5 символов значимых
    слов. Например, «изменения» и «изменений» дают одну основу «измен».
    """

    tokens = re.findall(r"[0-9a-zа-яё]+", text.lower())
    terms: set[str] = set()
    for token in tokens:
        if token in _QUERY_STOP_WORDS or len(token) < 4:
            continue
        terms.add(token[:5] if len(token) > 5 else token)
    return terms


def _fallback_rewritten_queries(original_query: str) -> list[str]:
    """Строит безопасные уточнения, не меняя предмет вопроса."""

    parts = re.split(
        r"\s+(?:и|или|а также)\s+|[;,]",
        original_query.strip(" \t\r\n?.!"),
        flags=re.IGNORECASE,
    )
    result = [part.strip() for part in parts if _search_terms(part)]
    if len(result) >= 2:
        return result[:2]
    return [original_query.strip()]


def _filter_rewritten_queries(
    original_query: str,
    rewritten_queries: list[str],
) -> list[str]:
    """Отбрасывает LLM-rewrite, ушедшие от темы исходного вопроса."""

    original_terms = _search_terms(original_query)
    if not original_terms:
        return _fallback_rewritten_queries(original_query)

    valid: list[str] = []
    seen: set[str] = set()

    for rewritten in rewritten_queries:
        cleaned = rewritten.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue

        overlap = original_terms & _search_terms(cleaned)
        if overlap:
            valid.append(cleaned)
            seen.add(key)

    return valid[:2] or _fallback_rewritten_queries(original_query)


def _copy_retriever_timings(
    retriever: Retriever,
    *,
    prefix: str,
    timings: dict[str, int],
) -> None:
    raw = getattr(retriever, "last_timings", None)
    if not isinstance(raw, dict):
        return
    for name, value in raw.items():
        if isinstance(name, str) and isinstance(value, int):
            timings[f"{prefix}_{name}"] = value


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: StructuredLLM,
        settings: Settings,
        passage_selector: PassageSelector | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.passage_selector = passage_selector
        self._answer_cache: OrderedDict[tuple[str, int], RagResult] = OrderedDict()
        self._answer_cache_lock = Lock()

    @staticmethod
    def _cache_key(query: str, top_k: int) -> tuple[str, int]:
        normalized = re.sub(r"\s+", " ", query.casefold()).strip(" .!?\t\r\n")
        return normalized, top_k

    def _cache_get(self, query: str, top_k: int, trace_id: str) -> RagResult | None:
        if self.settings.answer_cache_size <= 0:
            return None
        key = self._cache_key(query, top_k)
        with self._answer_cache_lock:
            cached = self._answer_cache.get(key)
            if cached is None:
                return None
            self._answer_cache.move_to_end(key)
            result = cached.model_copy(deep=True)
        result.trace_id = trace_id
        result.latency_ms = 0
        result.timings = {"cache_hit": 1, "total_ms": 0}
        result.diagnostics = [*result.diagnostics, "answer_cache_hit"]
        return result

    def _cache_put(self, query: str, top_k: int, result: RagResult) -> None:
        if self.settings.answer_cache_size <= 0 or result.status != RagStatus.SUCCESS:
            return
        key = self._cache_key(query, top_k)
        with self._answer_cache_lock:
            self._answer_cache[key] = result.model_copy(deep=True)
            self._answer_cache.move_to_end(key)
            while len(self._answer_cache) > self.settings.answer_cache_size:
                self._answer_cache.popitem(last=False)

    def _run_latency_optimized(
        self,
        *,
        query: str,
        top_k: int,
        trace_id: str,
        started_at: float,
        section_id: int = 0,
    ) -> RagResult:
        timings: dict[str, int] = {}
        diagnostics: list[str] = ["latency_optimized_mode"]

        if self.settings.quality_gate_enabled and self.settings.quality_gate_direct_queries_enabled:
            direct = direct_response(query)
            if direct is not None:
                return self._result(
                    trace_id=trace_id,
                    answer=direct.answer,
                    status=RagStatus.SUCCESS,
                    route=RouteType.NAIVE,
                    retrieval_attempts=0,
                    generation_attempts=0,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=[*diagnostics, f"query_gate={direct.reason}"],
                )

        if (
            self.settings.quality_gate_enabled
            and self.settings.quality_gate_external_intents_enabled
        ):
            external_reason = external_query_reason(query)
            if external_reason is not None:
                return self._result(
                    trace_id=trace_id,
                    answer="Этот вопрос не относится к подключённой базе знаний.",
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=RouteType.NAIVE,
                    retrieval_attempts=0,
                    generation_attempts=0,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=[*diagnostics, f"query_gate={external_reason}"],
                )

        if self.settings.quality_gate_enabled and self.settings.quality_gate_clarify_generic_enabled:
            clarification = clarification_for_underspecified(query)
            if clarification is not None:
                return self._result(
                    trace_id=trace_id,
                    answer=clarification,
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=RouteType.NAIVE,
                    retrieval_attempts=0,
                    generation_attempts=0,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=[*diagnostics, "query_gate=underspecified"],
                )

        cached = self._cache_get(query, top_k, trace_id)
        if cached is not None:
            return cached

        retrieval_query = normalize_search_query(query)
        if retrieval_query != query.strip():
            diagnostics.append("query_normalized_for_retrieval")

        search_k = max(
            top_k,
            self.settings.naive_context_top_k,
            self.settings.optimized_retrieval_top_k,
        )
        stage = perf_counter()
        try:
            search_queries = build_search_queries(retrieval_query)
            search_multiple = getattr(self.retriever, "search_multiple", None)
            if len(search_queries) > 1 and callable(search_multiple):
                chunks = search_multiple(search_queries, top_k=search_k, section_id=section_id)
                diagnostics.append(f"retrieval_query_variants={len(search_queries)}")
            else:
                chunks = self.retriever.search(query=search_queries[0], top_k=search_k, section_id=section_id)
        except Exception as exc:
            timings["initial_retrieval_ms"] = _elapsed_ms(stage)
            logger.exception("Latency-mode retrieval failed. trace_id=%s", trace_id)
            return self._result(
                trace_id=trace_id,
                answer="Поиск по базе знаний временно недоступен.",
                status=RagStatus.RETRIEVAL_ERROR,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=0,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=[*diagnostics, type(exc).__name__],
            )
        timings["initial_retrieval_ms"] = _elapsed_ms(stage)
        _copy_retriever_timings(self.retriever, prefix="initial", timings=timings)
        if not chunks:
            return self._result(
                trace_id=trace_id,
                answer="В базе знаний не найдено подходящих материалов.",
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=0,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=diagnostics,
            )

        evidence_level = "strong"
        evidence_score = 1.0
        if self.settings.quality_gate_enabled:
            evidence = assess_evidence(
                retrieval_query,
                chunks,
                weak_threshold=self.settings.quality_gate_weak_threshold,
                strong_threshold=self.settings.quality_gate_strong_threshold,
            )
            evidence_level = evidence.level
            evidence_score = evidence.score
            diagnostics.extend([
                f"evidence_gate={evidence.level}",
                f"evidence_score={evidence.score:.3f}",
                f"evidence_reason={evidence.reason}",
                f"evidence_exact_coverage={evidence.exact_coverage:.3f}",
                f"evidence_stem_coverage={evidence.stem_coverage:.3f}",
                f"evidence_title_coverage={evidence.title_coverage:.3f}",
                f"evidence_title_support={evidence.title_support:.3f}",
                f"evidence_lexical_support={evidence.lexical_support:.3f}",
                f"evidence_document_coherence={evidence.document_coherence:.3f}",
            ])
            if evidence.level == "weak":
                return self._result(
                    trace_id=trace_id,
                    answer=(
                        "В базе знаний не найдено достаточно релевантной информации "
                        "для надёжного ответа на этот вопрос."
                    ),
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=RouteType.NAIVE,
                    retrieval_attempts=1,
                    generation_attempts=0,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=diagnostics,
                )

        retrieval_clarification = clarification_from_retrieval(retrieval_query, chunks)
        if retrieval_clarification is not None:
            return self._result(
                trace_id=trace_id,
                answer=retrieval_clarification,
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=0,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=[*diagnostics, "query_gate=retrieval_ambiguity"],
            )

        if _needs_clarification(query, chunks):
            return self._result(
                trace_id=trace_id,
                answer=(
                    "Уточните, пожалуйста, изменения какого документа, процесса "
                    "или вида работ вы имеете в виду."
                ),
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=0,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=[*diagnostics, "clarification_required_ambiguous_reference"],
            )

        profile_name, desired_tokens, passage_chars, context_chars, context_chunks = (
            _latency_answer_profile(query, self.settings)
        )
        diagnostics.append(f"answer_profile={profile_name}")
        context_candidates = _coherent_context_order(
            query, chunks, profile_name, self.settings
        )
        if context_candidates and chunks and context_candidates[:2] != chunks[:2]:
            diagnostics.append("context_source_coherence=dominant_document")
        intent_ordered = _intent_context_order(query, context_candidates)
        if intent_ordered[:2] != context_candidates[:2]:
            diagnostics.append("context_intent_order=membership")
        context_candidates = intent_ordered
        stage = perf_counter()
        generation_chunks = prepare_context(
            query,
            context_candidates,
            focused=self.settings.focused_passages_enabled,
            passage_chars=passage_chars,
            neighbor_sentences=self.settings.focused_passage_neighbor_sentences,
            max_context_chars=context_chars,
            max_chunks=context_chunks,
            duplicate_jaccard=self.settings.context_duplicate_jaccard,
        )
        generation_chunks = assign_source_ids(generation_chunks)
        timings["context_focus_ms"] = _elapsed_ms(stage)
        timings["context_chars"] = sum(len(c.text) for c in generation_chunks)

        def use_extractive(reason: str, attempts: int) -> RagResult:
            stage_fallback = perf_counter()
            fallback = extractive_answer(
                query,
                generation_chunks,
                max_sentences=1 if profile_name == "fact" else 2,
            )
            timings["extractive_fallback_ms"] = _elapsed_ms(stage_fallback)
            diagnostics.append(reason)
            fallback_allowed = (
                not self.settings.quality_gate_enabled or evidence_level == "strong"
            )
            fallback_relevance = (
                assess_answer_relevance(query, fallback.answer)
                if fallback.answer
                else None
            )
            if fallback_relevance is not None and not fallback_relevance.valid:
                diagnostics.extend(
                    f"fallback_relevance_issue={issue}"
                    for issue in fallback_relevance.issues
                )
            if (
                fallback_allowed
                and fallback.answer
                and fallback.confidence >= self.settings.extractive_min_score
                and fallback_relevance is not None
                and fallback_relevance.valid
            ):
                result = self._result(
                    trace_id=trace_id,
                    answer=fallback.answer,
                    status=RagStatus.SUCCESS,
                    route=RouteType.NAIVE,
                    retrieval_attempts=1,
                    generation_attempts=attempts,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    chunks=generation_chunks,
                    diagnostics=[*diagnostics, f"extractive_confidence={fallback.confidence:.3f}"],
                )
                diagnostics.append("extractive_fallback_not_cached")
                return result
            return self._result(
                trace_id=trace_id,
                answer="В базе знаний недостаточно информации для надёжного ответа.",
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=attempts,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=diagnostics,
            )

        elapsed = (perf_counter() - started_at)
        remaining = self.settings.request_deadline_seconds - elapsed
        if remaining <= 2.0 or not hasattr(self.llm, "complete"):
            return use_extractive("llm_skipped_deadline_or_capability", 0)

        llm_budget = min(
            self.settings.llm_response_budget_seconds,
            max(2.0, remaining - 2.0),
        )

        # CPU-aware prompt fitting. llama.cpp can count the exact tokens after
        # applying the model's chat template without running inference. Keep the
        # full retrieval pool, but compress only the evidence passed to Qwen until
        # prompt+decode fit the measured SLA.
        user_prompt = build_fast_answer_prompt(query, generation_chunks)
        input_tokens: int | None = None
        count_tokens = getattr(self.llm, "count_input_tokens", None)
        recommend_prompt = getattr(self.llm, "recommend_prompt_tokens", None)
        target_prompt_tokens: int | None = None
        if callable(recommend_prompt):
            target_prompt_tokens = int(
                recommend_prompt(
                    desired_tokens=desired_tokens,
                    timeout_seconds=llm_budget,
                )
            )
            timings["llm_prompt_budget_tokens"] = target_prompt_tokens

        if callable(count_tokens):
            fit_stage = perf_counter()
            for _ in range(3):
                input_tokens = count_tokens(
                    system_prompt=FAST_ANSWER_SYSTEM,
                    user_prompt=user_prompt,
                )
                if not input_tokens or not target_prompt_tokens:
                    break
                if input_tokens <= target_prompt_tokens:
                    break

                ratio = max(0.45, min(0.90, target_prompt_tokens / input_tokens))
                context_chars = max(420, int(context_chars * ratio * 0.92))
                passage_chars = max(260, int(passage_chars * ratio * 0.96))
                if ratio < 0.72:
                    context_chunks = max(2, min(context_chunks, 3))
                generation_chunks = prepare_context(
                    query,
                    context_candidates,
                    focused=self.settings.focused_passages_enabled,
                    passage_chars=passage_chars,
                    neighbor_sentences=self.settings.focused_passage_neighbor_sentences,
                    max_context_chars=context_chars,
                    max_chunks=context_chunks,
                    duplicate_jaccard=self.settings.context_duplicate_jaccard,
                )
                generation_chunks = assign_source_ids(generation_chunks)
                user_prompt = build_fast_answer_prompt(query, generation_chunks)
            timings["prompt_fit_ms"] = _elapsed_ms(fit_stage)
            timings["context_chars"] = sum(len(c.text) for c in generation_chunks)
            if input_tokens:
                timings["llm_input_tokens"] = input_tokens
                diagnostics.append(f"llm_input_tokens={input_tokens}")
            if target_prompt_tokens:
                diagnostics.append(f"llm_prompt_budget_tokens={target_prompt_tokens}")

        token_limit = desired_tokens
        recommend = getattr(self.llm, "recommend_max_tokens", None)
        if callable(recommend):
            try:
                token_limit = int(
                    recommend(
                        desired_tokens=desired_tokens,
                        timeout_seconds=llm_budget,
                        input_tokens=input_tokens,
                    )
                )
            except TypeError:
                token_limit = int(
                    recommend(
                        desired_tokens=desired_tokens,
                        timeout_seconds=llm_budget,
                    )
                )
        timings["llm_max_tokens"] = token_limit
        diagnostics.append(f"llm_max_tokens={token_limit}")

        stage = perf_counter()
        llm_truncated = False
        try:
            completion = getattr(self.llm, "complete_with_hard_deadline", None)
            if callable(completion):
                text = completion(
                    system_prompt=FAST_ANSWER_SYSTEM,
                    user_prompt=user_prompt,
                    max_tokens=token_limit,
                    timeout_seconds=llm_budget,
                )
                diagnostics.append("hard_llm_deadline")
            else:
                text = self.llm.complete(
                    system_prompt=FAST_ANSWER_SYSTEM,
                    user_prompt=user_prompt,
                    max_tokens=token_limit,
                    timeout_seconds=llm_budget,
                )
            timings["llm_response_ms"] = _elapsed_ms(stage)
            perf = getattr(self.llm, "performance_snapshot", None)
            if callable(perf):
                snapshot = perf()
                mapping = {
                    "prompt_n": "llm_prompt_tokens",
                    "cache_n": "llm_cached_prompt_tokens",
                    "predicted_n": "llm_predicted_tokens",
                    "prompt_ms": "llm_server_prompt_ms",
                    "predicted_ms": "llm_server_decode_ms",
                }
                for source_key, target_key in mapping.items():
                    value = snapshot.get(source_key)
                    if isinstance(value, (int, float)):
                        timings[target_key] = round(value)
                prompt_tps = snapshot.get("prompt_tps_ema")
                decode_tps = snapshot.get("predicted_tps_ema")
                llm_truncated = bool(snapshot.get("finish_reason_length"))
                if llm_truncated:
                    diagnostics.append("llm_finish_reason=length")
                else:
                    diagnostics.append("llm_finish_reason=stop")
                if isinstance(prompt_tps, (int, float)):
                    diagnostics.append(f"llama_prompt_tps={float(prompt_tps):.1f}")
                if isinstance(decode_tps, (int, float)):
                    diagnostics.append(f"llama_decode_tps={float(decode_tps):.1f}")
        except LLMTimeoutError:
            timings["llm_response_ms"] = _elapsed_ms(stage)
            if self.settings.extractive_fallback_enabled:
                return use_extractive("llm_timeout_extractive_fallback", 1)
            raise
        except (LLMUnavailableError, LLMInvalidOutputError):
            timings["llm_response_ms"] = _elapsed_ms(stage)
            if self.settings.extractive_fallback_enabled:
                return use_extractive("llm_error_extractive_fallback", 1)
            raise

        if _llm_explicitly_abstained(text):
            diagnostics.append("llm_explicit_abstention")
            return self._result(
                trace_id=trace_id,
                answer="В базе знаний недостаточно информации для надёжного ответа.",
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=1,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=diagnostics,
            )

        generated = normalize_plain_answer(
            text,
            generation_chunks,
            query=query,
            truncated=llm_truncated,
            allow_posthoc_citations=(
                not self.settings.quality_gate_enabled or evidence_level == "strong"
            ),
        )
        if not generated.used_source_ids:
            if self.settings.extractive_fallback_enabled:
                return use_extractive("llm_answer_without_sources", 1)
            return self._result(
                trace_id=trace_id,
                answer=generated.answer,
                status=RagStatus.INSUFFICIENT_CONTEXT,
                route=RouteType.NAIVE,
                retrieval_attempts=1,
                generation_attempts=1,
                rewritten_queries=[],
                started_at=started_at,
                timings=timings,
                diagnostics=diagnostics,
            )

        validation = validate_answer_format(generated, generation_chunks)
        hard_issues = [issue for issue in validation.issues if "URL" in issue or "Несуществующие" in issue]
        if hard_issues and self.settings.extractive_fallback_enabled:
            diagnostics.extend(hard_issues)
            return use_extractive("format_guard_extractive_fallback", 1)

        if self.settings.quality_grounding_guard_enabled:
            grounding = assess_answer_grounding(
                generated.answer,
                generation_chunks,
                min_claim_coverage=self.settings.quality_grounding_min_claim_coverage,
            )
            diagnostics.append(f"grounding_checked_claims={grounding.checked_claims}")
            diagnostics.append(f"grounding_min_claim_coverage={grounding.min_claim_coverage:.3f}")
            if not grounding.valid:
                diagnostics.extend(f"grounding_issue={issue}" for issue in grounding.issues)
                if self.settings.extractive_fallback_enabled:
                    return use_extractive("grounding_guard_extractive_fallback", 1)
                return self._result(
                    trace_id=trace_id,
                    answer="В базе знаний недостаточно информации для надёжного ответа.",
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=RouteType.NAIVE,
                    retrieval_attempts=1,
                    generation_attempts=1,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=diagnostics,
                )

        if self.settings.quality_answer_relevance_guard_enabled:
            relevance = assess_answer_relevance(query, generated.answer)
            if not relevance.valid:
                diagnostics.extend(f"relevance_issue={issue}" for issue in relevance.issues)
                if self.settings.extractive_fallback_enabled:
                    return use_extractive("answer_relevance_extractive_fallback", 1)
                return self._result(
                    trace_id=trace_id,
                    answer="В базе знаний недостаточно информации для надёжного ответа.",
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=RouteType.NAIVE,
                    retrieval_attempts=1,
                    generation_attempts=1,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                    diagnostics=diagnostics,
                )

        result = self._result(
            trace_id=trace_id,
            answer=generated.answer,
            status=RagStatus.SUCCESS,
            route=RouteType.NAIVE,
            retrieval_attempts=1,
            generation_attempts=1,
            rewritten_queries=[],
            started_at=started_at,
            timings=timings,
            chunks=generation_chunks,
            diagnostics=diagnostics,
        )
        self._cache_put(query, top_k, result)
        return result

    def _select_fast_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        timing_prefix: str,
        timings: dict[str, int],
    ) -> list[RetrievedChunk]:
        """Готовит компактный top-k для fast-naive вызова."""

        stage = perf_counter()
        selected = select_naive_context(
            chunks,
            top_k=min(self.settings.naive_context_top_k, top_k),
        )
        timings[f"{timing_prefix}_context_selection_ms"] = _elapsed_ms(stage)

        if not selected or not self.settings.naive_passage_selection_enabled:
            return selected

        if self.passage_selector is None:
            raise ValueError(
                "Passage selection включён, но passage_selector не передан."
            )

        stage = perf_counter()
        selected = self.passage_selector.select(
            query=query,
            chunks=selected,
        )
        timings[f"{timing_prefix}_passage_selection_ms"] = _elapsed_ms(stage)

        return selected

    def _result(
        self,
        *,
        trace_id: str,
        answer: str,
        status: RagStatus,
        route: RouteType,
        retrieval_attempts: int,
        generation_attempts: int,
        rewritten_queries: list[str],
        started_at: float,
        timings: dict[str, int],
        chunks: list[RetrievedChunk] | None = None,
        diagnostics: list[str] | None = None,
    ) -> RagResult:
        total = _elapsed_ms(started_at)
        timings["total_ms"] = total
        sources = resolve_sources(answer, chunks or []) if chunks else []
        return RagResult(
            trace_id=trace_id,
            answer=answer,
            sources=sources,
            status=status,
            route_type=route,
            retrieval_attempts=retrieval_attempts,
            generation_attempts=generation_attempts,
            rewritten_queries=rewritten_queries,
            latency_ms=total,
            timings=timings,
            diagnostics=diagnostics or [],
        )

    def run(
        self,
        query: str,
        top_k: int = 5,
        trace_id: str | None = None,
        section_id: int = 0,
    ) -> RagResult:
        query = query.strip()
        if not query:
            raise ValueError("Вопрос не может быть пустым.")

        if (
            type(top_k) is not int
            or not MIN_FINAL_TOP_K <= top_k <= MAX_FINAL_TOP_K
        ):
            raise ValueError("top_k должен быть целым числом от 1 до 5.")

        trace_id = trace_id or str(uuid4())
        started_at = perf_counter()
        if self.settings.latency_optimized_mode:
            return self._run_latency_optimized(
                query=query,
                top_k=top_k,
                trace_id=trace_id,
                started_at=started_at,
                section_id=section_id,
            )
        timings: dict[str, int] = {}
        diagnostics: list[str] = []
        retrieval_attempts = 0
        generation_attempts = 0
        route = RouteType.NAIVE
        rewritten_queries: list[str] = []
        chunks: list[RetrievedChunk] = []
        generation_chunks: list[RetrievedChunk] = []
        generated: GeneratedAnswer | None = None
        generated_by_fast_naive = False
        final_grade = None

        try:
            stage = perf_counter()
            retrieval_attempts += 1
            try:
                chunks = self.retriever.search(query=query, top_k=top_k, section_id=section_id)
            except Exception as exc:
                timings["initial_retrieval_ms"] = _elapsed_ms(stage)
                logger.exception(
                    "Ошибка initial retrieval. trace_id=%s",
                    trace_id,
                )
                return self._result(
                    trace_id=trace_id,
                    answer="Поиск по базе знаний временно недоступен.",
                    status=RagStatus.RETRIEVAL_ERROR,
                    route=route,
                    retrieval_attempts=retrieval_attempts,
                    generation_attempts=generation_attempts,
                    rewritten_queries=rewritten_queries,
                    started_at=started_at,
                    timings=timings,
                    diagnostics=[*diagnostics, type(exc).__name__],
                )
            timings["initial_retrieval_ms"] = _elapsed_ms(stage)
            _copy_retriever_timings(
                self.retriever, prefix="initial", timings=timings
            )

            if not chunks:
                return self._result(
                    trace_id=trace_id,
                    answer="В базе знаний не найдено подходящих материалов.",
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=route,
                    retrieval_attempts=retrieval_attempts,
                    generation_attempts=generation_attempts,
                    rewritten_queries=[],
                    started_at=started_at,
                    timings=timings,
                )

            chunks = assign_source_ids(chunks)
            use_legacy_first_grade = not self.settings.fast_naive_enabled

            
            if self.settings.fast_naive_enabled:
                fast_chunks = self._select_fast_chunks(
                    query=query,
                    chunks=chunks,
                    top_k=top_k,
                    timing_prefix="naive",
                    timings=timings,
                )

                if not fast_chunks:
                    return self._result(
                        trace_id=trace_id,
                        answer="В базе знаний не найдено подходящих материалов.",
                        status=RagStatus.INSUFFICIENT_CONTEXT,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=[],
                        started_at=started_at,
                        timings=timings,
                    )

                try:
                    stage = perf_counter()

                    fast_response = run_fast_naive(
                        query,
                        fast_chunks,
                        self.llm,
                        max_tokens=self.settings.naive_max_tokens,
                    )

                    timings["naive_llm_response_ms"] = _elapsed_ms(stage)

                except (LLMInvalidOutputError, ValueError) as exc:
                    if not self.settings.legacy_naive_fallback_enabled:
                        raise
                    logger.warning(
                        "Fast naive не прошёл структурную проверку; "
                        "использован legacy fallback. trace_id=%s, error=%s",
                        trace_id,
                        exc,
                    )
                    diagnostics.append(
                        "fast_naive_legacy_fallback"
                    )

                    use_legacy_first_grade = True

                else:
                    if fast_response.decision == FastNaiveDecision.ANSWER:
                        generation_attempts += 1

                        generation_chunks = _filter_relevant_chunks(
                            fast_chunks,
                            fast_response.used_source_ids,
                        )

                        if not generation_chunks:
                            raise ValueError(
                                "Fast naive не выбрал существующий источник."
                            )

                        generated = GeneratedAnswer(
                            answer=fast_response.answer,
                            used_source_ids=fast_response.used_source_ids,
                            insufficient_context=False,
                        )

                        generated_by_fast_naive = True

                    else:
                        route = RouteType.CORRECTIVE
                        raw_rewritten_queries = fast_response.rewritten_queries
                        rewritten_queries = _filter_rewritten_queries(
                            query,
                            raw_rewritten_queries,
                        )
                        if rewritten_queries != raw_rewritten_queries:
                            diagnostics.append("rewritten_queries_filtered")
            # Резервный старый naive: отдельная LLM-1, затем LLM-2.
            if generated is None and route == RouteType.NAIVE and use_legacy_first_grade:
                stage = perf_counter()
                first_grade = grade_context(query, chunks, self.llm)
                timings["first_context_grading_ms"] = _elapsed_ms(stage)

                if first_grade.decision == ContextDecision.SUFFICIENT:
                    final_grade = first_grade
                else:
                    route = RouteType.CORRECTIVE
                    raw_rewritten_queries = first_grade.rewritten_queries
                    rewritten_queries = _filter_rewritten_queries(
                        query,
                        raw_rewritten_queries,
                    )
                    if rewritten_queries != raw_rewritten_queries:
                        diagnostics.append("rewritten_queries_filtered")

            if generated is None and route == RouteType.CORRECTIVE:
                if (
                    not rewritten_queries
                    or retrieval_attempts
                    >= self.settings.max_retrieval_attempts
                ):
                    return self._result(
                        trace_id=trace_id,
                        answer=(
                            "В базе знаний недостаточно информации "
                            "для надёжного ответа."
                        ),
                        status=RagStatus.INSUFFICIENT_CONTEXT,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=diagnostics,
                    )

                # Последняя защита перед повторным поиском: rewritten queries
                # не должны менять тему исходного вопроса.
                rewritten_queries = _filter_rewritten_queries(
                    query,
                    rewritten_queries,
                )
                multi_queries = list(dict.fromkeys([query, *rewritten_queries]))

                stage = perf_counter()
                retrieval_attempts += 1
                try:
                    chunks = self.retriever.search_multiple(
                        queries=multi_queries,
                        top_k=top_k,
                        section_id=section_id,
                    )
                except Exception as exc:
                    timings["corrective_multi_retrieval_ms"] = _elapsed_ms(stage)
                    logger.exception(
                        "Ошибка corrective retrieval. trace_id=%s",
                        trace_id,
                    )
                    return self._result(
                        trace_id=trace_id,
                        answer="Повторный поиск временно недоступен.",
                        status=RagStatus.RETRIEVAL_ERROR,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=[*diagnostics, type(exc).__name__],
                    )

                timings["corrective_multi_retrieval_ms"] = _elapsed_ms(
                    stage
                )
                _copy_retriever_timings(
                    self.retriever, prefix="corrective", timings=timings
                )

                if not chunks:
                    return self._result(
                        trace_id=trace_id,
                        answer=(
                            "Повторный поиск не нашёл "
                            "достаточной информации."
                        ),
                        status=RagStatus.INSUFFICIENT_CONTEXT,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=diagnostics,
                    )

                chunks = assign_source_ids(chunks)

                if self.settings.fast_naive_enabled:
                    corrective_chunks = self._select_fast_chunks(
                        query=query,
                        chunks=chunks,
                        top_k=top_k,
                        timing_prefix="corrective",
                        timings=timings,
                    )

                    if not corrective_chunks:
                        return self._result(
                            trace_id=trace_id,
                            answer=(
                                "Повторный поиск не позволил выделить "
                                "релевантные фрагменты."
                            ),
                            status=RagStatus.INSUFFICIENT_CONTEXT,
                            route=route,
                            retrieval_attempts=retrieval_attempts,
                            generation_attempts=generation_attempts,
                            rewritten_queries=rewritten_queries,
                            started_at=started_at,
                            timings=timings,
                            diagnostics=diagnostics,
                        )

                    try:
                        stage = perf_counter()

                        corrective_response = run_fast_naive(
                            query,
                            corrective_chunks,
                            self.llm,
                            max_tokens=self.settings.naive_max_tokens,
                        )

                        timings[
                            "corrective_llm_response_ms"
                        ] = _elapsed_ms(stage)

                    except (LLMInvalidOutputError, ValueError) as exc:
                        if not self.settings.legacy_naive_fallback_enabled:
                            raise

                        logger.warning(
                            "Fast corrective не прошёл структурную проверку; "
                            "использован legacy fallback. trace_id=%s, error=%s",
                            trace_id,
                            exc,
                        )

                        diagnostics.append(
                            "fast_corrective_legacy_fallback"
                        )

                        # Аварийный fallback работает по passages,
                        # а не по исходным полным чанкам.
                        chunks = corrective_chunks

                        stage = perf_counter()

                        second_grade = grade_context(
                            query,
                            chunks,
                            self.llm,
                        )

                        timings[
                            "second_context_grading_ms"
                        ] = _elapsed_ms(stage)

                        if (
                            second_grade.decision
                            != ContextDecision.SUFFICIENT
                        ):
                            return self._result(
                                trace_id=trace_id,
                                answer=(
                                    "В базе знаний недостаточно "
                                    "информации для надёжного ответа."
                                ),
                                status=RagStatus.INSUFFICIENT_CONTEXT,
                                route=route,
                                retrieval_attempts=retrieval_attempts,
                                generation_attempts=generation_attempts,
                                rewritten_queries=rewritten_queries,
                                started_at=started_at,
                                timings=timings,
                                diagnostics=diagnostics,
                            )

                        final_grade = second_grade

                    else:
                        if (
                            corrective_response.decision
                            == FastNaiveDecision.CORRECTIVE
                        ):
                            if corrective_response.reason:
                                diagnostics.append(
                                    corrective_response.reason
                                )

                            return self._result(
                                trace_id=trace_id,
                                answer=(
                                    "В базе знаний недостаточно "
                                    "информации для надёжного ответа."
                                ),
                                status=RagStatus.INSUFFICIENT_CONTEXT,
                                route=route,
                                retrieval_attempts=retrieval_attempts,
                                generation_attempts=generation_attempts,
                                rewritten_queries=rewritten_queries,
                                started_at=started_at,
                                timings=timings,
                                diagnostics=diagnostics,
                            )

                        generation_attempts += 1

                        generation_chunks = _filter_relevant_chunks(
                            corrective_chunks,
                            corrective_response.used_source_ids,
                        )

                        if not generation_chunks:
                            raise ValueError(
                                "Fast corrective не выбрал "
                                "существующий источник."
                            )

                        generated = GeneratedAnswer(
                            answer=corrective_response.answer,
                            used_source_ids=(
                                corrective_response.used_source_ids
                            ),
                            insufficient_context=False,
                        )

                        generated_by_fast_naive = True

                else:
                    # При явно отключённом fast naive остаётся
                    # прежний legacy corrective.
                    stage = perf_counter()

                    second_grade = grade_context(
                        query,
                        chunks,
                        self.llm,
                    )

                    timings[
                        "second_context_grading_ms"
                    ] = _elapsed_ms(stage)

                    if (
                        second_grade.decision
                        != ContextDecision.SUFFICIENT
                    ):
                        return self._result(
                            trace_id=trace_id,
                            answer=(
                                "В базе знаний недостаточно информации "
                                "для надёжного ответа."
                            ),
                            status=RagStatus.INSUFFICIENT_CONTEXT,
                            route=route,
                            retrieval_attempts=retrieval_attempts,
                            generation_attempts=generation_attempts,
                            rewritten_queries=rewritten_queries,
                            started_at=started_at,
                            timings=timings,
                            diagnostics=diagnostics,
                        )

                    final_grade = second_grade

            if generated is None:
                if final_grade is None:
                    raise ValueError("Не определён итоговый набор источников.")

                generation_chunks = _filter_relevant_chunks(
                    chunks,
                    final_grade.relevant_source_ids,
                )
                if not generation_chunks:
                    raise ValueError(
                        "LLM-1 не выбрала ни одного источника для генерации ответа."
                    )

                stage = perf_counter()
                generation_attempts += 1
                generated = generate_answer(
                    query,
                    generation_chunks,
                    self.llm,
                )
                timings["answer_generation_ms"] = _elapsed_ms(stage)
                generated_by_fast_naive = False

            if generated.insufficient_context:
                return self._result(
                    trace_id=trace_id,
                    answer=generated.answer,
                    status=RagStatus.INSUFFICIENT_CONTEXT,
                    route=route,
                    retrieval_attempts=retrieval_attempts,
                    generation_attempts=generation_attempts,
                    rewritten_queries=rewritten_queries,
                    started_at=started_at,
                    timings=timings,
                    diagnostics=diagnostics,
                )

            stage = perf_counter()
            validation = validate_answer_format(generated, generation_chunks)
            timings["naive_format_validation_ms"] = _elapsed_ms(stage)

            semantic_issues: list[str] = []
            run_llm_validation = self.settings.validation_mode == "always" or (
                self.settings.validation_mode == "corrective"
                and route == RouteType.CORRECTIVE
                and not generated_by_fast_naive
            )

            if run_llm_validation:
                stage = perf_counter()
                evaluation = validate_answer_with_llm(
                    query,
                    generation_chunks,
                    generated.answer,
                    self.llm,
                )
                timings["llm_answer_validation_ms"] = _elapsed_ms(stage)
                semantic_issues = collect_semantic_issues(evaluation)

            issues = [*validation.issues, *semantic_issues]
            if issues:
                if generation_attempts >= self.settings.max_generation_attempts:
                    return self._result(
                        trace_id=trace_id,
                        answer="Не удалось сформировать подтверждённый ответ.",
                        status=RagStatus.VALIDATION_ERROR,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=[*diagnostics, *issues],
                    )

                stage = perf_counter()
                generation_attempts += 1
                generated = generate_answer(
                    query,
                    generation_chunks,
                    self.llm,
                    feedback=issues,
                )
                timings["answer_regeneration_ms"] = _elapsed_ms(stage)
                generated_by_fast_naive = False

                if generated.insufficient_context:
                    return self._result(
                        trace_id=trace_id,
                        answer=generated.answer,
                        status=RagStatus.INSUFFICIENT_CONTEXT,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=diagnostics,
                    )
                validation = validate_answer_format(
                    generated,
                    generation_chunks,
                )
                semantic_issues = []

                run_llm_revalidation = (
                    self.settings.validation_mode == "always"
                    or (
                        self.settings.validation_mode == "corrective"
                        and route == RouteType.CORRECTIVE
                    )
                )

                if run_llm_revalidation:
                    stage = perf_counter()

                    evaluation = validate_answer_with_llm(
                        query,
                        generation_chunks,
                        generated.answer,
                        self.llm,
                    )

                    timings[
                        "llm_answer_revalidation_ms"
                    ] = _elapsed_ms(stage)

                    semantic_issues = collect_semantic_issues(
                        evaluation
                    )

                issues = [
                    *validation.issues,
                    *semantic_issues,
                ]

                if issues:
                    return self._result(
                        trace_id=trace_id,
                        answer=(
                            "Не удалось сформировать "
                            "подтверждённый ответ."
                        ),
                        status=RagStatus.VALIDATION_ERROR,
                        route=route,
                        retrieval_attempts=retrieval_attempts,
                        generation_attempts=generation_attempts,
                        rewritten_queries=rewritten_queries,
                        started_at=started_at,
                        timings=timings,
                        diagnostics=[
                            *diagnostics,
                            *issues,
                        ],
                    )

            return self._result(
                trace_id=trace_id,
                answer=generated.answer,
                status=RagStatus.SUCCESS,
                route=route,
                retrieval_attempts=retrieval_attempts,
                generation_attempts=generation_attempts,
                rewritten_queries=rewritten_queries,
                started_at=started_at,
                timings=timings,
                chunks=generation_chunks,
                diagnostics=diagnostics,
            )

        except LLMUnavailableError:
            return self._result(
                trace_id=trace_id,
                answer="Локальная языковая модель недоступна.",
                status=RagStatus.SERVICE_UNAVAILABLE,
                route=route,
                retrieval_attempts=retrieval_attempts,
                generation_attempts=generation_attempts,
                rewritten_queries=rewritten_queries,
                started_at=started_at,
                timings=timings,
                diagnostics=diagnostics,
            )
        except (LLMInvalidOutputError, ValueError) as exc:
            logger.exception(
                "Ошибка обработки RAG-запроса. trace_id=%s",
                trace_id,
            )

            return self._result(
                trace_id=trace_id,
                answer="Не удалось обработать запрос.",
                status=RagStatus.GENERATION_ERROR,
                route=route,
                retrieval_attempts=retrieval_attempts,
                generation_attempts=generation_attempts,
                rewritten_queries=rewritten_queries,
                started_at=started_at,
                timings=timings,
                diagnostics=[
                    *diagnostics,
                    "generation_error",
                ],
            )
        except Exception as exc:
            logger.exception(
                "Непредвиденная ошибка RAG. trace_id=%s",
                trace_id,
            )
            status = (
                RagStatus.RETRIEVAL_ERROR
                if generation_attempts == 0
                else RagStatus.GENERATION_ERROR
            )
            return self._result(
                trace_id=trace_id,
                answer="Сервис временно не смог обработать запрос.",
                status=status,
                route=route,
                retrieval_attempts=retrieval_attempts,
                generation_attempts=generation_attempts,
                rewritten_queries=rewritten_queries,
                started_at=started_at,
                timings=timings,
                diagnostics=[*diagnostics, type(exc).__name__],
            )
