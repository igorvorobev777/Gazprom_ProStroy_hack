from __future__ import annotations

import json
import re
import threading
from queue import Queue
from typing import Protocol, TypeVar

import httpx

try:
    from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
except ModuleNotFoundError:
    APIConnectionError = APITimeoutError = BadRequestError = Exception
    OpenAI = None

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def complete_with_hard_deadline(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float,
    ) -> str: ...

    def structured(
        self,
        *,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> T: ...

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str: ...


class LLMUnavailableError(RuntimeError):
    pass


class LLMTimeoutError(LLMUnavailableError):
    pass


class LLMInvalidOutputError(RuntimeError):
    pass


class LocalLLMClient:
    """OpenAI-compatible client optimized for a local llama-server/Qwen model.

    The client combines four latency controls:
    * zero hidden SDK retries;
    * llama.cpp's own prediction-time budget (t_max_predict_ms);
    * an outer wall-clock deadline independent of HTTP read semantics;
    * rolling server timings, used to adapt output length to observed decode speed.

    It also asks llama.cpp to reuse prompt/KV cache where possible and records
    prompt/decode throughput returned by llama-server for diagnostics.
    """

    def __init__(self, settings: Settings) -> None:
        if OpenAI is None:
            raise RuntimeError("Package openai is not installed.")
        self.settings = settings
        timeout = httpx.Timeout(
            settings.llm_timeout_seconds,
            connect=settings.llm_connect_timeout_seconds,
        )
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=timeout,
            max_retries=settings.llm_max_retries,
        )
        # One local model should not accumulate a queue of stale timed-out work.
        self._generation_slot = threading.Lock()
        self._perf_lock = threading.Lock()
        # Seed with conservative machine-specific values so the *first* real RAG
        # request is SLA-aware. Successful llama.cpp responses continuously refine
        # these estimates via EMA.
        self._predicted_tps_ema: float | None = settings.llm_initial_decode_tps
        self._prompt_tps_ema: float | None = settings.llm_initial_prompt_tps
        self._last_server_timings: dict[str, float | int] = {}
        self._token_count_url = (
            f"{str(settings.llm_base_url).rstrip('/')}/chat/completions/input_tokens"
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
        text = re.sub(r"^\s*</think>\s*", "", text, flags=re.I).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _response_extra(response: object) -> dict:
        value = getattr(response, "model_extra", None)
        if isinstance(value, dict):
            return value
        value = getattr(response, "__pydantic_extra__", None)
        if isinstance(value, dict):
            return value
        return {}

    def _record_server_timings(self, response: object) -> None:
        raw = self._response_extra(response).get("timings")
        if not isinstance(raw, dict):
            return

        cleaned: dict[str, float | int] = {}
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                cleaned[key] = value
        # finish_reason is useful to distinguish a complete concise answer from
        # one that hit max_tokens. Store it as numeric flags so existing timing
        # diagnostics stay backwards compatible.
        choices = getattr(response, "choices", None)
        if choices:
            reason = getattr(choices[0], "finish_reason", None)
            if reason == "length":
                cleaned["finish_reason_length"] = 1
            elif reason:
                cleaned["finish_reason_stop"] = 1

        if not cleaned:
            return

        alpha = self.settings.llm_perf_ema_alpha
        prompt_tps = cleaned.get("prompt_per_second")
        predicted_tps = cleaned.get("predicted_per_second")
        with self._perf_lock:
            self._last_server_timings = cleaned
            if isinstance(prompt_tps, (int, float)) and prompt_tps > 0:
                value = float(prompt_tps)
                self._prompt_tps_ema = (
                    value
                    if self._prompt_tps_ema is None
                    else alpha * value + (1.0 - alpha) * self._prompt_tps_ema
                )
            if isinstance(predicted_tps, (int, float)) and predicted_tps > 0:
                value = float(predicted_tps)
                self._predicted_tps_ema = (
                    value
                    if self._predicted_tps_ema is None
                    else alpha * value + (1.0 - alpha) * self._predicted_tps_ema
                )

    def performance_snapshot(self) -> dict[str, float | int]:
        with self._perf_lock:
            result = dict(self._last_server_timings)
            if self._prompt_tps_ema is not None:
                result["prompt_tps_ema"] = round(self._prompt_tps_ema, 3)
            if self._predicted_tps_ema is not None:
                result["predicted_tps_ema"] = round(self._predicted_tps_ema, 3)
            return result

    def count_input_tokens(self, *, system_prompt: str, user_prompt: str) -> int | None:
        """Ask llama.cpp to count the exact chat-template input tokens.

        The endpoint performs tokenization only; it does not run model inference.
        Failure is non-fatal because older OpenAI-compatible servers may not expose
        llama.cpp's token-count extension.
        """
        if not self.settings.llm_input_token_count_enabled:
            return None
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        try:
            with httpx.Client(
                timeout=httpx.Timeout(2.0, connect=1.0),
                headers=headers,
            ) as client:
                response = client.post(self._token_count_url, json=payload)
                response.raise_for_status()
                value = response.json().get("input_tokens")
                if isinstance(value, int) and value > 0:
                    return value
        except (httpx.HTTPError, ValueError, TypeError):
            return None
        return None

    def recommend_prompt_tokens(
        self,
        *,
        desired_tokens: int,
        timeout_seconds: float,
    ) -> int:
        """Maximum input tokens that still leave decode room inside the SLA."""
        with self._perf_lock:
            prompt_tps = self._prompt_tps_ema or self.settings.llm_initial_prompt_tps
            decode_tps = self._predicted_tps_ema or self.settings.llm_initial_decode_tps
        decode_reserve = desired_tokens / max(1.0, float(decode_tps))
        available_prompt_seconds = max(
            2.0,
            float(timeout_seconds)
            - decode_reserve
            - self.settings.llm_decode_margin_seconds,
        )
        return max(
            220,
            int(
                prompt_tps
                * available_prompt_seconds
                * self.settings.llm_prompt_budget_safety
            ),
        )

    def recommend_max_tokens(
        self,
        *,
        desired_tokens: int,
        timeout_seconds: float,
        input_tokens: int | None = None,
    ) -> int:
        """Fit *prompt + decode* into the model wall-clock budget."""
        if not self.settings.llm_adaptive_tokens:
            return desired_tokens
        with self._perf_lock:
            prompt_tps = self._prompt_tps_ema or self.settings.llm_initial_prompt_tps
            decode_tps = self._predicted_tps_ema or self.settings.llm_initial_decode_tps

        prompt_seconds = 0.0
        if input_tokens and input_tokens > 0:
            prompt_seconds = input_tokens / max(1.0, float(prompt_tps))
        decode_seconds = max(
            1.0,
            float(timeout_seconds)
            - prompt_seconds
            - self.settings.llm_decode_margin_seconds,
        )
        capacity = int(
            decode_tps
            * decode_seconds
            * self.settings.llm_prompt_budget_safety
        )
        floor = min(self.settings.llm_min_answer_tokens, desired_tokens)
        return max(floor, min(desired_tokens, capacity))

    def _chat_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        response_format: dict | None,
        timeout_seconds: float | None = None,
    ):
        extra_body: dict[str, object] = {
            "top_k": self.settings.llm_top_k,
            "min_p": self.settings.llm_min_p,
            "repeat_penalty": self.settings.llm_repeat_penalty,
            "presence_penalty": self.settings.llm_presence_penalty,
            "cache_prompt": self.settings.llm_cache_prompt,
            "n_cache_reuse": self.settings.llm_cache_reuse,
        }

        # llama.cpp supports a server-side prediction budget in addition to the
        # client timeout. It stops backend work instead of merely abandoning the
        # HTTP request. Keep it slightly inside the outer request deadline.
        if timeout_seconds is not None:
            server_budget = min(
                self.settings.llm_server_predict_budget_ms,
                max(1000, int((timeout_seconds - 0.75) * 1000)),
            )
            extra_body["t_max_predict_ms"] = server_budget

        kwargs = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": extra_body,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        client = self.client
        if timeout_seconds is not None:
            client = client.with_options(
                timeout=httpx.Timeout(
                    timeout_seconds,
                    connect=min(self.settings.llm_connect_timeout_seconds, timeout_seconds),
                ),
                max_retries=0,
            )
        response = client.chat.completions.create(**kwargs)
        self._record_server_timings(response)
        return response

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        if self.settings.qwen_no_think and "/no_think" not in user_prompt[-50:]:
            user_prompt = f"{user_prompt.rstrip()}\n\n/no_think"
        try:
            response = self._chat_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens or self.settings.naive_max_tokens,
                response_format=None,
                timeout_seconds=timeout_seconds,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError("llama-server exceeded the response time budget.") from exc
        except APIConnectionError as exc:
            raise LLMUnavailableError("llama-server is unavailable.") from exc
        content = self._clean_output(response.choices[0].message.content or "")
        if not content:
            raise LLMInvalidOutputError("The model returned an empty response.")
        return content

    def complete_with_hard_deadline(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float,
    ) -> str:
        if not self._generation_slot.acquire(blocking=False):
            raise LLMUnavailableError(
                "llama-server is still finishing a previous timed-out generation."
            )

        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                value = self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                result_queue.put(("ok", value))
            except BaseException as exc:
                result_queue.put(("error", exc))
            finally:
                self._generation_slot.release()

        thread = threading.Thread(
            target=worker,
            name="rag-llm-deadline",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=max(0.1, float(timeout_seconds)))
        if thread.is_alive():
            raise LLMTimeoutError(
                f"llama-server exceeded hard wall-clock budget ({timeout_seconds:.1f}s)."
            )

        status, payload = result_queue.get_nowait()
        if status == "ok":
            return str(payload)
        if isinstance(payload, BaseException):
            raise payload
        raise LLMInvalidOutputError("llama-server worker returned no result.")

    def structured(
        self,
        *,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> T:
        schema = response_model.model_json_schema()
        schema_instruction = (
            "\n\nReturn only JSON matching this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        token_limit = max_tokens or self.settings.llm_max_tokens
        json_schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": schema,
            },
        }
        try:
            try:
                response = self._chat_request(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=token_limit,
                    response_format=json_schema_format,
                )
            except BadRequestError:
                try:
                    response = self._chat_request(
                        system_prompt=system_prompt + schema_instruction,
                        user_prompt=user_prompt,
                        max_tokens=token_limit,
                        response_format={"type": "json_object"},
                    )
                except BadRequestError:
                    response = self._chat_request(
                        system_prompt=system_prompt + schema_instruction,
                        user_prompt=user_prompt,
                        max_tokens=token_limit,
                        response_format=None,
                    )
        except APITimeoutError as exc:
            raise LLMTimeoutError("llama-server timed out.") from exc
        except APIConnectionError as exc:
            raise LLMUnavailableError("llama-server is unavailable.") from exc

        content = self._clean_output(response.choices[0].message.content or "")
        if not content:
            raise LLMInvalidOutputError("The model returned an empty response.")
        try:
            return response_model.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise LLMInvalidOutputError("The model returned invalid structured output.") from exc

    def warmup(
        self,
        timeout_seconds: float | None = None,
        *,
        system_prompt: str | None = None,
    ) -> bool:
        """Prime runtime and the stable RAG system-prompt prefix in the KV cache."""
        try:
            self.complete(
                system_prompt=system_prompt or "/no_think\nReply with OK only.",
                user_prompt=(
                    "<QUESTION>\nwarmup\n</QUESTION>\n\n"
                    "<SOURCES>\n[S1] warmup\nOK\n</SOURCES>\n\n"
                    "Reply with OK only. /no_think"
                ),
                max_tokens=2,
                timeout_seconds=timeout_seconds or self.settings.llm_warmup_timeout_seconds,
            )
            return True
        except (LLMUnavailableError, LLMInvalidOutputError):
            return False
