from __future__ import annotations

import json
import statistics
import sys
from time import perf_counter

import httpx

from rag_orchestrator.config import get_settings
from rag_orchestrator.fast_answer import FAST_ANSWER_SYSTEM


SOURCES = """[S1] Test regulation
The permit is valid for no more than 14 calendar days. If work conditions change, a new permit must be issued.

[S2] Test procedure
The responsible manager approves the permit before work starts. The employee must stop work if unsafe conditions arise."""

QUERIES = [
    "Какой срок действия наряда-допуска?",
    "Что делать при изменении условий работ?",
    "Кто согласовывает наряд-допуск?",
]


def endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def health_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/health"


def main() -> None:
    settings = get_settings()
    url = endpoint(settings.llm_base_url)
    health_url = health_endpoint(settings.llm_base_url)

    try:
        with httpx.Client(timeout=2.0) as health_client:
            health = health_client.get(health_url)
            health.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        print("ERROR: llama-server is not reachable on 127.0.0.1:1234.")
        print("Start it in another PowerShell window and LEAVE THAT WINDOW OPEN:")
        print(r"  .\run_llama_cpu_stable.ps1")
        print("Then run this profiler again from a PowerShell where (.venv) is active.")
        print(f"Details: {exc}")
        sys.exit(2)

    rows = []
    with httpx.Client(timeout=25.0) as client:
        for query in QUERIES:
            user = (
                f"<QUESTION>\n{query}\n</QUESTION>\n\n"
                f"<SOURCES>\n{SOURCES}\n</SOURCES>\n\n"
                "Ответь кратко с меткой источника. /no_think"
            )
            payload = {
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": FAST_ANSWER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "max_tokens": 80,
                "stream": False,
                "top_k": settings.llm_top_k,
                "min_p": settings.llm_min_p,
                "cache_prompt": True,
                "n_cache_reuse": settings.llm_cache_reuse,
                "t_max_predict_ms": 18000,
            }
            started = perf_counter()
            response = client.post(url, json=payload)
            elapsed = (perf_counter() - started) * 1000
            response.raise_for_status()
            data = response.json()
            timings = data.get("timings") or {}
            row = {
                "query": query,
                "wall_ms": round(elapsed),
                "prompt_tokens": timings.get("prompt_n"),
                "cached_prompt_tokens": timings.get("cache_n"),
                "prompt_tps": timings.get("prompt_per_second"),
                "predicted_tokens": timings.get("predicted_n"),
                "decode_tps": timings.get("predicted_per_second"),
                "answer": ((data.get("choices") or [{}])[0].get("message") or {}).get("content", ""),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))

    prompt_tps = [float(r["prompt_tps"]) for r in rows if isinstance(r.get("prompt_tps"), (int, float))]
    decode_tps = [float(r["decode_tps"]) for r in rows if isinstance(r.get("decode_tps"), (int, float))]
    wall = [float(r["wall_ms"]) for r in rows]
    summary = {
        "samples": len(rows),
        "median_wall_ms": round(statistics.median(wall)),
        "median_prompt_tps": round(statistics.median(prompt_tps), 2) if prompt_tps else None,
        "median_decode_tps": round(statistics.median(decode_tps), 2) if decode_tps else None,
    }
    if decode_tps:
        tps = statistics.median(decode_tps)
        summary["estimated_decode_seconds_96_tokens"] = round(96 / tps, 2)
        summary["estimated_decode_seconds_160_tokens"] = round(160 / tps, 2)
        summary["cpu_sla_guidance"] = (
            "strong" if tps >= 10 else "borderline" if tps >= 6 else "short-answer profile recommended"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
