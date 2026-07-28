"""One-document OCR smoke diagnostics without benchmark artifact persistence."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final, cast

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.ollama_adapter import ChatResult, OllamaAdapter
from ocrbench.parsing import ParseResult, ParseStatus, parse_model_output
from ocrbench.preprocess import PreprocessError, preprocess
from ocrbench.prompts import PromptVersion
from ocrbench.runner import WARM_THRESHOLD_NS
from ocrbench.schema import compute_needs_review, ollama_format_schema

SMOKE_TIME_TARGET_MS: Final[float] = 30_000.0
_TEMPERATURE: Final[float] = 0.0


class SmokeError(OcrBenchError):
    """Raised when a smoke diagnostic cannot be completed."""


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """JSON-safe diagnostic state for one image or single-page PDF."""

    model_tag: str
    model_digest: str
    model_allowlisted: bool
    diagnostic_model: bool
    prompt_name: str
    prompt_hash: str
    server_version: str
    gpu_fully_loaded: bool | None
    gpu_placement: str | None
    source_kind: str
    width: int
    height: int
    preprocess_version: str
    first_attempt_status: ParseStatus
    parse_status: ParseStatus
    attempt_count: int
    prediction: dict[str, Any] | None
    needs_review: bool
    warm: bool | None
    within_30_seconds: bool
    warm_30_second_target_met: bool | None
    timings_ms: dict[str, float | None]
    prompt_tokens: int | None
    output_tokens: int | None
    tokens_per_sec: float | None

    @property
    def succeeded(self) -> bool:
        """Return whether the final response passed strict parsing and validation."""
        return self.parse_status is ParseStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible primitives without raw model output."""
        return {
            "attempt_count": self.attempt_count,
            "diagnostic_model": self.diagnostic_model,
            "dimensions": {"height": self.height, "width": self.width},
            "first_attempt_status": self.first_attempt_status.value,
            "gpu_fully_loaded": self.gpu_fully_loaded,
            "gpu_placement": self.gpu_placement,
            "model_allowlisted": self.model_allowlisted,
            "model_digest": self.model_digest,
            "model_tag": self.model_tag,
            "needs_review": self.needs_review,
            "output_tokens": self.output_tokens,
            "parse_status": self.parse_status.value,
            "prediction": copy.deepcopy(self.prediction),
            "preprocess_version": self.preprocess_version,
            "prompt_hash": self.prompt_hash,
            "prompt_name": self.prompt_name,
            "prompt_tokens": self.prompt_tokens,
            "server_version": self.server_version,
            "source_kind": self.source_kind,
            "succeeded": self.succeeded,
            "timings_ms": dict(self.timings_ms),
            "tokens_per_sec": self.tokens_per_sec,
            "warm": self.warm,
            "warm_30_second_target_met": self.warm_30_second_target_met,
            "within_30_seconds": self.within_30_seconds,
        }


def _sum_optional(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(cast(int, value) for value in values)


def _duration_ms(values: list[int | None]) -> float | None:
    total = _sum_optional(values)
    return None if total is None else total / 1_000_000


def _tokens_per_second(results: list[ChatResult]) -> float | None:
    output_tokens = _sum_optional([result.output_tokens for result in results])
    eval_duration_ns = _sum_optional([result.eval_duration_ns for result in results])
    if output_tokens is None or eval_duration_ns is None or eval_duration_ns <= 0:
        return None
    return output_tokens / (eval_duration_ns / 1_000_000_000)


def _call_adapter(
    *,
    adapter: OllamaAdapter,
    model_tag: str,
    prompt: PromptVersion,
    image_png: bytes,
    response_schema: dict[str, Any],
) -> ChatResult:
    try:
        return adapter.generate_structured(
            model=model_tag,
            prompt=prompt.text,
            image_png=image_png,
            format_schema=copy.deepcopy(response_schema),
            seed=config.SEED,
            temperature=_TEMPERATURE,
        )
    except Exception:
        raise SmokeError("Adapter inference failed") from None


def _parse_timed(raw_output: str) -> tuple[ParseResult, int]:
    started = perf_counter_ns()
    parsed = parse_model_output(raw_output)
    return parsed, perf_counter_ns() - started


def _adapter_identity(adapter: OllamaAdapter, model_tag: str) -> tuple[str, str]:
    try:
        digest = adapter.resolve_digest(model_tag)
    except Exception:
        raise SmokeError("Adapter failed to resolve the model digest") from None
    try:
        server_version = adapter.server_version()
    except Exception:
        raise SmokeError("Adapter failed to report its server version") from None
    return digest, server_version


def _gpu_after_inference(
    adapter: OllamaAdapter,
    model_tag: str,
) -> tuple[bool | None, str | None]:
    try:
        placement = adapter.gpu_placement(model_tag)
    except Exception:
        raise SmokeError("Adapter failed to report model GPU placement") from None
    if placement is None:
        return None, None
    return placement.fully_on_gpu, placement.detail


def run_smoke(
    *,
    adapter: OllamaAdapter,
    model_tag: str,
    image_path: Path,
    prompt: PromptVersion,
    allow_any_model: bool = False,
) -> SmokeResult:
    """Run one deterministic diagnostic, retrying only a strict parse failure once."""
    allowlisted = model_tag in config.ALLOWED_MODELS
    if not allowlisted and not allow_any_model:
        raise SmokeError(f"model tag is not allowlisted: {model_tag!r}")
    if not model_tag.strip():
        raise SmokeError("model tag must not be empty")

    digest, server_version = _adapter_identity(adapter, model_tag)
    total_started = perf_counter_ns()
    try:
        prepared = preprocess(Path(image_path))
    except PreprocessError:
        raise SmokeError("Input preprocessing failed") from None

    response_schema = ollama_format_schema()
    chat_results: list[ChatResult] = []
    parse_duration_ns = 0

    first_chat = _call_adapter(
        adapter=adapter,
        model_tag=model_tag,
        prompt=prompt,
        image_png=prepared.png_bytes,
        response_schema=response_schema,
    )
    chat_results.append(first_chat)
    first_parse, elapsed = _parse_timed(first_chat.content)
    parse_duration_ns += elapsed
    final_parse = first_parse

    if first_parse.status in {ParseStatus.NOT_JSON, ParseStatus.SCHEMA_VIOLATION}:
        retry_chat = _call_adapter(
            adapter=adapter,
            model_tag=model_tag,
            prompt=prompt,
            image_png=prepared.png_bytes,
            response_schema=response_schema,
        )
        chat_results.append(retry_chat)
        final_parse, elapsed = _parse_timed(retry_chat.content)
        parse_duration_ns += elapsed

    total_duration_ms = (perf_counter_ns() - total_started) / 1_000_000
    gpu_fully_loaded, gpu_placement = _gpu_after_inference(adapter, model_tag)
    load_values = [result.load_duration_ns for result in chat_results]
    if any(value is None for value in load_values):
        warm: bool | None = None
    elif all(cast(int, value) < WARM_THRESHOLD_NS for value in load_values):
        warm = True
    else:
        warm = False
    within_30_seconds = total_duration_ms <= SMOKE_TIME_TARGET_MS
    target_met = (
        within_30_seconds and final_parse.status is ParseStatus.OK if warm is True else None
    )
    prediction = (
        final_parse.document.model_dump(mode="json") if final_parse.document is not None else None
    )

    return SmokeResult(
        model_tag=model_tag,
        model_digest=digest,
        model_allowlisted=allowlisted,
        diagnostic_model=not allowlisted,
        prompt_name=prompt.name,
        prompt_hash=prompt.hash,
        server_version=server_version,
        gpu_fully_loaded=gpu_fully_loaded,
        gpu_placement=gpu_placement,
        source_kind=prepared.source_kind,
        width=prepared.width,
        height=prepared.height,
        preprocess_version=prepared.version,
        first_attempt_status=first_parse.status,
        parse_status=final_parse.status,
        attempt_count=len(chat_results),
        prediction=prediction,
        needs_review=compute_needs_review(final_parse.document),
        warm=warm,
        within_30_seconds=within_30_seconds,
        warm_30_second_target_met=target_met,
        timings_ms={
            "infer": _duration_ms(
                [
                    None
                    if result.prompt_eval_duration_ns is None or result.eval_duration_ns is None
                    else result.prompt_eval_duration_ns + result.eval_duration_ns
                    for result in chat_results
                ]
            ),
            "load": _duration_ms(load_values),
            "parse_validate": parse_duration_ns / 1_000_000,
            "preprocess": prepared.duration_ms,
            "total": total_duration_ms,
        },
        prompt_tokens=_sum_optional([result.prompt_tokens for result in chat_results]),
        output_tokens=_sum_optional([result.output_tokens for result in chat_results]),
        tokens_per_sec=_tokens_per_second(chat_results),
    )


def _number(value: float | None, *, suffix: str = "") -> str:
    return "unknown" if value is None else f"{value:.3f}{suffix}"


def _yes_no_unknown(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def format_smoke_result(result: SmokeResult) -> str:
    """Render an English diagnostic checklist without exposing raw model output."""
    placement = result.gpu_placement or "model not reported as loaded"
    gpu_line = (
        f"GPU placement: {placement} (fully on GPU: {_yes_no_unknown(result.gpu_fully_loaded)})"
    )
    if result.gpu_fully_loaded is not True:
        gpu_line = f"WARNING: GPU placement is not explicitly confirmed as 100% GPU.\n{gpu_line}"

    if result.warm_30_second_target_met is None:
        target = "NOT APPLICABLE (load state is cold or unknown)"
    else:
        target = "PASS" if result.warm_30_second_target_met else "FAIL"

    prediction = (
        json.dumps(
            result.prediction,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        if result.prediction is not None
        else "unavailable (strict parse/validation failed)"
    )
    timings = result.timings_ms
    prompt_tokens = str(result.prompt_tokens) if result.prompt_tokens is not None else "unknown"
    output_tokens = str(result.output_tokens) if result.output_tokens is not None else "unknown"
    return "\n".join(
        [
            "OCRBench smoke diagnostic",
            f"Server version: {result.server_version}",
            f"Model: {result.model_tag}",
            f"Model digest: {result.model_digest}",
            f"Model allowlisted: {_yes_no_unknown(result.model_allowlisted)}",
            f"Diagnostic-only model: {_yes_no_unknown(result.diagnostic_model)}",
            f"Prompt: {result.prompt_name}/{result.prompt_hash}",
            gpu_line,
            f"Preprocess source: {result.source_kind}",
            f"Preprocess output: {result.width} x {result.height} RGB PNG",
            f"Preprocess version: {result.preprocess_version}",
            f"Preprocess time: {_number(timings['preprocess'], suffix=' ms')}",
            f"Model load time: {_number(timings['load'], suffix=' ms')}",
            f"Infer time: {_number(timings['infer'], suffix=' ms')}",
            f"Parse/validation time: {_number(timings['parse_validate'], suffix=' ms')}",
            f"Total time: {_number(timings['total'], suffix=' ms')}",
            f"Warm inference: {_yes_no_unknown(result.warm)}",
            f"Within 30 seconds: {_yes_no_unknown(result.within_30_seconds)}",
            f"Warm 30-second target: {target}",
            f"Attempts: {result.attempt_count}",
            f"First parse status: {result.first_attempt_status.value}",
            f"Final parse status: {result.parse_status.value}",
            f"Needs review: {_yes_no_unknown(result.needs_review)}",
            f"Prompt tokens: {prompt_tokens}",
            f"Output tokens: {output_tokens}",
            f"Tokens/sec: {_number(result.tokens_per_sec)}",
            "Structured prediction:",
            prediction,
        ]
    )


__all__ = [
    "SMOKE_TIME_TARGET_MS",
    "SmokeError",
    "SmokeResult",
    "format_smoke_result",
    "run_smoke",
]
