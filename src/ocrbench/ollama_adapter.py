"""Type-safe Ollama adapter and deterministic fake for benchmark execution."""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import ollama
from pydantic import BaseModel

from ocrbench.config import ConfigError, OcrBenchError
from ocrbench.schema import OrderDocument

OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"
OLLAMA_KEEP_ALIVE: Final[str] = "10m"

_FAKE_SERVER_VERSION: Final[str] = "fake-ollama"
_FAKE_DURATION_NS: Final[int] = 1_000_000
_DEFAULT_FAKE_JSON: Final[str] = OrderDocument(
    customer_name=None,
    order_no=None,
    delivery_date=None,
    items=[],
).model_dump_json()


class ModelNotFoundError(OcrBenchError):
    """Raised when an exact Ollama model tag cannot be resolved."""


@dataclass(frozen=True)
class ChatResult:
    """Structured metadata returned by one non-streaming Ollama chat call."""

    content: str
    prompt_tokens: int | None
    output_tokens: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None


@dataclass(frozen=True)
class GpuPlacement:
    """Observed model placement reported by Ollama's process endpoint."""

    fully_on_gpu: bool
    detail: str


class OllamaAdapter(Protocol):
    """Interface used by the benchmark runner for model inference and metadata."""

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        image_png: bytes,
        format_schema: dict[str, Any],
        seed: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Generate one structured response without retries."""

    def resolve_digest(self, model: str) -> str:
        """Resolve an exact model tag to its immutable digest."""

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        """Return current GPU placement, or ``None`` if the model is not loaded."""

    def server_version(self) -> str:
        """Return the local Ollama server version."""


class _VersionResponse(BaseModel):
    """Typed response for the version endpoint absent from the locked 0.6.x client API."""

    version: str | None = None


class RealOllamaAdapter:
    """Thin adapter over the official client, permanently bound to localhost."""

    def __init__(self) -> None:
        self._client = ollama.Client(host=OLLAMA_HOST)

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        image_png: bytes,
        format_schema: dict[str, Any],
        seed: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Issue exactly one chat call and preserve the model's raw text output."""
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_png]}],
            format=format_schema,
            options={"temperature": temperature, "seed": seed},
            think=False,
            stream=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        return ChatResult(
            content=response.message.content or "",
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            total_duration_ns=response.total_duration,
            load_duration_ns=response.load_duration,
            prompt_eval_duration_ns=response.prompt_eval_duration,
            eval_duration_ns=response.eval_duration,
        )

    def resolve_digest(self, model: str) -> str:
        """Resolve a model by exact tag, rejecting absent or incomplete entries."""
        for listed_model in self._client.list().models:
            if listed_model.model == model and listed_model.digest is not None:
                return listed_model.digest
        raise ModelNotFoundError(f"Ollama model not found by exact tag: {model}")

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        """Interpret the model's VRAM share from Ollama's process listing."""
        for process_model in self._client.ps().models:
            process_name = (
                process_model.model if process_model.model is not None else process_model.name
            )
            if process_name != model:
                continue

            size = int(process_model.size) if process_model.size is not None else None
            size_vram = (
                int(process_model.size_vram) if process_model.size_vram is not None else None
            )
            if size is None or size_vram is None or size <= 0:
                return GpuPlacement(fully_on_gpu=False, detail="unknown GPU placement")
            if size_vram >= size:
                return GpuPlacement(fully_on_gpu=True, detail="100% GPU")

            gpu_percent = max(0, min(100, round(size_vram * 100 / size)))
            return GpuPlacement(
                fully_on_gpu=False,
                detail=f"{100 - gpu_percent}%/{gpu_percent}% CPU/GPU",
            )
        return None

    def server_version(self) -> str:
        """Read ``/api/version`` through the installed client's typed request path."""
        response = self._client._request(_VersionResponse, "GET", "/api/version")
        return response.version or ""


@dataclass(frozen=True)
class FakeCall:
    """Immutable record of one fake structured-generation call."""

    model: str
    prompt: str
    schema: dict[str, Any]
    seed: int
    temperature: float
    image_size: int


class FakeOllamaAdapter:
    """Return scripted responses in order while satisfying ``OllamaAdapter``."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        digest: str = "sha256:fake",
        fully_on_gpu: bool = True,
    ) -> None:
        self._responses = list(responses) if responses is not None else []
        self._next_response = 0
        self._always_response: str | None = None
        self._digest = digest
        self._fully_on_gpu = fully_on_gpu
        self.calls: list[FakeCall] = []

    @classmethod
    def always(cls, json_text: str) -> FakeOllamaAdapter:
        """Build a fake that returns the same response for every call."""
        adapter = cls()
        adapter._always_response = json_text
        return adapter

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        image_png: bytes,
        format_schema: dict[str, Any],
        seed: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Record one call and return the next deterministic fake result."""
        self.calls.append(
            FakeCall(
                model=model,
                prompt=prompt,
                schema=copy.deepcopy(format_schema),
                seed=seed,
                temperature=temperature,
                image_size=len(image_png),
            )
        )
        if self._always_response is not None:
            content = self._always_response
        else:
            if self._next_response >= len(self._responses):
                raise AssertionError("FakeOllamaAdapter scripted responses are exhausted")
            content = self._responses[self._next_response]
            self._next_response += 1

        return ChatResult(
            content=content,
            prompt_tokens=len(prompt) // 4,
            output_tokens=len(content) // 4,
            total_duration_ns=_FAKE_DURATION_NS,
            load_duration_ns=_FAKE_DURATION_NS,
            prompt_eval_duration_ns=_FAKE_DURATION_NS,
            eval_duration_ns=_FAKE_DURATION_NS,
        )

    def resolve_digest(self, model: str) -> str:
        """Return the configured deterministic fake digest."""
        del model
        return self._digest

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        """Return deterministic fake GPU placement for any model."""
        del model
        detail = "100% GPU" if self._fully_on_gpu else "100%/0% CPU/GPU"
        return GpuPlacement(fully_on_gpu=self._fully_on_gpu, detail=detail)

    def server_version(self) -> str:
        """Return a stable fake server version."""
        return _FAKE_SERVER_VERSION


def create_adapter(kind: str | None = None) -> OllamaAdapter:
    """Create an adapter from ``kind`` or ``OCRBENCH_ADAPTER`` (default: real)."""
    configured_kind = kind if kind is not None else os.environ.get("OCRBENCH_ADAPTER", "real")
    selected_kind = configured_kind.strip().lower()
    if selected_kind == "real":
        return RealOllamaAdapter()
    if selected_kind == "fake":
        return FakeOllamaAdapter.always(_DEFAULT_FAKE_JSON)
    raise ConfigError(f"unknown OCRBENCH_ADAPTER value: {configured_kind!r}")
