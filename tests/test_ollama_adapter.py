"""Unit tests for fake behavior and the network-isolated real adapter wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import ollama
import pytest
from pydantic import ByteSize

from ocrbench import config
from ocrbench.config import ConfigError
from ocrbench.ollama_adapter import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    FakeOllamaAdapter,
    GpuPlacement,
    ModelNotFoundError,
    OllamaAdapter,
    RealOllamaAdapter,
    create_adapter,
)
from ocrbench.schema import OrderDocument, ollama_format_schema


def _generate(adapter: OllamaAdapter, response_schema: dict[str, Any] | None = None) -> str:
    result = adapter.generate_structured(
        model="gemma4:12b",
        prompt="extract this document",
        image_png=b"png-bytes",
        format_schema=response_schema or ollama_format_schema(),
        seed=config.SEED,
    )
    return result.content


def _mock_real_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock(spec=ollama.Client)
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(ollama, "Client", client_factory)
    return client


def test_fake_satisfies_protocol_and_returns_scripted_responses_in_order() -> None:
    adapter: OllamaAdapter = FakeOllamaAdapter(["first", "second"])

    assert _generate(adapter) == "first"
    assert _generate(adapter) == "second"


def test_fake_raises_clear_assertion_when_script_is_exhausted() -> None:
    adapter = FakeOllamaAdapter([])

    with pytest.raises(AssertionError, match="responses are exhausted"):
        _generate(adapter)


def test_fake_always_returns_without_exhaustion() -> None:
    adapter = FakeOllamaAdapter.always("same")

    assert [_generate(adapter) for _ in range(3)] == ["same", "same", "same"]


def test_fake_records_inference_inputs_and_copies_schema() -> None:
    response_schema: dict[str, Any] = {"type": "object", "required": ["items"]}
    adapter = FakeOllamaAdapter(["response"])

    result = adapter.generate_structured(
        model="qwen3.5:9b",
        prompt="prompt",
        image_png=b"12345",
        format_schema=response_schema,
        seed=config.SEED,
        temperature=0.0,
    )
    response_schema["type"] = "changed"

    call = adapter.calls[0]
    assert call.model == "qwen3.5:9b"
    assert call.prompt == "prompt"
    assert call.schema == {"type": "object", "required": ["items"]}
    assert call.seed == config.SEED
    assert call.temperature == 0.0
    assert call.image_size == 5
    assert result.prompt_tokens == len("prompt") // 4
    assert result.output_tokens == len("response") // 4
    assert result.total_duration_ns == result.load_duration_ns == 1_000_000


def test_fake_metadata_is_deterministic() -> None:
    adapter = FakeOllamaAdapter(digest="sha256:configured", fully_on_gpu=False)

    assert adapter.resolve_digest("any-model") == "sha256:configured"
    assert adapter.gpu_placement("any-model") == GpuPlacement(
        fully_on_gpu=False, detail="100%/0% CPU/GPU"
    )
    assert adapter.server_version() == "fake-ollama"


def test_create_adapter_explicit_fake_has_schema_valid_default_response() -> None:
    adapter = create_adapter("fake")

    document = OrderDocument.model_validate_json(_generate(adapter))
    assert document.items == []


def test_create_adapter_uses_environment_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCRBENCH_ADAPTER", " FAKE ")

    assert isinstance(create_adapter(), FakeOllamaAdapter)


def test_create_adapter_rejects_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCRBENCH_ADAPTER", "remote")

    with pytest.raises(ConfigError, match="unknown OCRBENCH_ADAPTER"):
        create_adapter()


def test_real_adapter_fixes_client_host_and_maps_chat_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_real_client(monkeypatch)
    client.chat.return_value = ollama.ChatResponse(
        message=ollama.Message(role="assistant", content="raw output"),
        prompt_eval_count=12,
        eval_count=34,
        total_duration=100,
        load_duration=20,
        prompt_eval_duration=30,
        eval_duration=50,
    )
    adapter = RealOllamaAdapter()
    response_schema = ollama_format_schema()

    result = adapter.generate_structured(
        model="gemma4:12b",
        prompt="extract",
        image_png=b"image",
        format_schema=response_schema,
        seed=config.SEED,
    )

    cast(MagicMock, ollama.Client).assert_called_once_with(host=OLLAMA_HOST)
    client.chat.assert_called_once_with(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "extract", "images": [b"image"]}],
        format=response_schema,
        options={"temperature": 0.0, "seed": config.SEED},
        think=False,
        stream=False,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    assert result.content == "raw output"
    assert result.prompt_tokens == 12
    assert result.output_tokens == 34
    assert result.total_duration_ns == 100
    assert result.load_duration_ns == 20
    assert result.prompt_eval_duration_ns == 30
    assert result.eval_duration_ns == 50


def test_real_adapter_does_not_retry_when_chat_rejects_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_real_client(monkeypatch)
    client.chat.side_effect = RuntimeError("think is unsupported")
    adapter = RealOllamaAdapter()

    with pytest.raises(RuntimeError, match="think is unsupported"):
        _generate(adapter)

    assert client.chat.call_count == 1


def test_real_adapter_maps_missing_chat_fields_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_real_client(monkeypatch)
    client.chat.return_value = ollama.ChatResponse(message=ollama.Message(role="assistant"))

    result = RealOllamaAdapter().generate_structured(
        model="gemma4:12b",
        prompt="extract",
        image_png=b"image",
        format_schema={},
        seed=config.SEED,
    )

    assert result.content == ""
    assert result.prompt_tokens is None
    assert result.output_tokens is None
    assert result.total_duration_ns is None


def test_real_adapter_resolves_only_an_exact_model_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_real_client(monkeypatch)
    client.list.return_value = ollama.ListResponse(
        models=[
            ollama.ListResponse.Model(model="gemma4:latest", digest="sha256:wrong"),
            ollama.ListResponse.Model(model="gemma4:12b", digest="sha256:right"),
        ]
    )

    assert RealOllamaAdapter().resolve_digest("gemma4:12b") == "sha256:right"


@pytest.mark.parametrize(
    "models",
    [
        [],
        [ollama.ListResponse.Model(model="gemma4:latest", digest="sha256:other")],
        [ollama.ListResponse.Model(model="gemma4:12b", digest=None)],
    ],
)
def test_real_adapter_rejects_missing_model_or_digest(
    monkeypatch: pytest.MonkeyPatch,
    models: list[ollama.ListResponse.Model],
) -> None:
    client = _mock_real_client(monkeypatch)
    client.list.return_value = ollama.ListResponse(models=models)

    with pytest.raises(ModelNotFoundError, match="gemma4:12b"):
        RealOllamaAdapter().resolve_digest("gemma4:12b")


@pytest.mark.parametrize(
    ("size", "size_vram", "expected"),
    [
        (100, 100, GpuPlacement(True, "100% GPU")),
        (100, 120, GpuPlacement(True, "100% GPU")),
        (100, 45, GpuPlacement(False, "55%/45% CPU/GPU")),
        (None, 45, GpuPlacement(False, "unknown GPU placement")),
        (100, None, GpuPlacement(False, "unknown GPU placement")),
        (0, 0, GpuPlacement(False, "unknown GPU placement")),
    ],
)
def test_real_adapter_interprets_gpu_placement_safely(
    monkeypatch: pytest.MonkeyPatch,
    size: int | None,
    size_vram: int | None,
    expected: GpuPlacement,
) -> None:
    client = _mock_real_client(monkeypatch)
    client.ps.return_value = ollama.ProcessResponse(
        models=[
            ollama.ProcessResponse.Model(
                model="gemma4:12b",
                size=ByteSize(size) if size is not None else None,
                size_vram=ByteSize(size_vram) if size_vram is not None else None,
            )
        ]
    )

    assert RealOllamaAdapter().gpu_placement("gemma4:12b") == expected


def test_real_adapter_uses_name_when_process_model_field_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_real_client(monkeypatch)
    client.ps.return_value = ollama.ProcessResponse(
        models=[
            ollama.ProcessResponse.Model(
                name="gemma4:12b", size=ByteSize(100), size_vram=ByteSize(100)
            )
        ]
    )

    assert RealOllamaAdapter().gpu_placement("gemma4:12b") == GpuPlacement(True, "100% GPU")


def test_real_adapter_returns_none_when_model_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_real_client(monkeypatch)
    client.ps.return_value = ollama.ProcessResponse(models=[])

    assert RealOllamaAdapter().gpu_placement("gemma4:12b") is None


def test_real_adapter_reads_version_endpoint_and_handles_missing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_real_client(monkeypatch)
    client._request.side_effect = [SimpleNamespace(version="0.12.0"), SimpleNamespace(version=None)]
    adapter = RealOllamaAdapter()

    assert adapter.server_version() == "0.12.0"
    assert adapter.server_version() == ""
    assert client._request.call_count == 2
    assert client._request.call_args_list[0].args[1:] == ("GET", "/api/version")


def test_create_adapter_explicit_real_uses_fixed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_real_client(monkeypatch)

    assert isinstance(create_adapter("real"), RealOllamaAdapter)
    cast(MagicMock, ollama.Client).assert_called_once_with(host=OLLAMA_HOST)
