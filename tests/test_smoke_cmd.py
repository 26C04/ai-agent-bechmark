"""Fake-adapter contracts for the one-document smoke command."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import ocrbench.cli as cli_module
import ocrbench.smoke as smoke_module
from ocrbench.cli import main
from ocrbench.ollama_adapter import ChatResult, GpuPlacement
from ocrbench.prompts import PromptVersion
from ocrbench.runner import WARM_THRESHOLD_NS

_VALID_RESPONSE = json.dumps(
    {
        "customer_name": "Synthetic Customer",
        "order_no": "SYNTHETIC-ORDER-0001",
        "delivery_date": "2026-08-01",
        "items": [{"part_no": "SYNTHETIC-PART", "material": "Synthetic Alloy", "num_pieces": 1}],
    }
)
_PROMPT = PromptVersion("base", "a" * 12, "synthetic prompt")
_FULL_GPU = GpuPlacement(True, "100% GPU")


class _SmokeAdapter:
    """Protocol-compatible fake that exposes every smoke diagnostic input."""

    def __init__(
        self,
        responses: Sequence[ChatResult],
        *,
        placement: GpuPlacement | None = _FULL_GPU,
    ) -> None:
        self._responses = list(responses)
        self._placement = placement
        self.calls: list[dict[str, Any]] = []
        self.metadata_calls = 0

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
        self.calls.append(
            {
                "format_schema": copy.deepcopy(format_schema),
                "image_png": bytes(image_png),
                "model": model,
                "prompt": prompt,
                "seed": seed,
                "temperature": temperature,
            }
        )
        return self._responses[len(self.calls) - 1]

    def resolve_digest(self, model: str) -> str:
        del model
        self.metadata_calls += 1
        return "sha256:synthetic"

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        del model
        self.metadata_calls += 1
        return self._placement

    def server_version(self) -> str:
        self.metadata_calls += 1
        return "synthetic-ollama"


def _chat(
    content: str,
    *,
    prompt_tokens: int | None = 12,
    output_tokens: int | None = 34,
    total_duration_ns: int | None = 1_000_000,
    load_duration_ns: int | None = 1_000_000,
    prompt_eval_duration_ns: int | None = 1_000_000,
    eval_duration_ns: int | None = 1_000_000,
) -> ChatResult:
    return ChatResult(
        content=content,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_ns=total_duration_ns,
        load_duration_ns=load_duration_ns,
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        eval_duration_ns=eval_duration_ns,
    )


def _write_source(path: Path) -> None:
    """Create a tiny image or one-page PDF accepted by the shared preprocessor."""
    Image.new("RGB", (24, 18), (10, 20, 30)).save(
        path, format="PDF" if path.suffix == ".pdf" else "PNG"
    )


def _install_fake(monkeypatch: pytest.MonkeyPatch, adapter: _SmokeAdapter) -> None:
    monkeypatch.setattr(cli_module, "create_adapter", lambda: adapter)
    monkeypatch.setattr(cli_module, "get_active", lambda root, name: _PROMPT)


@pytest.mark.parametrize(("suffix", "expected_kind"), [(".png", "image"), (".pdf", "pdf")])
def test_smoke_success_emits_complete_human_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
    expected_kind: str,
) -> None:
    image_path = tmp_path / f"synthetic{suffix}"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat(_VALID_RESPONSE)])
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path)]) == 0

    output = capsys.readouterr().out.lower()
    for expected in (
        "server",
        "model",
        "digest",
        "gpu",
        "preprocess",
        expected_kind,
        "load",
        "infer",
        "token",
        "parse",
        "needs review",
        "30",
    ):
        assert expected in output
    assert "synthetic-order-0001" in output
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("invalid_response", ["RAW_SECRET_DO_NOT_LEAK", '{"items":[]}'])
def test_smoke_retries_one_identical_input_then_fails_without_raw_output_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invalid_response: str,
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat(invalid_response), _chat(invalid_response)])
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path)]) == 1

    captured = capsys.readouterr()
    assert len(adapter.calls) == 2
    assert adapter.calls[0] == adapter.calls[1]
    assert "RAW_SECRET_DO_NOT_LEAK" not in captured.out
    assert "RAW_SECRET_DO_NOT_LEAK" not in captured.err
    assert "schema_violation" in captured.out or "not_json" in captured.out


def test_smoke_json_parse_failure_returns_exit_one_without_raw_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat("PRIVATE_RAW_RESPONSE"), _chat("PRIVATE_RAW_RESPONSE")])
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path), "--json"]) == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["parse_status"] == "not_json"
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["succeeded"] is False
    assert "PRIVATE_RAW_RESPONSE" not in captured.out
    assert "PRIVATE_RAW_RESPONSE" not in captured.err


@pytest.mark.parametrize("placement", [GpuPlacement(False, "50%/50% CPU/GPU"), None])
def test_smoke_emphasizes_warning_when_gpu_is_not_confirmed_fully_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    placement: GpuPlacement | None,
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat(_VALID_RESPONSE)], placement=placement)
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path)]) == 0
    output = capsys.readouterr().out.lower()
    assert "warning" in output
    assert "gpu" in output
    assert len(adapter.calls) == 1


def test_smoke_model_allowlist_is_checked_before_any_adapter_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat(_VALID_RESPONSE)])
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "not-allowlisted:model", "--image", str(image_path)]) == 2
    assert "allowlisted" in capsys.readouterr().err
    assert adapter.calls == []
    assert adapter.metadata_calls == 0

    assert (
        main(
            [
                "smoke",
                "--model",
                "not-allowlisted:model",
                "--image",
                str(image_path),
                "--allow-any-model",
            ]
        )
        == 0
    )
    assert len(adapter.calls) == 1
    assert adapter.metadata_calls == 3


def test_allow_any_model_on_allowlisted_model_has_no_diagnostic_only_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter([_chat(_VALID_RESPONSE)])
    _install_fake(monkeypatch, adapter)

    assert (
        main(
            [
                "smoke",
                "--model",
                "gemma4:12b",
                "--image",
                str(image_path),
                "--allow-any-model",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_model"] is False
    assert "diagnostic_only_warning" not in payload


def test_smoke_reports_cold_loads_and_enforces_the_warm_30_second_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    cold = smoke_module.run_smoke(
        adapter=_SmokeAdapter([_chat(_VALID_RESPONSE, load_duration_ns=WARM_THRESHOLD_NS)]),
        model_tag="gemma4:12b",
        image_path=image_path,
        prompt=_PROMPT,
    )
    assert cold.warm is False
    assert cold.warm_30_second_target_met is None

    ticks = iter((0, 100, 200, 30_000_000_001))
    monkeypatch.setattr(smoke_module, "perf_counter_ns", lambda: next(ticks))
    warm = smoke_module.run_smoke(
        adapter=_SmokeAdapter([_chat(_VALID_RESPONSE)]),
        model_tag="gemma4:12b",
        image_path=image_path,
        prompt=_PROMPT,
    )
    assert warm.warm is True
    assert warm.within_30_seconds is False
    assert warm.warm_30_second_target_met is False


def test_smoke_excludes_slow_identity_metadata_from_document_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    ticks = iter(
        (
            0,
            40_000_000_000,
            40_000_000_100,
            40_000_000_200,
            40_000_000_300,
            69_999_000_000,
        )
    )

    def clock() -> int:
        return next(ticks)

    monkeypatch.setattr(smoke_module, "perf_counter_ns", clock)

    class _SlowIdentityAdapter(_SmokeAdapter):
        def resolve_digest(self, model: str) -> str:
            clock()
            return super().resolve_digest(model)

        def server_version(self) -> str:
            clock()
            return super().server_version()

    result = smoke_module.run_smoke(
        adapter=_SlowIdentityAdapter([_chat(_VALID_RESPONSE)]),
        model_tag="gemma4:12b",
        image_path=image_path,
        prompt=_PROMPT,
    )

    assert result.timings_ms["total"] == pytest.approx(29_998.9999)
    assert result.within_30_seconds is True
    assert result.warm_30_second_target_met is True


def test_smoke_surfaces_missing_timing_and_token_metadata_as_json_safe_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_source(image_path)
    adapter = _SmokeAdapter(
        [
            _chat(
                _VALID_RESPONSE,
                prompt_tokens=None,
                output_tokens=None,
                total_duration_ns=None,
                load_duration_ns=None,
                prompt_eval_duration_ns=None,
                eval_duration_ns=None,
            )
        ]
    )
    _install_fake(monkeypatch, adapter)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["warm"] is None
    assert json.dumps(payload, allow_nan=False)
