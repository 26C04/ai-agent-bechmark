"""Adversarial tests for split isolation and blind-output guardrails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ocrbench import config
from ocrbench.dataset import Manifest
from ocrbench.metrics import RunSummary, SourceKindSummary
from ocrbench.ollama_adapter import ChatResult, GpuPlacement
from ocrbench.prompts import PromptVersion
from ocrbench.runner import run_benchmark
from ocrbench.splitguard import (
    DetailReportAccessError,
    FinalAccessError,
    SplitGuardError,
    console_summary_for,
    ensure_detail_report_allowed,
    ensure_final_allowed,
    resolve_split_dir,
)


class _NeverCalledAdapter:
    """Adapter whose methods prove that the Final gate runs before side effects."""

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
        raise AssertionError("inference must not run before the Final gate")

    def resolve_digest(self, model: str) -> str:
        raise AssertionError("metadata must not be read before the Final gate")

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        raise AssertionError("GPU placement must not be read before the Final gate")

    def server_version(self) -> str:
        raise AssertionError("server version must not be read before the Final gate")


def _summary() -> RunSummary:
    return RunSummary(
        n_docs=7,
        exact_match_count=5,
        exact_match_rate=0.731923,
        exact_match_ci=(0.312345, 0.912345),
        schema_valid_rate=0.845671,
        first_attempt_json_rate=0.734562,
        after_retry_success_rate=0.623451,
        header_field_accuracy={"customer_name": 0.534219},
        item_field_accuracy={"part_no": 0.423198},
        by_source_kind={
            "doc-deadbeefcafe": SourceKindSummary(
                n_docs=3,
                exact_match_rate=0.312987,
                schema_valid_rate=0.876123,
            )
        },
        error_tag_counts={"result-count-canary-13": 13},
        latency_warm_p50_ms=1234.567,
        latency_warm_p95_ms=8765.432,
        mean_timings_ms={"total": 3456.789},
        token_totals={"output": 4567},
        mean_tokens_per_sec=23.4567,
        needs_review_count=2,
    )


@pytest.mark.parametrize(
    ("confirm_flag", "environment_value", "allowed"),
    [
        (False, None, False),
        (True, None, False),
        (False, "1", False),
        (True, "1", True),
    ],
)
def test_final_gate_requires_both_exact_confirmations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    confirm_flag: bool,
    environment_value: str | None,
    allowed: bool,
) -> None:
    if environment_value is None:
        monkeypatch.delenv("OCRBENCH_ALLOW_FINAL", raising=False)
    else:
        monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", environment_value)

    if allowed:
        ensure_final_allowed(confirm_final_flag=confirm_flag)
    else:
        with pytest.raises(FinalAccessError, match="NTFS ACLs are the primary"):
            ensure_final_allowed(confirm_final_flag=confirm_flag)


@pytest.mark.parametrize("environment_value", ["", "0", "true", "01", "1 ", " 1"])
def test_final_gate_rejects_ambiguous_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str,
) -> None:
    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", environment_value)

    with pytest.raises(FinalAccessError):
        ensure_final_allowed(confirm_final_flag=True)


def test_final_gate_rejects_truthy_non_boolean_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")

    with pytest.raises(FinalAccessError):
        ensure_final_allowed(confirm_final_flag=1)  # type: ignore[arg-type]


def test_split_directory_resolution_never_uses_data_root_for_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    final_root = tmp_path / "final"
    monkeypatch.setattr(config, "data_dir", lambda: data_root)
    monkeypatch.setattr(config, "final_dir", lambda: final_root)

    assert resolve_split_dir("dev") == data_root
    assert resolve_split_dir("selection") == data_root
    assert resolve_split_dir("final") == final_root

    monkeypatch.setattr(
        config,
        "data_dir",
        lambda: (_ for _ in ()).throw(AssertionError("Final must not inspect data_dir")),
    )
    assert resolve_split_dir("final") == final_root


def test_split_directory_resolution_rejects_unknown_split() -> None:
    with pytest.raises(SplitGuardError, match="Unsupported"):
        resolve_split_dir("development")  # type: ignore[arg-type]


def test_dev_console_summary_contains_aggregate_rates() -> None:
    rendered = console_summary_for("dev", _summary())
    payload = json.loads(rendered)

    assert payload["n_docs"] == 7
    assert payload["exact_match_rate"] == pytest.approx(0.731923)
    assert payload["error_tag_counts"] == {"result-count-canary-13": 13}


@pytest.mark.parametrize("split", ["selection", "final"])
def test_blind_console_summary_contains_only_completion_count(
    split: str,
) -> None:
    rendered = console_summary_for(split, _summary())

    assert rendered == (
        "Completed 7 documents. Run artifacts were written to the requested output directory."
    )
    for forbidden in (
        "0.731923",
        "0.312345",
        "result-count-canary-13",
        "doc-deadbeefcafe",
        "1234.567",
        "4567",
        "needs_review",
    ):
        assert forbidden not in rendered


def test_console_summary_rejects_unknown_split() -> None:
    with pytest.raises(SplitGuardError):
        console_summary_for("selection-preview", _summary())


def test_detail_report_is_allowed_only_for_development() -> None:
    ensure_detail_report_allowed("dev")

    for split in ("selection", "final"):
        with pytest.raises(DetailReportAccessError, match="Development"):
            ensure_detail_report_allowed(split)

    with pytest.raises(SplitGuardError, match="Unsupported"):
        ensure_detail_report_allowed("unknown")


def test_runner_rejects_final_before_adapter_or_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OCRBENCH_ALLOW_FINAL", raising=False)
    runs_root = tmp_path / "must-not-be-created"

    with pytest.raises(FinalAccessError):
        run_benchmark(
            adapter=_NeverCalledAdapter(),
            data_root=tmp_path / "unreadable-final",
            manifest=Manifest(schema_version=1, docs=[]),
            split="final",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=PromptVersion("base", "000000000000", "invalid-before-gate"),
            runs_root=runs_root,
        )

    assert not runs_root.exists()
