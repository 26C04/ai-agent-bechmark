"""Regression tests for CLI compatibility gates and blind-operation boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.metrics import PairedComparison, ReproSummary, RunSummary
from ocrbench.prompts import PromptVersion
from ocrbench.selection import CandidateEval, LoopState


def _summary() -> RunSummary:
    return RunSummary(
        n_docs=1,
        exact_match_count=1,
        exact_match_rate=1.0,
        exact_match_ci=(0.0, 1.0),
        schema_valid_rate=1.0,
        first_attempt_json_rate=1.0,
        after_retry_success_rate=None,
        header_field_accuracy={"customer_name": 1.0, "order_no": 1.0, "delivery_date": 1.0},
        item_field_accuracy={"part_no": 1.0, "material": 1.0, "num_pieces": 1.0},
        by_source_kind={},
        error_tag_counts={},
        latency_warm_p50_ms=1.0,
        latency_warm_p95_ms=1.0,
        mean_timings_ms={
            "preprocess": 0.0,
            "load": 0.0,
            "infer": 0.0,
            "parse_validate": 0.0,
            "total": 0.0,
        },
        token_totals={"prompt": 1, "output": 1},
        mean_tokens_per_sec=1.0,
        needs_review_count=0,
    )


def _run(
    *,
    split: str = "dev",
    doc_id: str = "doc-0123456789ab",
    model: str = "gemma4:12b",
    prompt_hash: str = "a" * 12,
    adapter_kind: str = "fake",
    run_id: str = "run",
) -> Any:
    manifest = SimpleNamespace(
        split=split,
        dataset_fingerprint="f" * 64,
        preprocess_version="v1",
        ollama_version="fake-1.0",
        seed=20260721,
        temperature=0.0,
        adapter_kind=adapter_kind,
        gpu_fully_loaded=True,
        participation="primary",
        model_tag=model,
        model_digest=f"sha256:{model}",
        prompt_name="base",
        prompt_hash=prompt_hash,
        run_id=run_id,
        started_at_utc="20260724T010203Z",
    )
    return SimpleNamespace(manifest=manifest, documents=[SimpleNamespace(doc_id=doc_id)])


def _load(monkeypatch: pytest.MonkeyPatch, *runs: Any) -> None:
    iterator = iter(runs)
    monkeypatch.setattr(cli_module, "load_run", lambda path: next(iterator))


def _candidate(results: Any) -> CandidateEval:
    manifest = results.manifest
    return CandidateEval(PromptVersion(manifest.prompt_name, manifest.prompt_hash, ""), _summary())


def test_invalid_run_id_is_rejected_before_downstream_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli_module, "resolve_split_dir", lambda split: calls.append("root"))
    monkeypatch.setattr(cli_module, "create_adapter", lambda: calls.append("adapter"))
    assert (
        main(
            [
                "run",
                "--model",
                "gemma4:12b",
                "--split",
                "dev",
                "--prompt-name",
                "base",
                "--run-id",
                "bad/id",
            ]
        )
        == 2
    )
    assert calls == []
    assert "run-id" in capsys.readouterr().err


@pytest.mark.parametrize("right", [_run(doc_id="doc-fedcba987654"), _run(adapter_kind="real")])
def test_compare_rejects_exact_doc_id_and_protocol_mismatches(
    right: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _load(monkeypatch, _run(), right)
    monkeypatch.setattr(cli_module, "paired_comparison", lambda left, right: pytest.fail("metrics"))
    assert main(["compare-runs", "--run-dirs", "left", "right"]) == 2
    assert "traceback" not in capsys.readouterr().err.lower()


def test_compare_contracts_cover_two_three_selection_and_final(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _load(monkeypatch, _run(), _run(model="qwen3.5:9b", prompt_hash="b" * 12))
    monkeypatch.setattr(
        cli_module, "paired_comparison", lambda left, right: PairedComparison(1, 0, 0, 1, 0, 1.0)
    )
    assert main(["compare-runs", "--run-dirs", "left", "right", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    _load(monkeypatch, _run(), _run(), _run(prompt_hash="b" * 12))
    monkeypatch.setattr(cli_module, "reproducibility", lambda values: pytest.fail("metrics"))
    assert main(["compare-runs", "--run-dirs", "one", "two", "three"]) == 2
    capsys.readouterr()

    canary = "SYNTHETIC-SECRET-CANARY"
    _load(
        monkeypatch, _run(split="selection", doc_id=canary), _run(split="selection", doc_id=canary)
    )
    assert main(["compare-runs", "--run-dirs", "one", "two", "--json"]) == 2
    output = capsys.readouterr()
    assert canary not in output.out + output.err

    for flag, allowed, expected in (
        (False, False, 2),
        (True, False, 2),
        (False, True, 2),
        (True, True, 0),
    ):
        if allowed:
            monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
        else:
            monkeypatch.delenv("OCRBENCH_ALLOW_FINAL", raising=False)
        _load(monkeypatch, _run(split="final"), _run(split="final"))
        args = ["compare-runs", "--run-dirs", "one", "two", "--json"]
        if flag:
            args.append("--confirm-final")
        assert main(args) == expected
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == ("ok" if expected == 0 else "error")


def test_incompatible_select_and_loop_do_not_mutate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, candidate = _run(), _run(model="qwen3.5:9b")
    _load(monkeypatch, base, candidate)
    monkeypatch.setattr(cli_module, "apply_adoption", lambda *args: pytest.fail("mutation"))
    assert (
        main(
            [
                "select",
                "adopt",
                "--baseline-run",
                "base",
                "--candidate-run",
                "candidate",
                "--prompt-name",
                "base",
            ]
        )
        == 2
    )
    state = LoopState(
        started_at_utc="2026-07-24T00:00:00Z",
        candidates_tried=0,
        consecutive_no_improve=0,
        consecutive_errors=0,
        best_exact_match_rate=0.0,
        baseline_prompt_hash="a" * 12,
    )
    _load(monkeypatch, base, candidate)
    monkeypatch.setattr(cli_module, "load_loop_state", lambda path: state)
    monkeypatch.setattr(cli_module, "save_loop_state", lambda path, state: pytest.fail("mutation"))
    assert (
        main(
            [
                "loop",
                "record",
                "--state",
                "state",
                "--baseline-run",
                "base",
                "--candidate-run",
                "candidate",
            ]
        )
        == 2
    )


def test_prompt_select_loop_and_three_run_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "prompts"
    monkeypatch.setenv("OCRBENCH_PROMPTS_DIR", str(registry))
    first, second = tmp_path / "one.txt", tmp_path / "two.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    first_hash = hashlib.sha256(b"first").hexdigest()[:12]
    second_hash = hashlib.sha256(b"second").hexdigest()[:12]
    for arguments in (
        ("prompt", "add", "--name", "base", "--file", str(first)),
        ("prompt", "add", "--name", "base", "--file", str(second)),
        ("prompt", "activate", "--name", "base", "--hash", first_hash),
        ("prompt", "activate", "--name", "base", "--hash", second_hash),
        ("prompt", "list", "--name", "base"),
        ("prompt", "show", "--name", "base", "--hash", second_hash),
        ("prompt", "rollback", "--name", "base"),
    ):
        assert main(list(arguments)) == 0
        capsys.readouterr()
    state = tmp_path / "loop.json"
    assert main(["loop", "init", "--state", str(state), "--baseline-prompt-hash", first_hash]) == 0
    assert main(["loop", "check", "--state", str(state)]) == 0
    assert main(["loop", "record", "--state", str(state), "--error"]) == 0
    capsys.readouterr()
    _load(monkeypatch, _run(), _run(), _run())
    monkeypatch.setattr(
        cli_module, "reproducibility", lambda values: ReproSummary(1, 1.0, (1.0, 1.0, 1.0))
    )
    assert main(["compare-runs", "--run-dirs", "one", "two", "three"]) == 0
