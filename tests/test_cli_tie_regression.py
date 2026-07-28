"""Focused regressions for CLI run-ID validation and blind-selection stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from tests.test_cli_regressions import _candidate, _load, _run


@pytest.mark.parametrize("run_id", ["bad/path", "_leading", "a" * 65])
def test_invalid_run_id_taxonomy_rejects_before_downstream_work(
    run_id: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
                run_id,
            ]
        )
        == 2
    )
    assert calls == []
    assert "run-id" in capsys.readouterr().err


def test_exact_pick_best_ties_use_canonical_stable_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    qwen_path, gemma_path = tmp_path / "qwen", tmp_path / "gemma"
    qwen_path.mkdir()
    gemma_path.mkdir()
    qwen = _run(split="selection", model="qwen3.5:9b", run_id="later")
    gemma = _run(split="selection", model="gemma4:12b", run_id="earlier")
    monkeypatch.setattr(cli_module, "_candidate_from_results", _candidate)

    _load(monkeypatch, qwen, gemma)
    assert (
        main(["select", "pick-best", "--candidate-runs", str(qwen_path), str(gemma_path), "--json"])
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    _load(monkeypatch, gemma, qwen)
    assert (
        main(["select", "pick-best", "--candidate-runs", str(gemma_path), str(qwen_path), "--json"])
        == 0
    )
    second = json.loads(capsys.readouterr().out)

    expected = {
        "model_tag": "gemma4:12b",
        "model_digest": "sha256:gemma4:12b",
        "run_dir": str(gemma_path.resolve()),
        "prompt_name": "base",
        "prompt_hash": "a" * 12,
    }
    for payload in (first, second):
        assert {key: payload[key] for key in expected} == expected
