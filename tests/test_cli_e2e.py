"""Synthetic end-to-end coverage for the public CLI without an Ollama server."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.ollama_adapter import FakeOllamaAdapter


def _write_source_image(path: Path, color: tuple[int, int, int]) -> None:
    """Create a tiny synthetic public-safe source image."""
    Image.new("RGB", (24, 18), color).save(path, format="PNG")


def _write_assignments(path: Path) -> None:
    """Write a two-split synthetic assignment sheet for CLI manifest building."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "source_kind", "split"])
        writer.writeheader()
        writer.writerows(
            [
                {"file": "incoming-dev.png", "source_kind": "scan", "split": "dev"},
                {
                    "file": "incoming-selection.png",
                    "source_kind": "photo",
                    "split": "selection",
                },
            ]
        )


def _write_ground_truth(root: Path) -> str:
    """Create deterministic synthetic ground truth after anonymous files are built."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "customer_name": "Synthetic Customer",
        "order_no": "SYNTHETIC-ORDER-0001",
        "delivery_date": "2026-08-01",
        "items": [{"part_no": "SYNTHETIC-PART", "material": "Synthetic Alloy", "num_pieces": 1}],
    }
    for entry in manifest["docs"]:
        (root / "gt" / f"{entry['doc_id']}.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    return json.dumps(payload, separators=(",", ":"))


def _run_directory(runs_root: Path, run_id: str) -> Path:
    """Find the deterministic run directory while keeping timestamps opaque to tests."""
    matches = list(runs_root.glob(f"*_{run_id}"))
    assert len(matches) == 1
    return matches[0]


def test_synthetic_cli_workflow_is_retryable_blind_and_final_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the required CLI workflow through a fake adapter only."""
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    gt_root = data_root / "gt"
    runs_root = tmp_path / "runs"
    prompts_root = tmp_path / "prompts"
    raw_root.mkdir(parents=True)
    gt_root.mkdir()
    _write_source_image(raw_root / "incoming-dev.png", (10, 20, 30))
    _write_source_image(raw_root / "incoming-selection.png", (40, 50, 60))
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments)

    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(data_root))
    monkeypatch.setenv("OCRBENCH_FINAL_DIR", str(tmp_path / "final-data"))
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs_root))
    monkeypatch.setenv("OCRBENCH_PROMPTS_DIR", str(prompts_root))
    monkeypatch.setenv("OCRBENCH_ADAPTER", "fake")

    assert (
        main(
            [
                "dataset",
                "build-manifest",
                "--root",
                str(data_root),
                "--assignments",
                str(assignments),
            ]
        )
        == 0
    )
    capsys.readouterr()
    synthetic_json = _write_ground_truth(data_root)

    assert main(["dataset", "validate", "--root", str(data_root)]) == 0
    capsys.readouterr()
    assert main(["dataset", "fingerprint", "--root", str(data_root)]) == 0
    assert len(capsys.readouterr().out.strip()) == 64

    prompt_file = tmp_path / "prompt.txt"
    prompt_text = "Extract structured fields and return JSON only."
    prompt_file.write_text(prompt_text, encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    assert main(["prompt", "add", "--name", "base", "--file", str(prompt_file)]) == 0
    capsys.readouterr()
    assert main(["prompt", "activate", "--name", "base", "--hash", prompt_hash]) == 0
    capsys.readouterr()

    adapter = FakeOllamaAdapter(
        ["not JSON", synthetic_json, synthetic_json, synthetic_json, synthetic_json]
    )
    monkeypatch.setattr(cli_module, "create_adapter", lambda: adapter)

    first_run_args = [
        "run",
        "--model",
        "gemma4:12b",
        "--split",
        "dev",
        "--prompt-name",
        "base",
        "--run-id",
        "first",
    ]
    assert main(first_run_args) == 0
    capsys.readouterr()
    first_run = _run_directory(runs_root, "first")
    first_results = json.loads((first_run / "results.json").read_text(encoding="utf-8"))
    assert first_results["documents"][0]["attempt_count"] == 2

    second_run_args = [
        "run",
        "--model",
        "gemma4:12b",
        "--split",
        "dev",
        "--prompt-name",
        "base",
        "--run-id",
        "second",
    ]
    assert main(second_run_args) == 0
    capsys.readouterr()
    second_run = _run_directory(runs_root, "second")

    assert main(["report", "--run-dir", str(first_run)]) == 0
    capsys.readouterr()
    assert (first_run / "report.md").is_file()
    assert (first_run / "analysis.md").is_file()

    exported = tmp_path / "summary.json"
    assert (
        main(["export-summary", "--run-dir", str(first_run), "--out", str(exported), "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    exported_payload = json.loads(exported.read_text(encoding="utf-8"))
    assert exported_payload["manifest"]["split"] == "dev"
    assert "SYNTHETIC-ORDER-0001" not in exported.read_text(encoding="utf-8")

    assert main(["compare-runs", "--run-dirs", str(first_run), str(second_run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    selection_run_args = [
        "run",
        "--model",
        "gemma4:12b",
        "--split",
        "selection",
        "--prompt-name",
        "base",
        "--run-id",
        "selection-first",
    ]
    assert main(selection_run_args) == 0
    selection_output = capsys.readouterr().out
    assert "SYNTHETIC-ORDER-0001" not in selection_output
    selection_run = _run_directory(runs_root, "selection-first")
    selection_run_args[-1] = "selection-second"
    assert main(selection_run_args) == 0
    capsys.readouterr()
    second_selection_run = _run_directory(runs_root, "selection-second")

    assert (
        main(
            [
                "select",
                "pick-best",
                "--candidate-runs",
                str(selection_run),
                str(second_selection_run),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    blocked_export = tmp_path / "selection-summary.json"
    assert (
        main(["export-summary", "--run-dir", str(selection_run), "--out", str(blocked_export)]) == 2
    )
    assert not blocked_export.exists()
    capsys.readouterr()
    assert main(["report", "--run-dir", str(selection_run), "--detail"]) == 2
    capsys.readouterr()

    calls_before_final = len(adapter.calls)
    run_dirs_before_final = sorted(runs_root.iterdir())
    assert (
        main(
            [
                "run",
                "--model",
                "gemma4:12b",
                "--split",
                "final",
                "--prompt-name",
                "base",
            ]
        )
        == 2
    )
    final_error = capsys.readouterr().err
    assert "confirm-final" in final_error
    assert len(adapter.calls) == calls_before_final
    assert sorted(runs_root.iterdir()) == run_dirs_before_final
