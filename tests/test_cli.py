"""Black-box contracts for the public ``ocrbench`` command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench import __version__
from ocrbench.cli import main
from ocrbench.prompts import PromptVersion


def test_version_option_returns_success_and_prints_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{__version__}\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["dataset", "--help"],
        ["dataset", "build-manifest", "--help"],
        ["dataset", "validate", "--help"],
        ["dataset", "fingerprint", "--help"],
        ["prompt", "--help"],
        ["prompt", "add", "--help"],
        ["prompt", "list", "--help"],
        ["prompt", "show", "--help"],
        ["prompt", "activate", "--help"],
        ["prompt", "rollback", "--help"],
        ["run", "--help"],
        ["report", "--help"],
        ["export-summary", "--help"],
        ["select", "--help"],
        ["select", "adopt", "--help"],
        ["select", "pick-best", "--help"],
        ["loop", "--help"],
        ["loop", "init", "--help"],
        ["loop", "check", "--help"],
        ["loop", "record", "--help"],
        ["compare-runs", "--help"],
    ],
)
def test_documented_commands_and_leaves_have_help(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Every T17 command is discoverable without executing a benchmark action."""
    assert main(arguments) == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()
    assert captured.err == ""


def test_unknown_command_is_a_clean_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["definitely-not-a-command"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "usage:" not in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_missing_required_argument_is_a_clean_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "traceback" not in captured.err.lower()


def test_unknown_model_is_an_expected_error_without_external_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(tmp_path / "runs"))

    assert (
        main(
            [
                "run",
                "--model",
                "unapproved:model",
                "--split",
                "dev",
                "--prompt-name",
                "base",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "unapproved:model" in captured.err
    assert not (tmp_path / "runs").exists()


def test_missing_run_directory_is_an_expected_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", "--run-dir", str(tmp_path / "missing-run")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "run directory" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_json_flag_emits_parseable_success_for_agent_facing_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state.json"

    assert (
        main(
            [
                "loop",
                "init",
                "--state",
                str(state),
                "--baseline-prompt-hash",
                "a" * 12,
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["baseline_prompt_hash"] == "a" * 12
    assert state.is_file()


def test_json_flag_emits_one_parseable_error_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "export-summary",
                "--run-dir",
                str(tmp_path / "missing-run"),
                "--out",
                str(tmp_path / "summary.json"),
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert captured.err == ""
    assert "traceback" not in json.dumps(payload).lower()


def test_adapter_failure_returns_runtime_exit_one_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = PromptVersion("base", "a" * 12, "synthetic prompt")
    monkeypatch.setattr(cli_module, "resolve_split_dir", lambda split: tmp_path)
    monkeypatch.setattr(cli_module, "load_manifest", lambda root: object())
    monkeypatch.setattr(cli_module, "get_active", lambda root, name: prompt)

    def fail_adapter() -> object:
        raise RuntimeError("synthetic adapter failure")

    monkeypatch.setattr(cli_module, "create_adapter", fail_adapter)

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
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "runtime error: synthetic adapter failure\n"
