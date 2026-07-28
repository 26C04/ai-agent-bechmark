"""Opt-in local Ollama integration coverage for the smoke command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.ollama_adapter import FakeOllamaAdapter
from ocrbench.prompts import PromptVersion


def _write_image(path: Path) -> None:
    """Create a public-safe synthetic image accepted by the smoke command."""
    Image.new("RGB", (24, 18), (10, 20, 30)).save(path, format="PNG")


def test_smoke_json_parse_failure_returns_exit_one_without_raw_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep JSON diagnostics safe when strict parsing fails after its sole retry."""
    image_path = tmp_path / "synthetic.png"
    _write_image(image_path)
    adapter = FakeOllamaAdapter(["PRIVATE_RAW_RESPONSE", "PRIVATE_RAW_RESPONSE"])
    prompt = PromptVersion("base", "a" * 12, "synthetic prompt")
    monkeypatch.setattr(cli_module, "create_adapter", lambda: adapter)
    monkeypatch.setattr(cli_module, "get_active", lambda root, name: prompt)

    assert main(["smoke", "--model", "gemma4:12b", "--image", str(image_path), "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["parse_status"] == "not_json"
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["succeeded"] is False
    assert "PRIVATE_RAW_RESPONSE" not in json.dumps(payload)


@pytest.mark.ollama
def test_smoke_command_runs_against_a_local_ollama_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Require a running local Ollama server and a pulled model; never run this in normal CI."""
    image_path = tmp_path / "synthetic-smoke.png"
    _write_image(image_path)
    model_tag = os.environ.get("OCRBENCH_SMOKE_REAL_MODEL", "gemma4:12b")
    monkeypatch.setenv("OCRBENCH_ADAPTER", "real")

    assert main(["smoke", "--model", model_tag, "--image", str(image_path)]) == 0
    assert "Final parse status" in capsys.readouterr().out
