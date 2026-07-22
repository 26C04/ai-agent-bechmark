"""Tests for environment-backed configuration."""

from collections.abc import Callable
from pathlib import Path

import pytest

from ocrbench.config import ALLOWED_MODELS, ConfigError, data_dir, final_dir, runs_dir

PathFunction = Callable[[], Path]


@pytest.mark.parametrize(
    ("variable_name", "function"),
    [
        ("OCRBENCH_DATA_DIR", data_dir),
        ("OCRBENCH_FINAL_DIR", final_dir),
        ("OCRBENCH_RUNS_DIR", runs_dir),
    ],
)
def test_configured_directory_is_returned(
    monkeypatch: pytest.MonkeyPatch, variable_name: str, function: PathFunction
) -> None:
    monkeypatch.setenv(variable_name, "C:/external/benchmark")
    assert function() == Path("C:/external/benchmark")


@pytest.mark.parametrize(
    ("variable_name", "function"),
    [
        ("OCRBENCH_DATA_DIR", data_dir),
        ("OCRBENCH_FINAL_DIR", final_dir),
        ("OCRBENCH_RUNS_DIR", runs_dir),
    ],
)
def test_missing_directory_configuration_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, variable_name: str, function: PathFunction
) -> None:
    monkeypatch.delenv(variable_name, raising=False)
    with pytest.raises(ConfigError, match=variable_name):
        function()


def test_allowed_models_match_the_initial_adr_allowlist() -> None:
    assert ALLOWED_MODELS == (
        "gemma4:12b",
        "qwen3.5:9b",
        "minicpm-v4.5:8b",
        "glm-ocr:bf16",
        "ministral-3:14b",
    )
