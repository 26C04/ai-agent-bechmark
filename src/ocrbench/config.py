"""Shared configuration constants and environment-backed paths."""

import os
from pathlib import Path
from typing import Final


class OcrBenchError(Exception):
    """Base exception for expected OCR benchmark errors."""


class ConfigError(OcrBenchError):
    """Raised when required benchmark configuration is unavailable."""


SEED: Final[int] = 20260721
ALLOWED_MODELS: Final[tuple[str, ...]] = (
    "gemma4:12b",
    "qwen3.5:9b",
    "minicpm-v4.5:8b",
    "glm-ocr:bf16",
    "ministral-3:14b",
)


def _required_path(variable_name: str) -> Path:
    value = os.environ.get(variable_name)
    if value is None or not value.strip():
        raise ConfigError(f"{variable_name} must be set")
    return Path(value)


def data_dir() -> Path:
    """Return the external Development and Selection dataset root."""
    return _required_path("OCRBENCH_DATA_DIR")


def final_dir() -> Path:
    """Return the external, ACL-isolated Final test dataset root."""
    return _required_path("OCRBENCH_FINAL_DIR")


def runs_dir() -> Path:
    """Return the external benchmark run output root."""
    return _required_path("OCRBENCH_RUNS_DIR")
