"""Explicit integration tests for a locally installed and configured Ollama server."""

import os

import pytest

from ocrbench.config import ALLOWED_MODELS
from ocrbench.ollama_adapter import ModelNotFoundError, RealOllamaAdapter

pytestmark = pytest.mark.ollama

_TEST_MODEL_ENV = "OCRBENCH_OLLAMA_TEST_MODEL"


def _pulled_model() -> str:
    """Return the model tag that must already be pulled before this test is run."""
    return os.environ.get(_TEST_MODEL_ENV, ALLOWED_MODELS[0])


def test_local_server_reports_a_version() -> None:
    """Require a running Ollama server at localhost with its version endpoint available."""
    assert RealOllamaAdapter().server_version()


def test_local_server_resolves_a_pulled_model_digest() -> None:
    """Require the selected model to be pulled before explicitly running Ollama tests."""
    digest = RealOllamaAdapter().resolve_digest(_pulled_model())

    assert digest.startswith("sha256:")


def test_local_server_rejects_an_absent_exact_model_tag() -> None:
    """Require a running local server; no model with this synthetic tag may be installed."""
    with pytest.raises(ModelNotFoundError):
        RealOllamaAdapter().resolve_digest("ocrbench-definitely-absent:never")
