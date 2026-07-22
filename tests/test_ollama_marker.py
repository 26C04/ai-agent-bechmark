"""Placeholder that keeps the explicit Ollama test command well-formed."""

import pytest


@pytest.mark.ollama
@pytest.mark.skip(reason="real Ollama integration coverage is added by T18")
def test_real_ollama_placeholder() -> None:
    """Document that real Ollama tests require an explicit local setup."""
