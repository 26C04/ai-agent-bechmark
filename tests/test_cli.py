"""Tests for the initial CLI stub."""

import pytest

from ocrbench import __version__
from ocrbench.cli import main


def test_version_option_returns_success_and_prints_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{__version__}\n"
