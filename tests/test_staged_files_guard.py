"""Tests for the staged-file data leak guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.guard_staged_files import MAX_FILE_SIZE, _staged_size, check_staged_paths


@pytest.mark.parametrize(
    "path",
    [
        "data/x.json",
        "runs/result.json",
        "scan.pdf",
        "scan.jpg",
        "scan.jpeg",
        "scan.png",
        "scan.tif",
        "scan.tiff",
        "scan.bmp",
        "scan.webp",
        "scan.heic",
        "tests/gt_sample.json",
        "tests/sample_gt.json",
        ".env",
        "config/.env",
        "secrets.env",
    ],
)
def test_forbidden_staged_path_is_rejected(path: str) -> None:
    violations = check_staged_paths([path], lambda _path: 100)

    assert {violation.path for violation in violations} == {path}


def test_file_larger_than_limit_is_rejected() -> None:
    path = "src/ocrbench/large.py"

    violations = check_staged_paths([path], lambda _path: MAX_FILE_SIZE + 1)

    assert {violation.path for violation in violations} == {path}


def test_file_at_limit_is_allowed() -> None:
    assert check_staged_paths(["src/ocrbench/limit.py"], lambda _path: MAX_FILE_SIZE) == []


@pytest.mark.parametrize(
    "path",
    [
        "src/ocrbench/foo.py",
        "tests/fixtures/public/mini.png",
        "prompts/base/abc123.txt",
    ],
)
def test_safe_staged_path_is_allowed(path: str) -> None:
    assert check_staged_paths([path], lambda _path: 100) == []


def test_public_fixture_larger_than_limit_is_rejected() -> None:
    path = "tests/fixtures/public/big.png"

    violations = check_staged_paths([path], lambda _path: 6_000_000)

    assert {violation.path for violation in violations} == {path}


@pytest.mark.parametrize(
    "path",
    [
        "tests/fixtures/public/../private/leak.pdf",
        "tests/fixtures/public/./leak.pdf",
        "tests//fixtures/public/leak.pdf",
        "tests\\fixtures\\public\\leak.pdf",
        "/tests/fixtures/public/leak.pdf",
        "C:/tests/fixtures/public/leak.pdf",
    ],
)
def test_noncanonical_path_is_rejected_without_accessing_it(path: str) -> None:
    def unexpected_size_access(_path: str) -> int:
        raise AssertionError("unsafe paths must not reach the filesystem or Git index")

    violations = check_staged_paths([path], unexpected_size_access)

    assert len(violations) == 1
    assert violations[0].path == path
    assert "canonical" in violations[0].reason


def test_public_fixture_exception_requires_exact_casing() -> None:
    path = "TESTS/FIXTURES/PUBLIC/leak.PDF"

    violations = check_staged_paths([path], lambda _path: 100)

    assert {violation.path for violation in violations} == {path}
    assert any("image" in violation.reason for violation in violations)


def test_staged_size_uses_index_blob_not_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    staged_file = tmp_path / "oversized.bin"
    staged_file.write_bytes(b"x" * (MAX_FILE_SIZE + 1))
    subprocess.run(["git", "add", "--", staged_file.name], cwd=tmp_path, check=True)
    staged_file.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    assert _staged_size(staged_file.name) == MAX_FILE_SIZE + 1
    violations = check_staged_paths([staged_file.name], _staged_size)
    assert {violation.path for violation in violations} == {staged_file.name}


def test_gitignore_blocks_private_data_and_allows_public_fixture() -> None:
    cases = {
        "data/x.json": True,
        "foo.pdf": True,
        "tests/gt_sample.json": True,
        ".env": True,
        "tests/fixtures/public/sample.png": False,
        "TESTS/FIXTURES/PUBLIC/sample.png": True,
    }

    for path, expected_ignored in cases.items():
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.ignoreCase=false",
                "check-ignore",
                "--quiet",
                "--no-index",
                path,
            ],
            check=False,
        )
        assert (result.returncode == 0) is expected_ignored, path
