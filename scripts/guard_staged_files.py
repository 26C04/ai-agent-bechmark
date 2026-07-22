"""Reject staged files that may contain benchmark secrets or bulky scans."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_FILE_SIZE = 5_000_000
PUBLIC_FIXTURE_PREFIX = "tests/fixtures/public/"
FORBIDDEN_MEDIA_SUFFIXES = frozenset(
    {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}
)


@dataclass(frozen=True)
class Violation:
    """A staged path and the policy rule it violates."""

    path: str
    reason: str


def _git_path(path: str) -> str:
    """Return a canonical repository-relative Git path or raise ``ValueError``."""
    components = path.split("/")
    is_drive_path = len(path) >= 2 and path[1] == ":"
    if (
        not path
        or "\0" in path
        or "\\" in path
        or path.startswith("/")
        or is_drive_path
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise ValueError(f"non-canonical Git path: {path!r}")
    return path


def check_staged_paths(paths: list[str], size_of: Callable[[str], int]) -> list[Violation]:
    """Return all policy violations found among staged repository paths."""
    violations: list[Violation] = []

    for original_path in paths:
        try:
            path = _git_path(original_path)
        except ValueError:
            violations.append(
                Violation(original_path, "path is not a canonical repository-relative Git path")
            )
            continue

        lower_path = path.lower()
        filename = PurePosixPath(lower_path).name
        suffix = PurePosixPath(lower_path).suffix
        size = size_of(path)

        if size > MAX_FILE_SIZE:
            violations.append(Violation(path, f"file exceeds {MAX_FILE_SIZE:,} bytes"))

        if path.startswith(PUBLIC_FIXTURE_PREFIX):
            continue

        if lower_path.startswith(("data/", "runs/")):
            violations.append(Violation(path, "path is inside a forbidden data directory"))
        if suffix in FORBIDDEN_MEDIA_SUFFIXES:
            violations.append(Violation(path, "document or image files are forbidden"))
        if filename.endswith(".json") and (
            filename.startswith("gt_") or filename.endswith("_gt.json")
        ):
            violations.append(Violation(path, "ground-truth JSON files are forbidden"))
        if filename == ".env" or suffix == ".env":
            violations.append(Violation(path, "environment files are forbidden"))

    return violations


def _staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def _staged_size(path: str) -> int:
    """Return the stage-0 index blob size rather than the mutable worktree size."""
    result = subprocess.run(
        ["git", "cat-file", "-s", f":0:{path}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
    )
    return int(result.stdout.strip())


def main(argv: Sequence[str] | None = None) -> int:
    """Check the Git index and print actionable violations."""
    del argv
    try:
        violations = check_staged_paths(_staged_paths(), _staged_size)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Unable to inspect staged files: {error}", file=sys.stderr)
        return 1

    if not violations:
        return 0

    print("Commit blocked: staged files violate the repository data policy:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation.path}: {violation.reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
