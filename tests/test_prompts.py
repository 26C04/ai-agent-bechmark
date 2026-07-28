"""Tests for the append-only content-addressed prompt registry."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Literal

import pytest

import ocrbench.prompts as prompts_module
from ocrbench.prompts import (
    PromptRegistryError,
    activate,
    add_prompt,
    get_active,
    lint_prompt,
    list_versions,
    rollback,
)


def _history(root: Path, name: str) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in (root / name / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_prompt_lifecycle_and_history_are_append_only(tmp_path: Path) -> None:
    first = add_prompt(tmp_path, "base", "first", note="initial")
    activate(tmp_path, "base", first.hash, note="use initial")
    second = add_prompt(tmp_path, "base", "second", note="candidate")
    activate(tmp_path, "base", second.hash, note="use candidate")

    assert get_active(tmp_path, "base") == second
    assert rollback(tmp_path, "base") == first
    assert get_active(tmp_path, "base") == first
    assert list_versions(tmp_path, "base") == sorted(
        [first, second], key=lambda version: version.hash
    )
    assert [(entry["action"], entry["hash"]) for entry in _history(tmp_path, "base")] == [
        ("add", first.hash),
        ("activate", first.hash),
        ("add", second.hash),
        ("activate", second.hash),
        ("rollback", first.hash),
    ]


def test_add_is_idempotent_but_records_each_request(tmp_path: Path) -> None:
    first = add_prompt(tmp_path, "base", "same")
    again = add_prompt(tmp_path, "base", "same")

    assert again == first
    assert list((tmp_path / "base").glob("*.txt")) == [tmp_path / "base" / f"{first.hash}.txt"]
    assert [entry["action"] for entry in _history(tmp_path, "base")] == ["add", "add"]


def test_add_rejects_text_that_cannot_be_encoded_as_utf8(tmp_path: Path) -> None:
    with pytest.raises(PromptRegistryError, match="valid UTF-8"):
        add_prompt(tmp_path, "base", "invalid\ud800")

    assert not (tmp_path / "base").exists()


def test_activation_and_rollback_reject_missing_state(tmp_path: Path) -> None:
    version = add_prompt(tmp_path, "base", "first")
    with pytest.raises(PromptRegistryError, match="exactly 12"):
        activate(tmp_path, "base", "nope")
    with pytest.raises(PromptRegistryError, match="prompt version"):
        activate(tmp_path, "base", "0" * 12)
    with pytest.raises(PromptRegistryError, match="active marker"):
        get_active(tmp_path, "base")
    activate(tmp_path, "base", version.hash)
    with pytest.raises(PromptRegistryError, match="no previous active"):
        rollback(tmp_path, "base")


def test_registry_rejects_tampering_and_unsafe_names(tmp_path: Path) -> None:
    version = add_prompt(tmp_path, "base", "first")
    path = tmp_path / "base" / f"{version.hash}.txt"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(PromptRegistryError, match="content hash"):
        list_versions(tmp_path, "base")
    with pytest.raises(PromptRegistryError, match="prompt name"):
        add_prompt(tmp_path, "../escape", "text")


def test_missing_history_is_tampering_after_the_first_add(tmp_path: Path) -> None:
    add_prompt(tmp_path, "base", "first")
    (tmp_path / "base" / "history.jsonl").unlink()

    with pytest.raises(PromptRegistryError, match=r"history.*missing"):
        list_versions(tmp_path, "base")
    with pytest.raises(PromptRegistryError, match=r"history.*missing"):
        add_prompt(tmp_path, "base", "second")


def test_crlf_version_text_is_hashed_without_newline_translation(tmp_path: Path) -> None:
    text = "first\r\nsecond\r\n"
    version = add_prompt(tmp_path, "base", text)

    assert version.hash == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    assert list_versions(tmp_path, "base") == [version]


def test_registry_rejects_symlink_components_when_supported(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    directory = tmp_path / "base"
    directory.mkdir()
    link = directory / "aaaaaaaaaaaa.txt"
    try:
        os.symlink(outside, link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(PromptRegistryError, match="symlink"):
        list_versions(tmp_path, "base")


def test_lint_prompt_warns_conservatively() -> None:
    assert lint_prompt("Read the material column and return JSON only.") == []
    warnings = lint_prompt("Use x=20 20px and bbox values. Example: a sample.")
    assert len(warnings) == 2
    assert "coordinate" in warnings[0].lower()
    assert "Few-shot" in warnings[1]


def test_active_is_restored_when_history_append_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = add_prompt(tmp_path, "base", "first")
    second = add_prompt(tmp_path, "base", "second")
    activate(tmp_path, "base", first.hash)
    original_append = prompts_module._append_history

    def fail_activate_append(
        directory: Path,
        name: str,
        action: Literal["add", "activate", "rollback"],
        hash12: str,
        note: str,
        *,
        allow_create: bool = False,
    ) -> None:
        if action == "activate":
            raise PromptRegistryError("simulated history failure")
        original_append(directory, name, action, hash12, note, allow_create=allow_create)

    monkeypatch.setattr(prompts_module, "_append_history", fail_activate_append)
    with pytest.raises(PromptRegistryError, match="simulated"):
        activate(tmp_path, "base", second.hash)

    assert get_active(tmp_path, "base") == first
    assert _history(tmp_path, "base")[-1]["hash"] == first.hash


def test_per_name_writers_do_not_lose_history_entries(tmp_path: Path) -> None:
    failures: list[Exception] = []

    def add(index: int) -> None:
        try:
            add_prompt(tmp_path, "base", f"prompt {index}")
        except Exception as error:  # pragma: no cover - assertion helper
            failures.append(error)

    threads = [threading.Thread(target=add, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(_history(tmp_path, "base")) == 12
    assert len(list_versions(tmp_path, "base")) == 12


def _add_from_process(root: str, index: int, start: object) -> None:
    start.wait()  # type: ignore[attr-defined]
    add_prompt(Path(root), "base", f"process prompt {index}")


def test_os_lock_serializes_independent_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_add_from_process, args=(str(tmp_path), index, start))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert len(_history(tmp_path, "base")) == 4
    assert len(list_versions(tmp_path, "base")) == 4


def test_tracked_base_prompt_is_consistent_and_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "prompts"
    version = get_active(root, "base")

    assert version.hash == hashlib.sha256(version.text.encode("utf-8")).hexdigest()[:12]
    assert lint_prompt(version.text) == []
    history = _history(root, "base")
    assert history[-1]["action"] == "activate"
    assert history[-1]["hash"] == version.hash
