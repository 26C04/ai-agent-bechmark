"""Append-only, content-addressed prompt version registry."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from ocrbench.config import OcrBenchError

_HASH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{12}$")
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_TIMESTAMP_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{8}T\d{6}Z$")
_ACTIONS: Final[frozenset[str]] = frozenset({"add", "activate", "rollback"})
_THREAD_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class PromptRegistryError(OcrBenchError):
    """Raised when a prompt registry is malformed, unsafe, or inconsistent."""


@dataclass(frozen=True)
class PromptVersion:
    """An immutable content-addressed prompt version."""

    name: str
    hash: str
    text: str


@dataclass(frozen=True)
class _HistoryEntry:
    timestamp: str
    action: Literal["add", "activate", "rollback"]
    hash: str
    note: str


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or path.is_junction():
        return True
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    return bool(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        and getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise PromptRegistryError(
            "prompt name must start with a lowercase letter and contain only lowercase "
            "letters, digits, or hyphens"
        )


def _validate_hash(hash12: str) -> None:
    if not isinstance(hash12, str) or _HASH_PATTERN.fullmatch(hash12) is None:
        raise PromptRegistryError("prompt hash must contain exactly 12 lowercase hex digits")


def _registry_dir(registry_root: Path, *, create: bool) -> Path:
    root = Path(registry_root)
    try:
        if create:
            root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse(root):
            raise PromptRegistryError("registry root must not be a symlink or reparse alias")
        details = root.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise PromptRegistryError(f"registry root does not exist: {root}") from error
    except OSError as error:
        raise PromptRegistryError(f"cannot inspect registry root {root}: {error}") from error
    if not stat.S_ISDIR(details.st_mode):
        raise PromptRegistryError(f"registry root is not a directory: {root}")
    return root


def _prompt_dir(registry_root: Path, name: str, *, create: bool) -> Path:
    _validate_name(name)
    root = _registry_dir(registry_root, create=create)
    directory = root / name
    try:
        if create:
            directory.mkdir(exist_ok=True)
        if _is_link_or_reparse(directory):
            raise PromptRegistryError(
                f"prompt directory {name!r} must not be a symlink or reparse alias"
            )
        details = directory.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise PromptRegistryError(f"prompt {name!r} does not exist") from error
    except OSError as error:
        raise PromptRegistryError(f"cannot inspect prompt directory {name!r}: {error}") from error
    if not stat.S_ISDIR(details.st_mode):
        raise PromptRegistryError(f"prompt path {name!r} is not a directory")
    return directory


def _regular_file(path: Path, label: str) -> Path:
    try:
        if _is_link_or_reparse(path):
            raise PromptRegistryError(f"{label} must not be a symlink or reparse alias")
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise PromptRegistryError(f"{label} is missing") from error
    except OSError as error:
        raise PromptRegistryError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(details.st_mode):
        raise PromptRegistryError(f"{label} is not a regular file")
    return path


def _read_text(path: Path, label: str) -> str:
    _regular_file(path, label)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return stream.read()
    except (OSError, UnicodeError) as error:
        raise PromptRegistryError(f"cannot read {label}: {error}") from error


def _version_path(directory: Path, hash12: str) -> Path:
    _validate_hash(hash12)
    return directory / f"{hash12}.txt"


def _read_version(directory: Path, name: str, hash12: str) -> PromptVersion:
    path = _version_path(directory, hash12)
    text = _read_text(path, f"prompt version {name}/{path.name}")
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if actual_hash != hash12:
        raise PromptRegistryError(
            f"prompt version {name}/{path.name} has content hash {actual_hash}, not {hash12}"
        )
    return PromptVersion(name, hash12, text)


def _write_new_version(path: Path, text: str) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return False
    except OSError as error:
        raise PromptRegistryError(f"cannot create prompt version {path.name}: {error}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, UnicodeError) as error:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PromptRegistryError(f"cannot write prompt version {path.name}: {error}") from error
    return True


def _history_path(directory: Path) -> Path:
    return directory / "history.jsonl"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_history(directory: Path, name: str) -> list[_HistoryEntry]:
    text = _read_text(_history_path(directory), f"prompt history {name}/history.jsonl")
    if not text:
        return []
    if not text.endswith("\n"):
        raise PromptRegistryError(f"prompt history {name}/history.jsonl must end with a newline")
    entries: list[_HistoryEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise PromptRegistryError(
                f"prompt history {name}/history.jsonl has an empty line at {line_number}"
            )
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise PromptRegistryError(
                f"prompt history {name}/history.jsonl line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(value, dict) or set(value) != {"ts", "action", "hash", "note"}:
            raise PromptRegistryError(
                f"prompt history {name}/history.jsonl line {line_number} has an invalid schema"
            )
        timestamp, action, hash12, note = value["ts"], value["action"], value["hash"], value["note"]
        if (
            not isinstance(timestamp, str)
            or _TIMESTAMP_PATTERN.fullmatch(timestamp) is None
            or not isinstance(action, str)
            or action not in _ACTIONS
            or not isinstance(hash12, str)
            or _HASH_PATTERN.fullmatch(hash12) is None
            or not isinstance(note, str)
        ):
            raise PromptRegistryError(
                f"prompt history {name}/history.jsonl line {line_number} has invalid values"
            )
        try:
            datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
        except ValueError as error:
            raise PromptRegistryError(
                f"prompt history {name}/history.jsonl line {line_number} "
                "has an invalid UTC timestamp"
            ) from error
        _read_version(directory, name, hash12)
        entries.append(
            _HistoryEntry(
                timestamp, cast(Literal["add", "activate", "rollback"], action), hash12, note
            )
        )
    return entries


def _version_hashes(directory: Path, name: str) -> list[str]:
    try:
        files = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        raise PromptRegistryError(f"cannot list prompt versions for {name!r}: {error}") from error
    hashes: list[str] = []
    for entry in files:
        if entry.name in {"active", "history.jsonl"}:
            _regular_file(entry, f"registry component {name}/{entry.name}")
            continue
        match = re.fullmatch(r"([0-9a-f]{12})\.txt", entry.name)
        if match is None:
            raise PromptRegistryError(f"unexpected registry component {name}/{entry.name}")
        _regular_file(entry, f"prompt version {name}/{entry.name}")
        hashes.append(match.group(1))
    return hashes


def _active_exists(directory: Path) -> bool:
    marker = directory / "active"
    return marker.exists() or marker.is_symlink()


def _validate_registry(directory: Path, name: str, *, require_active: bool) -> list[str]:
    hashes = _version_hashes(directory, name)
    history = _parse_history(directory, name)
    add_hashes = {entry.hash for entry in history if entry.action == "add"}
    if set(hashes) != add_hashes:
        raise PromptRegistryError(
            f"prompt history {name}/history.jsonl does not describe every version"
        )
    active_events = [entry for entry in history if entry.action in {"activate", "rollback"}]
    if _active_exists(directory):
        active_hash = _read_active_hash(directory, name)
        if not active_events or active_events[-1].hash != active_hash:
            raise PromptRegistryError(
                f"active marker for {name!r} does not match the latest activation history"
            )
    elif require_active or active_events:
        raise PromptRegistryError(f"active marker for {name!r} is missing")
    return hashes


def _append_history(
    directory: Path,
    name: str,
    action: Literal["add", "activate", "rollback"],
    hash12: str,
    note: str,
    *,
    allow_create: bool = False,
) -> None:
    if not isinstance(note, str):
        raise PromptRegistryError("history note must be a string")
    _validate_hash(hash12)
    path = _history_path(directory)
    if path.exists() or path.is_symlink():
        _parse_history(directory, name)
    elif allow_create:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as error:
            raise PromptRegistryError(
                f"cannot create prompt history for {name!r}: {error}"
            ) from error
        os.close(descriptor)
    else:
        raise PromptRegistryError(f"prompt history {name}/history.jsonl is missing")
    entry = {
        "ts": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "action": action,
        "hash": hash12,
        "note": note,
    }
    try:
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, UnicodeError) as error:
        raise PromptRegistryError(f"cannot append prompt history for {name!r}: {error}") from error


def _read_active_hash(directory: Path, name: str) -> str:
    text = _read_text(directory / "active", f"active marker for {name!r}")
    if not text.endswith("\n") or text.count("\n") != 1:
        raise PromptRegistryError(
            f"active marker for {name!r} must contain exactly one newline-terminated hash"
        )
    hash12 = text[:-1]
    _validate_hash(hash12)
    return hash12


def _write_active(directory: Path, name: str, hash12: str) -> None:
    _validate_hash(hash12)
    active = directory / "active"
    if _active_exists(directory):
        _regular_file(active, f"active marker for {name!r}")
    temporary = directory / f".active-{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(f"{hash12}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, active)
    except (OSError, UnicodeError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PromptRegistryError(
            f"cannot atomically update active marker for {name!r}: {error}"
        ) from error


@contextmanager
def _process_lock(root: Path, name: str) -> Iterator[None]:
    """Use an advisory OS lock that releases automatically when a process exits."""
    path = root / f".ocrbench-{name}.lock"
    if path.exists() or path.is_symlink():
        _regular_file(path, f"writer lock for {name!r}")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PromptRegistryError(f"writer lock for {name!r} is not a regular file")
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 30.0
            while True:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    if time.monotonic() >= deadline:
                        raise PromptRegistryError(
                            f"timed out waiting for writer lock for {name!r}"
                        ) from error
                    time.sleep(0.02)
        else:
            fcntl = __import__("fcntl")

            fcntl.flock(descriptor, fcntl.LOCK_EX)
    except (OSError, PromptRegistryError) as error:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if isinstance(error, PromptRegistryError):
            raise
        raise PromptRegistryError(f"cannot acquire writer lock for {name!r}: {error}") from error
    assert descriptor is not None
    try:
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl = __import__("fcntl")

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _writer_lock(registry_root: Path, name: str, *, create_root: bool) -> Iterator[None]:
    root = _registry_dir(registry_root, create=create_root)
    key = (str(root.absolute()), name)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with lock, _process_lock(root, name):
        yield


def add_prompt(registry_root: Path, name: str, text: str, note: str = "") -> PromptVersion:
    """Append a content-addressed prompt version, preserving existing versions."""
    if not isinstance(text, str):
        raise PromptRegistryError("prompt text must be a string")
    _validate_name(name)
    try:
        encoded_text = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PromptRegistryError("prompt text must be valid UTF-8") from error
    hash12 = hashlib.sha256(encoded_text).hexdigest()[:12]
    with _writer_lock(registry_root, name, create_root=True):
        directory = _prompt_dir(registry_root, name, create=True)
        history = _history_path(directory)
        history_present = history.exists() or history.is_symlink()
        if history_present:
            _validate_registry(directory, name, require_active=False)
        elif tuple(directory.iterdir()):
            raise PromptRegistryError(f"prompt history {name}/history.jsonl is missing")
        path = _version_path(directory, hash12)
        created = _write_new_version(path, text)
        version = _read_version(directory, name, hash12)
        if not created and version.text != text:
            raise PromptRegistryError(
                f"prompt version {name}/{path.name} conflicts with requested content"
            )
        _append_history(directory, name, "add", hash12, note, allow_create=not history_present)
        return version


def list_versions(registry_root: Path, name: str) -> list[PromptVersion]:
    """Return all valid prompt versions in deterministic hash order."""
    _validate_name(name)
    with _writer_lock(registry_root, name, create_root=False):
        directory = _prompt_dir(registry_root, name, create=False)
        return [
            _read_version(directory, name, item)
            for item in _validate_registry(directory, name, require_active=False)
        ]


def get_active(registry_root: Path, name: str) -> PromptVersion:
    """Return the version named by the validated active marker."""
    _validate_name(name)
    with _writer_lock(registry_root, name, create_root=False):
        directory = _prompt_dir(registry_root, name, create=False)
        _validate_registry(directory, name, require_active=True)
        return _read_version(directory, name, _read_active_hash(directory, name))


def activate(registry_root: Path, name: str, hash12: str, note: str = "") -> None:
    """Atomically make an existing version active and append its history event."""
    _validate_name(name)
    _validate_hash(hash12)
    with _writer_lock(registry_root, name, create_root=False):
        directory = _prompt_dir(registry_root, name, create=False)
        _validate_registry(directory, name, require_active=False)
        _read_version(directory, name, hash12)
        previous_hash = _read_active_hash(directory, name) if _active_exists(directory) else None
        _write_active(directory, name, hash12)
        try:
            _append_history(directory, name, "activate", hash12, note)
        except PromptRegistryError:
            if previous_hash is None:
                try:
                    (directory / "active").unlink()
                except OSError as error:
                    raise PromptRegistryError(
                        f"history append failed and active marker could not be removed: {error}"
                    ) from None
            else:
                _write_active(directory, name, previous_hash)
            raise


def rollback(registry_root: Path, name: str) -> PromptVersion:
    """Restore the active version immediately preceding the current active event."""
    _validate_name(name)
    with _writer_lock(registry_root, name, create_root=False):
        directory = _prompt_dir(registry_root, name, create=False)
        _validate_registry(directory, name, require_active=True)
        current_hash = _read_active_hash(directory, name)
        active_entries = [
            entry
            for entry in _parse_history(directory, name)
            if entry.action in {"activate", "rollback"}
        ]
        if len(active_entries) < 2:
            raise PromptRegistryError(
                f"prompt {name!r} has no previous active version to roll back to"
            )
        previous = active_entries[-2]
        _write_active(directory, name, previous.hash)
        try:
            _append_history(directory, name, "rollback", previous.hash, "")
        except PromptRegistryError:
            _write_active(directory, name, current_hash)
            raise
        return _read_version(directory, name, previous.hash)


def lint_prompt(text: str) -> list[str]:
    """Return conservative ADR §7 warnings without rejecting a prompt."""
    if not isinstance(text, str):
        raise PromptRegistryError("prompt text must be a string")
    warnings: list[str] = []
    if re.search(r"\b[xy]\s*=|\d\s*px\b|\bbbox\b|座標", text, flags=re.IGNORECASE):
        warnings.append("Possible absolute coordinate instruction detected (ADR §7).")
    if re.search(r"example|例[:\uFF1A]", text, flags=re.IGNORECASE):
        warnings.append(
            "Few-shot examples may not contain real values; confirm any example is synthetic (ADR §7)."
        )
    return warnings
