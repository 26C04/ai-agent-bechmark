"""Secure dataset manifests, split validation, and reproducible fingerprints.

Real benchmark data lives outside the repository in one of these roots::

    <dataset-root>/
    ├── raw/<doc-id>.<pdf|jpg|jpeg|png>
    ├── gt/<doc-id>.json
    └── manifest.json

Manifest file names are relative to ``raw/`` and must be anonymous, canonical
single-component names. All file hashing is streamed so source documents do
not need to fit in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from ocrbench.config import OcrBenchError
from ocrbench.schema import OrderDocument

SourceKind = Literal["scan", "photo"]
Split = Literal["dev", "selection", "final"]

_ALLOWED_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png"})
_DOC_ID_PATTERN = re.compile(r"^doc-[0-9a-f]{12}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024
_CSV_COLUMNS = ("file", "source_kind", "split")
_EXPECTED_SPLIT_COUNTS: tuple[tuple[Split, int, int, int], ...] = (
    ("dev", 40, 20, 20),
    ("selection", 20, 10, 10),
    ("final", 40, 20, 20),
)
_FileIdentity = tuple[int, int]


class DatasetError(OcrBenchError):
    """Report one or more deterministic dataset validation violations."""

    violations: tuple[str, ...]

    def __init__(self, violations: str | Iterable[str]) -> None:
        items = (violations,) if isinstance(violations, str) else tuple(violations)
        if not items:
            items = ("an unspecified dataset error occurred",)
        self.violations = items
        message = "Dataset validation failed:\n" + "\n".join(
            f"  - {violation}" for violation in items
        )
        super().__init__(message)


class DocEntry(BaseModel):
    """One anonymous source document and its fixed benchmark assignment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    doc_id: str
    source_kind: SourceKind
    file: str
    sha256: str
    split: Split

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        """Require the lowercase content-derived anonymous ID format."""
        if _DOC_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("doc_id must match 'doc-' followed by 12 lowercase hex digits")
        return value

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        """Require a canonical anonymous file name relative to ``raw/``."""
        _validate_raw_filename(value)
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a complete lowercase SHA-256 digest."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must contain exactly 64 lowercase hex digits")
        return value

    @model_validator(mode="after")
    def validate_anonymous_identity(self) -> Self:
        """Keep the anonymous ID, digest, and stored file name consistent."""
        expected_id = f"doc-{self.sha256[:12]}"
        violations: list[str] = []
        if self.doc_id != expected_id:
            violations.append(f"doc_id must be {expected_id!r} for the declared sha256")
        if PurePosixPath(self.file).stem != self.doc_id:
            violations.append("file stem must equal doc_id")
        if violations:
            raise ValueError("; ".join(violations))
        return self


class Manifest(BaseModel):
    """Versioned dataset manifest with no implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    docs: list[DocEntry]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        """Reject booleans and coercible strings even though they compare to ``1``."""
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value


@dataclass(frozen=True)
class _ManifestProbe:
    index: int
    label: str
    doc_id: str | None
    file: str | None
    sha256: str | None


@dataclass
class _BuildRecord:
    row_number: int
    original_name: str
    source_kind: str
    split: str
    source: Path | None = None
    extension: str | None = None
    sha256: str | None = None
    doc_id: str | None = None
    target: Path | None = None
    file_identity: _FileIdentity | None = None


@dataclass
class _RenameState:
    source: Path
    staging: Path
    target: Path
    location: Literal["source", "staging", "linked", "target"] = "source"


class _MutationIntegrityError(Exception):
    """Carry deterministic post-mutation integrity violations to rollback."""

    violations: tuple[str, ...]

    def __init__(self, violations: Iterable[str]) -> None:
        self.violations = tuple(violations)
        super().__init__("; ".join(self.violations))


def _validate_raw_filename(value: str) -> None:
    if (
        not value
        or value != value.strip()
        or "\0" in value
        or "/" in value
        or "\\" in value
        or ":" in value
        or value in {".", ".."}
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
    ):
        raise ValueError(
            "file must be a canonical single-component path relative to the raw directory"
        )
    suffix = PurePosixPath(value).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise ValueError(f"file extension must be one of: {allowed}")


def _sha256_file(path: Path) -> str:
    """Return a file SHA-256 while reading bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(info: os.stat_result) -> _FileIdentity:
    return info.st_dev, info.st_ino


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether a logical path is a symlink, junction, or reparse alias."""
    if path.is_symlink() or path.is_junction():
        return True
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _root_anchor(root: Path) -> tuple[Path | None, str | None]:
    try:
        if not root.exists():
            return None, f"dataset root does not exist: {root}"
        if not root.is_dir():
            return None, f"dataset root is not a directory: {root}"
        return root.resolve(strict=True), None
    except OSError as error:
        return None, f"cannot resolve dataset root {root}: {error}"


def _resolve_member(
    root: Path,
    anchor: Path,
    directory: str,
    filename: str,
) -> tuple[Path | None, str | None]:
    base = root / directory
    candidate = base / filename
    try:
        resolved_base = base.resolve(strict=False)
        if not resolved_base.is_relative_to(anchor):
            return None, f"{directory}/ escapes the dataset root through a symlink"
        resolved_candidate = candidate.resolve(strict=False)
        if not resolved_candidate.is_relative_to(resolved_base):
            return (
                None,
                f"{directory}/{filename} escapes the {directory}/ directory through a symlink",
            )
        return resolved_candidate, None
    except (OSError, RuntimeError) as error:
        return None, f"cannot safely resolve {directory}/{filename}: {error}"


def _regular_member(
    root: Path,
    anchor: Path,
    directory: str,
    filename: str,
) -> tuple[Path | None, str | None]:
    logical_directory = root / directory
    logical_path = logical_directory / filename
    try:
        if _is_link_or_reparse(logical_directory):
            return None, f"{directory}/ must not be a symlink or reparse alias"
        if _is_link_or_reparse(logical_path):
            return (
                None,
                f"{directory}/{filename} must not be a symlink or reparse alias",
            )
    except OSError as error:
        return None, f"cannot inspect {directory}/{filename} for aliases: {error}"

    path, resolve_error = _resolve_member(root, anchor, directory, filename)
    if resolve_error is not None or path is None:
        return None, resolve_error
    try:
        if not path.exists():
            return None, f"{directory}/{filename} is missing"
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            return None, f"{directory}/{filename} is not a regular file"
    except OSError as access_error:
        return None, f"cannot inspect {directory}/{filename}: {access_error}"
    return path, None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)


def _validation_messages(error: ValidationError) -> list[str]:
    messages: list[str] = []
    for detail in error.errors(include_url=False):
        location = ".".join(str(component) for component in detail["loc"]) or "<entry>"
        messages.append(f"{location}: {detail['msg']}")
    return messages


def _probe_from_mapping(index: int, value: dict[str, object]) -> _ManifestProbe:
    raw_doc_id = value.get("doc_id")
    doc_id = (
        raw_doc_id
        if isinstance(raw_doc_id, str) and _DOC_ID_PATTERN.fullmatch(raw_doc_id)
        else None
    )
    raw_file = value.get("file")
    file: str | None = None
    if isinstance(raw_file, str):
        try:
            _validate_raw_filename(raw_file)
        except ValueError:
            pass
        else:
            file = raw_file
    raw_sha256 = value.get("sha256")
    sha256 = (
        raw_sha256
        if isinstance(raw_sha256, str) and _SHA256_PATTERN.fullmatch(raw_sha256)
        else None
    )
    label_value = raw_doc_id if isinstance(raw_doc_id, str) else f"docs[{index}]"
    return _ManifestProbe(
        index=index,
        label=f"docs[{index}] ({label_value!r})",
        doc_id=doc_id,
        file=file,
        sha256=sha256,
    )


def _parse_manifest_structure(document: object) -> tuple[list[str], list[_ManifestProbe]]:
    violations: list[str] = []
    probes: list[_ManifestProbe] = []
    if not isinstance(document, dict):
        return ["manifest.json must contain a JSON object"], probes

    mapping = cast(dict[str, object], document)
    allowed_keys = {"schema_version", "docs"}
    for key in sorted(set(mapping) - allowed_keys):
        violations.append(f"manifest contains forbidden top-level field {key!r}")

    if "schema_version" not in mapping:
        violations.append("manifest.schema_version is required")
    else:
        version = mapping["schema_version"]
        if type(version) is not int or version != 1:
            violations.append("manifest.schema_version must be the integer 1")

    if "docs" not in mapping:
        violations.append("manifest.docs is required")
        return violations, probes
    raw_docs = mapping["docs"]
    if not isinstance(raw_docs, list):
        violations.append("manifest.docs must be a JSON array")
        return violations, probes

    for index, raw_entry in enumerate(raw_docs):
        if not isinstance(raw_entry, dict):
            violations.append(f"docs[{index}] must be a JSON object")
            continue
        entry_mapping = cast(dict[str, object], raw_entry)
        probes.append(_probe_from_mapping(index, entry_mapping))
        try:
            DocEntry.model_validate(entry_mapping)
        except ValidationError as error:
            violations.extend(
                f"docs[{index}] is invalid: {message}" for message in _validation_messages(error)
            )
    return violations, probes


def _load_gt_violation(path: Path, display_path: str) -> str | None:
    try:
        document = _read_json(path)
        OrderDocument.model_validate(document)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValidationError):
            details = "; ".join(_validation_messages(error))
        else:
            details = str(error)
        return f"{display_path} is invalid: {details}"
    return None


def load_manifest(root: Path) -> Manifest:
    """Load and comprehensively validate ``manifest.json``.

    Once the JSON object and ``docs`` array are readable, independent structural,
    duplicate-ID, raw-file, hash, and ground-truth violations are accumulated in
    a deterministic order before :class:`DatasetError` is raised.
    """
    anchor, root_error = _root_anchor(root)
    if root_error is not None or anchor is None:
        raise DatasetError(root_error or "cannot resolve dataset root")

    logical_manifest = root / "manifest.json"
    try:
        if _is_link_or_reparse(logical_manifest):
            raise DatasetError("manifest.json must not be a symlink, junction, or reparse alias")
        manifest_info = logical_manifest.stat(follow_symlinks=False)
        if not stat.S_ISREG(manifest_info.st_mode):
            raise DatasetError("manifest.json is not a regular file")
    except FileNotFoundError as error:
        raise DatasetError("manifest.json is missing") from error
    except DatasetError:
        raise
    except OSError as error:
        raise DatasetError(f"cannot inspect logical manifest.json: {error}") from error

    manifest_path, path_error = _resolve_member(root, anchor, ".", "manifest.json")
    if path_error is not None or manifest_path is None:
        raise DatasetError(path_error or "cannot resolve manifest.json")
    try:
        document = _read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DatasetError(f"manifest.json is not valid UTF-8 JSON: {error}") from error

    structural_violations, probes = _parse_manifest_structure(document)
    duplicate_violations: list[str] = []
    positions: dict[str, list[int]] = defaultdict(list)
    for probe in probes:
        if probe.doc_id is not None:
            positions[probe.doc_id].append(probe.index)
    for doc_id in sorted(positions):
        indexes = positions[doc_id]
        if len(indexes) > 1:
            rendered = ", ".join(f"docs[{index}]" for index in indexes)
            duplicate_violations.append(f"duplicate doc_id {doc_id!r} appears in {rendered}")

    raw_violations: list[tuple[str, str]] = []
    gt_violations: list[tuple[str, str]] = []
    for probe in probes:
        if probe.file is not None:
            raw_path, raw_error = _regular_member(root, anchor, "raw", probe.file)
            if raw_error is not None or raw_path is None:
                raw_violations.append(
                    (probe.label, f"{probe.label}: {raw_error or 'raw file is unavailable'}")
                )
            elif probe.sha256 is not None:
                try:
                    actual_sha256 = _sha256_file(raw_path)
                except OSError as error:
                    raw_violations.append(
                        (probe.label, f"{probe.label}: cannot hash raw/{probe.file}: {error}")
                    )
                else:
                    if actual_sha256 != probe.sha256:
                        raw_violations.append(
                            (
                                probe.label,
                                f"{probe.label}: raw/{probe.file} sha256 mismatch "
                                f"(declared {probe.sha256}, actual {actual_sha256})",
                            )
                        )

        if probe.doc_id is not None:
            gt_name = f"{probe.doc_id}.json"
            gt_path, gt_error = _regular_member(root, anchor, "gt", gt_name)
            if gt_error is not None or gt_path is None:
                gt_violations.append(
                    (probe.label, f"{probe.label}: {gt_error or 'ground truth is unavailable'}")
                )
            else:
                invalid_gt = _load_gt_violation(gt_path, f"gt/{gt_name}")
                if invalid_gt is not None:
                    gt_violations.append((probe.label, f"{probe.label}: {invalid_gt}"))

    violations = [
        *structural_violations,
        *duplicate_violations,
        *(message for _, message in sorted(raw_violations)),
        *(message for _, message in sorted(gt_violations)),
    ]
    if violations:
        raise DatasetError(violations)

    try:
        return Manifest.model_validate(document)
    except ValidationError as error:
        raise DatasetError(_validation_messages(error)) from error


def validate_split_counts(manifest: Manifest, *, strict: bool) -> list[str]:
    """Compare a manifest with the exact ADR split/source-kind count table."""
    counts = Counter((entry.split, entry.source_kind) for entry in manifest.docs)
    violations: list[str] = []
    for split, expected_total, expected_scan, expected_photo in _EXPECTED_SPLIT_COUNTS:
        actual_scan = counts[(split, "scan")]
        actual_photo = counts[(split, "photo")]
        actual_total = actual_scan + actual_photo
        if actual_total != expected_total:
            violations.append(
                f"split {split!r}: expected {expected_total} documents, found {actual_total}"
            )
        if actual_scan != expected_scan:
            violations.append(
                f"split {split!r} source_kind 'scan': expected {expected_scan}, found {actual_scan}"
            )
        if actual_photo != expected_photo:
            violations.append(
                f"split {split!r} source_kind 'photo': expected {expected_photo}, "
                f"found {actual_photo}"
            )
    if strict and violations:
        raise DatasetError(violations)
    return violations


def _framed_update(digest: object, label: str, payload: bytes) -> None:
    hasher = cast("hashlib._Hash", digest)
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(4, "big"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def dataset_fingerprint(root: Path, manifest: Manifest) -> str:
    """Hash canonical manifest data and the current raw/GT bytes."""
    anchor, root_error = _root_anchor(root)
    if root_error is not None or anchor is None:
        raise DatasetError(root_error or "cannot resolve dataset root")

    sorted_docs = sorted(manifest.docs, key=lambda entry: entry.doc_id)
    duplicate_ids = [
        doc_id
        for doc_id, count in Counter(entry.doc_id for entry in sorted_docs).items()
        if count > 1
    ]
    if duplicate_ids:
        raise DatasetError(
            f"duplicate doc_id {doc_id!r} cannot be fingerprinted"
            for doc_id in sorted(duplicate_ids)
        )

    canonical_manifest = json.dumps(
        {
            "docs": [entry.model_dump(mode="json") for entry in sorted_docs],
            "schema_version": manifest.schema_version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    file_digests: list[tuple[str, bytes, bytes]] = []
    violations: list[str] = []
    for entry in sorted_docs:
        raw_path, raw_error = _regular_member(root, anchor, "raw", entry.file)
        gt_name = f"{entry.doc_id}.json"
        gt_path, gt_error = _regular_member(root, anchor, "gt", gt_name)
        if raw_error is not None or raw_path is None:
            violations.append(f"{entry.doc_id}: {raw_error or 'raw file is unavailable'}")
        if gt_error is not None or gt_path is None:
            violations.append(f"{entry.doc_id}: {gt_error or 'ground truth is unavailable'}")
        if raw_path is None or gt_path is None:
            continue
        try:
            raw_digest = bytes.fromhex(_sha256_file(raw_path))
        except OSError as error:
            violations.append(f"{entry.doc_id}: cannot hash raw/{entry.file}: {error}")
            continue
        try:
            gt_digest = bytes.fromhex(_sha256_file(gt_path))
        except OSError as error:
            violations.append(f"{entry.doc_id}: cannot hash gt/{gt_name}: {error}")
            continue
        file_digests.append((entry.doc_id, raw_digest, gt_digest))
    if violations:
        raise DatasetError(violations)

    digest = hashlib.sha256()
    _framed_update(digest, "manifest", canonical_manifest)
    for doc_id, raw_digest, gt_digest in file_digests:
        _framed_update(digest, f"raw:{doc_id}", raw_digest)
        _framed_update(digest, f"gt:{doc_id}", gt_digest)
    return digest.hexdigest()


def _read_assignments_csv(assignments_csv: Path) -> tuple[list[_BuildRecord], list[str]]:
    records: list[_BuildRecord] = []
    violations: list[str] = []
    try:
        with assignments_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                return records, ["assignments CSV is empty"]

            duplicate_columns = sorted(
                column for column, count in Counter(header).items() if count > 1
            )
            missing_columns = sorted(set(_CSV_COLUMNS) - set(header))
            extra_columns = sorted(set(header) - set(_CSV_COLUMNS))
            if duplicate_columns:
                violations.append(
                    "assignments CSV has duplicate columns: " + ", ".join(duplicate_columns)
                )
            if missing_columns:
                violations.append(
                    "assignments CSV is missing columns: " + ", ".join(missing_columns)
                )
            if extra_columns:
                violations.append(
                    "assignments CSV has unsupported columns: " + ", ".join(extra_columns)
                )
            if duplicate_columns or missing_columns or extra_columns:
                return records, violations

            indexes = {name: header.index(name) for name in _CSV_COLUMNS}
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    violations.append(
                        f"CSV row {row_number}: expected {len(header)} values, found {len(row)}"
                    )
                    continue
                records.append(
                    _BuildRecord(
                        row_number=row_number,
                        original_name=row[indexes["file"]],
                        source_kind=row[indexes["source_kind"]],
                        split=row[indexes["split"]],
                    )
                )
    except (OSError, UnicodeError, csv.Error) as error:
        violations.append(f"cannot read assignments CSV {assignments_csv}: {error}")
    return records, violations


def _preflight_build(
    root: Path,
    assignments_csv: Path,
) -> tuple[list[_BuildRecord], list[str], Path | None]:
    records, violations = _read_assignments_csv(assignments_csv)
    anchor, root_error = _root_anchor(root)
    if root_error is not None or anchor is None:
        violations.append(root_error or "cannot resolve dataset root")
        return records, violations, None

    raw_dir = root / "raw"
    try:
        if _is_link_or_reparse(raw_dir):
            violations.append("raw/ must not be a symlink or reparse alias")
            return records, violations, anchor
        raw_resolved = raw_dir.resolve(strict=False)
        if not raw_resolved.is_relative_to(anchor):
            violations.append("raw/ escapes the dataset root through a symlink")
            return records, violations, anchor
        if not raw_dir.exists() or not raw_dir.is_dir():
            violations.append("raw/ directory is missing")
            return records, violations, anchor
    except (OSError, RuntimeError) as error:
        violations.append(f"cannot inspect raw/ directory: {error}")
        return records, violations, anchor

    for record in records:
        prefix = f"CSV row {record.row_number}"
        try:
            _validate_raw_filename(record.original_name)
        except ValueError as error:
            violations.append(f"{prefix}: invalid file {record.original_name!r}: {error}")
        else:
            record.extension = PurePosixPath(record.original_name).suffix.lower()
            logical_source = raw_dir / record.original_name
            try:
                if _is_link_or_reparse(logical_source):
                    violations.append(
                        f"{prefix}: source file must not be a symlink or reparse alias"
                    )
                    continue
            except OSError as error:
                violations.append(f"{prefix}: cannot inspect raw/{record.original_name}: {error}")
                continue
            source, path_error = _resolve_member(root, anchor, "raw", record.original_name)
            if path_error is not None or source is None:
                violations.append(f"{prefix}: {path_error or 'cannot resolve source file'}")
            else:
                record.source = source
                try:
                    if not source.exists():
                        violations.append(f"{prefix}: raw/{record.original_name} is missing")
                    else:
                        info = logical_source.stat(follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode):
                            violations.append(
                                f"{prefix}: raw/{record.original_name} is not a regular file"
                            )
                except OSError as error:
                    violations.append(
                        f"{prefix}: cannot inspect raw/{record.original_name}: {error}"
                    )

        if record.source_kind not in {"scan", "photo"}:
            violations.append(
                f"{prefix}: source_kind must be 'scan' or 'photo', found {record.source_kind!r}"
            )
        if record.split not in {"dev", "selection", "final"}:
            violations.append(
                f"{prefix}: split must be 'dev', 'selection', or 'final', found {record.split!r}"
            )

    source_rows: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.source is not None:
            source_rows[record.original_name.casefold()].append(record.row_number)
    for name_key in sorted(source_rows):
        rows = source_rows[name_key]
        if len(rows) > 1:
            rendered_rows = ", ".join(str(row) for row in rows)
            violations.append(
                f"duplicate CSV file assignment for {name_key!r} in rows {rendered_rows}"
            )

    for record in records:
        if record.source is None:
            continue
        logical_source = raw_dir / record.original_name
        try:
            before = logical_source.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                continue
            sha256 = _sha256_file(record.source)
            after = logical_source.stat(follow_symlinks=False)
            if _identity(before) != _identity(after):
                violations.append(
                    f"CSV row {record.row_number}: raw/{record.original_name} "
                    "changed identity while it was hashed"
                )
                continue
            record.file_identity = _identity(after)
            record.sha256 = sha256
            record.doc_id = f"doc-{sha256[:12]}"
            if record.extension is not None:
                record.target = raw_resolved / f"{record.doc_id}{record.extension}"
        except OSError as error:
            violations.append(
                f"CSV row {record.row_number}: cannot hash raw/{record.original_name}: {error}"
            )

    content_groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.sha256 is not None and record.source is not None:
            content_groups[record.sha256][record.original_name.casefold()].append(record.row_number)
    for sha256 in sorted(content_groups):
        distinct_sources = content_groups[sha256]
        if len(distinct_sources) > 1:
            rendered = ", ".join(sorted(distinct_sources))
            violations.append(f"duplicate source content with sha256 {sha256}: {rendered}")

    prefix_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.doc_id is not None and record.sha256 is not None:
            prefix_groups[record.doc_id].add(record.sha256)
    for doc_id in sorted(prefix_groups):
        hashes = prefix_groups[doc_id]
        if len(hashes) > 1:
            violations.append(f"SHA-256 prefix collision for {doc_id}: {', '.join(sorted(hashes))}")

    try:
        children_by_casefold: dict[str, list[Path]] = defaultdict(list)
        for child in raw_dir.iterdir():
            children_by_casefold[child.name.casefold()].append(child)
    except OSError as error:
        violations.append(f"cannot enumerate raw/ for target collisions: {error}")
        children_by_casefold = {}

    def platform_path_key(path: Path) -> str:
        absolute = os.path.abspath(path)
        normalized = os.path.normpath(absolute)
        return os.path.normcase(normalized)

    target_rows: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.target is None or record.source is None:
            continue
        target_key = record.target.name.casefold()
        target_rows[target_key].append(record.row_number)
        existing = children_by_casefold.get(target_key, [])
        conflicts: list[Path] = []
        source_path_key = platform_path_key(raw_dir / record.original_name)
        for child in existing:
            try:
                child_identity = _identity(child.stat(follow_symlinks=False))
            except OSError:
                conflicts.append(child)
                continue
            is_actual_source_entry = (
                record.file_identity is not None
                and child_identity == record.file_identity
                and platform_path_key(child) == source_path_key
            )
            if not is_actual_source_entry:
                conflicts.append(child)
        if conflicts:
            rendered = ", ".join(sorted(child.name for child in conflicts))
            violations.append(
                f"CSV row {record.row_number}: target raw/{record.target.name} "
                f"collides with existing path(s): {rendered}"
            )
    for target_key in sorted(target_rows):
        rows = target_rows[target_key]
        if len(rows) > 1:
            rendered_rows = ", ".join(str(row) for row in rows)
            violations.append(
                f"multiple CSV rows resolve to target {target_key!r}: {rendered_rows}"
            )

    manifest_path = root / "manifest.json"
    try:
        resolved_manifest = manifest_path.resolve(strict=False)
        if not resolved_manifest.is_relative_to(anchor):
            violations.append("manifest.json escapes the dataset root through a symlink")
        elif _is_link_or_reparse(manifest_path):
            violations.append("manifest.json must not be a symlink or reparse alias")
        elif manifest_path.exists() and not manifest_path.is_file():
            violations.append("manifest.json exists but is not a regular file")
    except (OSError, RuntimeError) as error:
        violations.append(f"cannot inspect manifest.json target: {error}")

    return records, violations, anchor


def _complete_record_violation(record: _BuildRecord) -> str | None:
    if (
        record.source is None
        or record.target is None
        or record.sha256 is None
        or record.doc_id is None
        or record.file_identity is None
    ):
        return f"CSV row {record.row_number}: preflight record is incomplete"
    return None


def _revalidate_sources(
    root: Path,
    anchor: Path,
    records: list[_BuildRecord],
) -> list[str]:
    """Verify every logical source again before the first rename."""
    violations: list[str] = []
    raw_dir = root / "raw"
    for record in records:
        prefix = f"CSV row {record.row_number}"
        incomplete = _complete_record_violation(record)
        if incomplete is not None:
            violations.append(incomplete)
            continue
        assert record.source is not None
        assert record.sha256 is not None
        assert record.file_identity is not None
        logical_source = raw_dir / record.original_name
        try:
            if _is_link_or_reparse(logical_source):
                violations.append(
                    f"{prefix}: raw/{record.original_name} became a symlink or reparse alias"
                )
                continue
            resolved, path_error = _resolve_member(root, anchor, "raw", record.original_name)
            if path_error is not None or resolved is None:
                violations.append(f"{prefix}: {path_error or 'cannot resolve source file'}")
                continue
            info = logical_source.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                violations.append(
                    f"{prefix}: raw/{record.original_name} is no longer a regular file"
                )
                continue
            if resolved != record.source or _identity(info) != record.file_identity:
                violations.append(
                    f"{prefix}: raw/{record.original_name} changed file identity after preflight"
                )
            actual_sha256 = _sha256_file(resolved)
            if actual_sha256 != record.sha256:
                violations.append(
                    f"{prefix}: raw/{record.original_name} changed content after preflight "
                    f"(expected {record.sha256}, actual {actual_sha256})"
                )
        except OSError as error:
            violations.append(f"{prefix}: cannot revalidate raw/{record.original_name}: {error}")
    return violations


def _revalidate_targets(
    root: Path,
    anchor: Path,
    records: list[_BuildRecord],
) -> list[str]:
    """Verify every anonymous target before publishing ``manifest.json``."""
    violations: list[str] = []
    for record in records:
        prefix = f"CSV row {record.row_number}"
        incomplete = _complete_record_violation(record)
        if incomplete is not None:
            violations.append(incomplete)
            continue
        assert record.target is not None
        assert record.sha256 is not None
        assert record.doc_id is not None
        assert record.file_identity is not None
        target_name = record.target.name
        logical_target = root / "raw" / target_name
        try:
            if _is_link_or_reparse(logical_target):
                violations.append(f"{prefix}: raw/{target_name} is a symlink or reparse alias")
                continue
            resolved, path_error = _resolve_member(root, anchor, "raw", target_name)
            if path_error is not None or resolved is None:
                violations.append(f"{prefix}: {path_error or 'cannot resolve target file'}")
                continue
            info = logical_target.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                violations.append(f"{prefix}: raw/{target_name} is not a regular file")
                continue
            if resolved != record.target or _identity(info) != record.file_identity:
                violations.append(
                    f"{prefix}: raw/{target_name} does not preserve the source file identity"
                )
            actual_sha256 = _sha256_file(resolved)
            actual_doc_id = f"doc-{actual_sha256[:12]}"
            if actual_sha256 != record.sha256:
                violations.append(
                    f"{prefix}: raw/{target_name} sha256 changed before manifest publication "
                    f"(expected {record.sha256}, actual {actual_sha256})"
                )
            if actual_doc_id != record.doc_id or PurePosixPath(target_name).stem != actual_doc_id:
                violations.append(
                    f"{prefix}: raw/{target_name} no longer matches its content-derived doc_id"
                )
        except OSError as error:
            violations.append(f"{prefix}: cannot revalidate raw/{target_name}: {error}")
    return violations


def _link_no_overwrite(source: Path, destination: Path) -> None:
    """Atomically create a same-volume hard link without replacing a path."""
    os.link(source, destination, follow_symlinks=False)


def _rename_no_overwrite(source: Path, destination: Path) -> None:
    """Move a regular same-volume file without ever replacing ``destination``."""
    _link_no_overwrite(source, destination)
    try:
        source.unlink()
    except OSError as unlink_error:
        try:
            destination.unlink()
        except OSError as cleanup_error:
            raise OSError(
                f"could not unlink source {source} after linking {destination}; "
                f"destination cleanup also failed: {cleanup_error}"
            ) from unlink_error
        raise OSError(
            f"could not unlink source {source} after linking {destination}; "
            "the destination link was removed"
        ) from unlink_error


def _rollback_renames(states: list[_RenameState]) -> list[str]:
    failures: list[str] = []
    for state in reversed(states):
        try:
            if state.location == "source":
                continue
            if state.location == "linked":
                if state.target.exists() or state.target.is_symlink():
                    try:
                        same_file = os.path.samefile(state.staging, state.target)
                    except OSError:
                        same_file = False
                    if same_file:
                        state.target.unlink()
                    else:
                        failures.append(
                            f"rollback left unexpected target {state.target.name!r} in place"
                        )
                _rename_no_overwrite(state.staging, state.source)
            elif state.location == "staging":
                _rename_no_overwrite(state.staging, state.source)
            else:
                _rename_no_overwrite(state.target, state.source)
            state.location = "source"
        except OSError as error:
            failures.append(f"rollback could not restore {state.source.name!r}: {error}")
    return failures


def _write_manifest_temp(root: Path, payload: bytes) -> Path:
    descriptor, temp_name = tempfile.mkstemp(prefix=".ocrbench-manifest-", suffix=".tmp", dir=root)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def build_manifest(root: Path, assignments_csv: Path) -> Manifest:
    """Build an anonymous manifest after complete validation and safe moves.

    Every source is re-resolved and rehashed after preflight and before the first
    rename. Destination creation uses atomic hard-link creation and never
    replaces an existing path on Windows or POSIX. Anonymous targets are checked
    again before the manifest is atomically published. Ordinary I/O or integrity
    failures trigger a best-effort rollback to the original source names.
    """
    records, violations, anchor = _preflight_build(root, assignments_csv)
    if violations:
        raise DatasetError(violations)
    if anchor is None:
        raise DatasetError("dataset root was not resolved during preflight")

    source_violations = _revalidate_sources(root, anchor, records)
    if source_violations:
        raise DatasetError(source_violations)

    entries: list[DocEntry] = []
    for record in records:
        incomplete = _complete_record_violation(record)
        if incomplete is not None:
            raise DatasetError(incomplete)
        assert record.sha256 is not None
        assert record.doc_id is not None
        assert record.target is not None
        entries.append(
            DocEntry(
                doc_id=record.doc_id,
                source_kind=cast(SourceKind, record.source_kind),
                file=record.target.name,
                sha256=record.sha256,
                split=cast(Split, record.split),
            )
        )
    entries.sort(key=lambda entry: entry.doc_id)
    manifest = Manifest(schema_version=1, docs=entries)
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    try:
        temp_manifest = _write_manifest_temp(root, payload)
    except OSError as error:
        raise DatasetError(f"cannot prepare atomic manifest write: {error}") from error

    states: list[_RenameState] = []
    try:
        for record in records:
            assert record.source is not None
            assert record.target is not None
            if record.source.name == record.target.name:
                continue
            state = _RenameState(
                source=record.source,
                staging=record.source.parent / f".ocrbench-stage-{uuid.uuid4().hex}",
                target=record.target,
            )
            states.append(state)
            _rename_no_overwrite(state.source, state.staging)
            state.location = "staging"

        for state in states:
            _link_no_overwrite(state.staging, state.target)
            state.location = "linked"

        target_violations = _revalidate_targets(root, anchor, records)
        if target_violations:
            raise _MutationIntegrityError(target_violations)

        for state in states:
            state.staging.unlink()
            state.location = "target"

        final_violations = _revalidate_targets(root, anchor, records)
        if final_violations:
            raise _MutationIntegrityError(final_violations)

        os.replace(temp_manifest, root / "manifest.json")
    except (OSError, _MutationIntegrityError) as error:
        rollback_failures = _rollback_renames(states)
        try:
            temp_manifest.unlink(missing_ok=True)
        except OSError as cleanup_error:
            rollback_failures.append(f"could not remove temporary manifest: {cleanup_error}")
        if isinstance(error, _MutationIntegrityError):
            primary = [
                "post-rename integrity validation failed; rollback was attempted",
                *error.violations,
            ]
        else:
            primary = [f"dataset mutation failed and was rolled back: {error}"]
        raise DatasetError([*primary, *rollback_failures]) from error

    return manifest
