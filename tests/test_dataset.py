"""Tests for secure dataset manifests and deterministic fingerprints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

import ocrbench.dataset as dataset_module
from ocrbench.dataset import (
    DatasetError,
    DocEntry,
    Manifest,
    build_manifest,
    dataset_fingerprint,
    load_manifest,
    validate_split_counts,
)
from tests.helpers import make_mini_dataset, make_order_document

_SHA_A = "a" * 64
_DOC_A = "doc-" + _SHA_A[:12]


def _write_csv(
    path: Path,
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = ("file", "source_kind", "split"),
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_ground_truth(root: Path, manifest: Manifest) -> None:
    (root / "gt").mkdir(exist_ok=True)
    for index, entry in enumerate(manifest.docs):
        (root / "gt" / f"{entry.doc_id}.json").write_text(
            json.dumps(
                make_order_document(index).model_dump(mode="json"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _build_source_root(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "dataset"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    for name, content in files.items():
        (raw_dir / name).write_bytes(content)
    return root


def _entry(index: int, split: str, source_kind: str) -> DocEntry:
    sha256 = f"{index:064x}"
    doc_id = f"doc-{sha256[:12]}"
    return DocEntry.model_validate(
        {
            "doc_id": doc_id,
            "source_kind": source_kind,
            "file": f"{doc_id}.png",
            "sha256": sha256,
            "split": split,
        }
    )


def _full_count_manifest() -> Manifest:
    docs: list[DocEntry] = []
    index = 1
    for split, per_kind in (("dev", 20), ("selection", 10), ("final", 20)):
        for source_kind in ("scan", "photo"):
            for _ in range(per_kind):
                docs.append(_entry(index, split, source_kind))
                index += 1
    return Manifest(schema_version=1, docs=docs)


def test_manifest_models_are_strict_and_forbid_extra_fields() -> None:
    valid_entry = {
        "doc_id": _DOC_A,
        "source_kind": "scan",
        "file": f"{_DOC_A}.png",
        "sha256": _SHA_A,
        "split": "dev",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DocEntry.model_validate({**valid_entry, "customer_name": "must not leak"})
    with pytest.raises(ValidationError, match="Input should be 'scan' or 'photo'"):
        DocEntry.model_validate({**valid_entry, "source_kind": 1})
    with pytest.raises(ValidationError, match="schema_version"):
        Manifest.model_validate({"schema_version": "1", "docs": []})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Manifest.model_validate({"schema_version": 1, "docs": [], "extra": True})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("doc_id", "customer-a", "doc_id must match"),
        ("sha256", "A" * 64, "lowercase hex"),
        ("file", "../outside.png", "single-component"),
        ("file", "/absolute.png", "single-component"),
        ("file", r"C:\private\scan.png", "single-component"),
        ("file", r"folder\scan.png", "single-component"),
        ("file", "nested/scan.png", "single-component"),
        ("file", f"{_DOC_A}.gif", "extension"),
        ("file", f"{_DOC_A}-other.png", "file stem must equal doc_id"),
    ],
)
def test_doc_entry_rejects_nonanonymous_or_unsafe_values(
    field: str, value: str, message: str
) -> None:
    data = {
        "doc_id": _DOC_A,
        "source_kind": "scan",
        "file": f"{_DOC_A}.png",
        "sha256": _SHA_A,
        "split": "dev",
    }
    data[field] = value
    with pytest.raises(ValidationError, match=message):
        DocEntry.model_validate(data)


def test_doc_entry_requires_doc_id_to_match_declared_hash() -> None:
    with pytest.raises(ValidationError, match="for the declared sha256"):
        DocEntry(
            doc_id=_DOC_A,
            source_kind="photo",
            file=f"{_DOC_A}.jpg",
            sha256="b" * 64,
            split="selection",
        )


def test_load_manifest_accepts_generated_mini_dataset(tmp_path: Path) -> None:
    root = tmp_path / "mini"
    expected = make_mini_dataset(root)

    assert load_manifest(root) == expected


def test_load_manifest_aggregates_all_readable_dataset_violations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mini"
    manifest = make_mini_dataset(root)
    first, second, third = manifest.docs[:3]
    (root / "raw" / first.file).unlink()
    (root / "raw" / second.file).write_bytes(b"changed")
    (root / "gt" / f"{second.doc_id}.json").write_text(
        '{"customer_name": "missing every other required field"}',
        encoding="utf-8",
    )
    (root / "gt" / f"{third.doc_id}.json").unlink()

    payload = manifest.model_dump(mode="json")
    payload["docs"].append(first.model_dump(mode="json"))
    (root / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError) as captured:
        load_manifest(root)

    violations = captured.value.violations
    duplicate_index = next(
        index for index, item in enumerate(violations) if "duplicate doc_id" in item
    )
    missing_raw_index = next(
        index for index, item in enumerate(violations) if f"raw/{first.file} is missing" in item
    )
    mismatch_index = next(
        index for index, item in enumerate(violations) if "sha256 mismatch" in item
    )
    invalid_gt_index = next(
        index
        for index, item in enumerate(violations)
        if f"gt/{second.doc_id}.json is invalid" in item
    )
    missing_gt_index = next(
        index
        for index, item in enumerate(violations)
        if f"gt/{third.doc_id}.json is missing" in item
    )
    assert duplicate_index < missing_raw_index
    assert duplicate_index < mismatch_index
    assert mismatch_index < invalid_gt_index
    assert mismatch_index < missing_gt_index


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "[]",
        '{"schema_version": 1}',
        '{"schema_version": "1", "docs": {}}',
        '{"schema_version": 1, "docs": [], "unknown": true}',
        '{"schema_version": 1, "schema_version": 1, "docs": []}',
    ],
)
def test_load_manifest_rejects_malformed_manifest(tmp_path: Path, payload: str) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(payload, encoding="utf-8")

    with pytest.raises(DatasetError):
        load_manifest(root)


def test_load_manifest_rejects_raw_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "mini"
    manifest = make_mini_dataset(root)
    entry = manifest.docs[0]
    raw_path = root / "raw" / entry.file
    outside = tmp_path / "outside.png"
    outside.write_bytes(raw_path.read_bytes())
    raw_path.unlink()
    try:
        os.symlink(outside, raw_path)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DatasetError, match=r"symlink or reparse alias"):
        load_manifest(root)


def test_fingerprint_is_order_independent_and_byte_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "mini"
    manifest = make_mini_dataset(root)
    reordered = Manifest(schema_version=1, docs=list(reversed(manifest.docs)))

    baseline = dataset_fingerprint(root, manifest)
    assert dataset_fingerprint(root, manifest) == baseline
    assert dataset_fingerprint(root, reordered) == baseline

    raw_path = root / "raw" / manifest.docs[0].file
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b"\0")
    assert dataset_fingerprint(root, manifest) != baseline

    raw_path.write_bytes(original)
    gt_path = root / "gt" / f"{manifest.docs[0].doc_id}.json"
    gt_path.write_bytes(gt_path.read_bytes() + b" ")
    assert dataset_fingerprint(root, manifest) != baseline


def test_dataset_hashing_does_not_use_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mini"
    manifest = make_mini_dataset(root)

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"non-streaming read attempted for {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    assert load_manifest(root) == manifest
    assert len(dataset_fingerprint(root, manifest)) == 64


def test_validate_split_counts_matches_exact_adr_table() -> None:
    assert validate_split_counts(_full_count_manifest(), strict=False) == []
    assert validate_split_counts(_full_count_manifest(), strict=True) == []


def test_validate_split_counts_returns_or_raises_every_deviation(
    tmp_path: Path,
) -> None:
    manifest = make_mini_dataset(tmp_path / "mini")

    warnings = validate_split_counts(manifest, strict=False)
    assert len(warnings) == 9
    assert warnings[0] == "split 'dev': expected 40 documents, found 4"
    with pytest.raises(DatasetError) as captured:
        validate_split_counts(manifest, strict=True)
    assert list(captured.value.violations) == warnings


def test_build_manifest_round_trip_anonymizes_names(tmp_path: Path) -> None:
    root = _build_source_root(
        tmp_path,
        {
            "Customer Alpha.PNG": b"synthetic scan bytes",
            "order-007.pdf": b"%PDF-1.4\nsynthetic\n%%EOF\n",
        },
    )
    assignments = tmp_path / "assignments.csv"
    _write_csv(
        assignments,
        [
            ("Customer Alpha.PNG", "scan", "dev"),
            ("order-007.pdf", "photo", "selection"),
        ],
    )

    manifest = build_manifest(root, assignments)
    assert [entry.doc_id for entry in manifest.docs] == sorted(
        entry.doc_id for entry in manifest.docs
    )
    assert all(entry.file.startswith("doc-") for entry in manifest.docs)
    assert not (root / "raw" / "Customer Alpha.PNG").exists()
    assert not (root / "raw" / "order-007.pdf").exists()
    persisted = (root / "manifest.json").read_text(encoding="utf-8")
    assert "Customer Alpha" not in persisted
    assert "order-007" not in persisted

    _write_ground_truth(root, manifest)
    assert load_manifest(root) == manifest


def test_build_manifest_is_independent_of_csv_row_order(tmp_path: Path) -> None:
    files = {"first.png": b"first", "second.jpg": b"second"}
    first_root = _build_source_root(tmp_path / "one", files)
    second_root = _build_source_root(tmp_path / "two", files)
    first_csv = tmp_path / "one.csv"
    second_csv = tmp_path / "two.csv"
    rows = [
        ("first.png", "scan", "dev"),
        ("second.jpg", "photo", "selection"),
    ]
    _write_csv(first_csv, rows)
    _write_csv(second_csv, list(reversed(rows)))

    assert build_manifest(first_root, first_csv) == build_manifest(second_root, second_csv)
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json"
    ).read_bytes()


def test_build_manifest_rejects_duplicate_content_without_mutation(
    tmp_path: Path,
) -> None:
    root = _build_source_root(
        tmp_path,
        {"first.png": b"identical", "second.jpg": b"identical"},
    )
    assignments = tmp_path / "assignments.csv"
    _write_csv(
        assignments,
        [
            ("first.png", "scan", "dev"),
            ("second.jpg", "photo", "selection"),
        ],
    )

    with pytest.raises(DatasetError, match="duplicate source content"):
        build_manifest(root, assignments)

    assert (root / "raw" / "first.png").read_bytes() == b"identical"
    assert (root / "raw" / "second.jpg").read_bytes() == b"identical"
    assert not (root / "manifest.json").exists()


def test_build_manifest_aggregates_csv_preflight_and_never_partially_renames(
    tmp_path: Path,
) -> None:
    root = _build_source_root(tmp_path, {"source.png": b"source"})
    digest = hashlib.sha256(b"source").hexdigest()
    collision = root / "raw" / f"doc-{digest[:12]}.png"
    collision.write_bytes(b"existing target")
    assignments = tmp_path / "assignments.csv"
    _write_csv(
        assignments,
        [
            ("source.png", "scan", "dev"),
            ("source.png", "photo", "selection"),
            ("../escape.png", "camera", "training"),
            ("missing.bmp", "scan", "dev"),
        ],
    )

    with pytest.raises(DatasetError) as captured:
        build_manifest(root, assignments)

    message = str(captured.value)
    assert "duplicate CSV file assignment" in message
    assert "single-component" in message
    assert "source_kind must be" in message
    assert "split must be" in message
    assert "extension must be" in message
    assert "collides with existing" in message
    assert (root / "raw" / "source.png").read_bytes() == b"source"
    assert collision.read_bytes() == b"existing target"
    assert not (root / "manifest.json").exists()
    assert not list((root / "raw").glob(".ocrbench-stage-*"))


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (("file", "file", "split"), "duplicate columns"),
        (("file", "source_kind"), "missing columns"),
        (("file", "source_kind", "split", "comment"), "unsupported columns"),
    ],
)
def test_build_manifest_rejects_invalid_csv_columns_without_mutation(
    tmp_path: Path, header: tuple[str, ...], expected: str
) -> None:
    root = _build_source_root(tmp_path, {"source.png": b"source"})
    assignments = tmp_path / "assignments.csv"
    _write_csv(assignments, [("source.png", "scan", "dev")], header=header)

    with pytest.raises(DatasetError, match=expected):
        build_manifest(root, assignments)

    assert (root / "raw" / "source.png").exists()
    assert not (root / "manifest.json").exists()


def test_build_manifest_detects_sha_prefix_collision_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_source_root(tmp_path, {"first.png": b"first", "second.jpg": b"second"})
    assignments = tmp_path / "assignments.csv"
    _write_csv(
        assignments,
        [
            ("first.png", "scan", "dev"),
            ("second.jpg", "photo", "selection"),
        ],
    )

    def colliding_hash(path: Path) -> str:
        suffix = "1" * 52 if path.name == "first.png" else "2" * 52
        return "abcdef123456" + suffix

    monkeypatch.setattr(dataset_module, "_sha256_file", colliding_hash)
    with pytest.raises(DatasetError, match="SHA-256 prefix collision"):
        build_manifest(root, assignments)

    assert (root / "raw" / "first.png").exists()
    assert (root / "raw" / "second.jpg").exists()
    assert not (root / "manifest.json").exists()


def test_build_manifest_rolls_back_a_mid_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_source_root(tmp_path, {"first.png": b"first", "second.jpg": b"second"})
    assignments = tmp_path / "assignments.csv"
    _write_csv(
        assignments,
        [
            ("first.png", "scan", "dev"),
            ("second.jpg", "photo", "selection"),
        ],
    )
    real_rename: Callable[[Path, Path], None] = dataset_module._rename_no_overwrite
    staging_calls = 0

    def fail_second_staging_rename(source: Path, destination: Path) -> None:
        nonlocal staging_calls
        if destination.name.startswith(".ocrbench-stage-"):
            staging_calls += 1
            if staging_calls == 2:
                raise OSError("simulated rename failure")
        real_rename(source, destination)

    monkeypatch.setattr(
        dataset_module,
        "_rename_no_overwrite",
        fail_second_staging_rename,
    )
    with pytest.raises(DatasetError, match="was rolled back"):
        build_manifest(root, assignments)

    assert (root / "raw" / "first.png").read_bytes() == b"first"
    assert (root / "raw" / "second.jpg").read_bytes() == b"second"
    assert not (root / "manifest.json").exists()
    assert not list((root / "raw").glob(".ocrbench-stage-*"))
    assert not list(root.glob(".ocrbench-manifest-*.tmp"))


def test_build_manifest_rejects_source_symlink(tmp_path: Path) -> None:
    root = _build_source_root(tmp_path, {})
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    source = root / "raw" / "source.png"
    try:
        os.symlink(outside, source)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    assignments = tmp_path / "assignments.csv"
    _write_csv(assignments, [("source.png", "scan", "dev")])

    with pytest.raises(DatasetError, match="must not be a symlink"):
        build_manifest(root, assignments)

    assert source.is_symlink()
    assert not (root / "manifest.json").exists()


def test_dataset_error_has_stable_english_aggregation() -> None:
    error = DatasetError(["first violation", "second violation"])

    assert error.violations == ("first violation", "second violation")
    assert str(error) == ("Dataset validation failed:\n  - first violation\n  - second violation")
