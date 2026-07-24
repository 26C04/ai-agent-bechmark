"""Regression tests for dataset mutation races and filesystem aliases."""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import ocrbench.dataset as dataset_module
from ocrbench.config import OcrBenchError
from ocrbench.dataset import DatasetError, build_manifest, dataset_fingerprint, load_manifest
from tests.helpers import make_mini_dataset


def _write_assignments(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("file", "source_kind", "split"))
        writer.writerows(rows)


def _source_root(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "dataset"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    for name, content in files.items():
        (raw_dir / name).write_bytes(content)
    return root


def test_dataset_error_uses_the_shared_expected_error_base() -> None:
    assert issubclass(DatasetError, OcrBenchError)


def test_atomic_move_does_not_overwrite_destination_injected_at_link_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"source")
    real_link = os.link

    def inject_destination(
        source_path: Path,
        destination_path: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        destination_path.write_bytes(b"racer")
        real_link(
            source_path,
            destination_path,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", inject_destination)

    with pytest.raises(FileExistsError):
        dataset_module._rename_no_overwrite(source, destination)

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"racer"


def test_build_rolls_back_when_target_appears_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(tmp_path, {"source.png": b"source"})
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments, [("source.png", "scan", "dev")])
    real_link: Callable[[Path, Path], None] = dataset_module._link_no_overwrite

    def inject_target(source: Path, destination: Path) -> None:
        if destination.name.startswith("doc-"):
            destination.write_bytes(b"racer")
        real_link(source, destination)

    monkeypatch.setattr(dataset_module, "_link_no_overwrite", inject_target)

    with pytest.raises(DatasetError, match="was rolled back"):
        build_manifest(root, assignments)

    digest = hashlib.sha256(b"source").hexdigest()
    target = root / "raw" / f"doc-{digest[:12]}.png"
    assert (root / "raw" / "source.png").read_bytes() == b"source"
    assert target.read_bytes() == b"racer"
    assert not (root / "manifest.json").exists()
    assert not list((root / "raw").glob(".ocrbench-stage-*"))


def test_sources_changed_after_preflight_are_aggregated_before_any_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(
        tmp_path,
        {"content.png": b"content-before", "identity.jpg": b"identity-stable"},
    )
    assignments = tmp_path / "assignments.csv"
    _write_assignments(
        assignments,
        [
            ("content.png", "scan", "dev"),
            ("identity.jpg", "photo", "selection"),
        ],
    )
    real_preflight = dataset_module._preflight_build

    def tampering_preflight(
        root_path: Path,
        csv_path: Path,
    ) -> tuple[
        list[dataset_module._BuildRecord],
        list[str],
        Path | None,
    ]:
        result = real_preflight(root_path, csv_path)
        (root_path / "raw" / "content.png").write_bytes(b"content-after")
        identity_path = root_path / "raw" / "identity.jpg"
        unchanged_bytes = identity_path.read_bytes()
        replacement_path = root_path / "raw" / "identity-replacement.jpg"
        replacement_path.write_bytes(unchanged_bytes)
        os.replace(replacement_path, identity_path)
        return result

    move_calls = 0
    real_move: Callable[[Path, Path], None] = dataset_module._rename_no_overwrite

    def track_move(source: Path, destination: Path) -> None:
        nonlocal move_calls
        move_calls += 1
        real_move(source, destination)

    monkeypatch.setattr(dataset_module, "_preflight_build", tampering_preflight)
    monkeypatch.setattr(dataset_module, "_rename_no_overwrite", track_move)

    with pytest.raises(DatasetError) as captured:
        build_manifest(root, assignments)

    assert "changed content after preflight" in str(captured.value)
    assert "changed file identity after preflight" in str(captured.value)
    assert move_calls == 0
    assert (root / "raw" / "content.png").read_bytes() == b"content-after"
    assert (root / "raw" / "identity.jpg").read_bytes() == b"identity-stable"
    assert not (root / "manifest.json").exists()
    assert not list((root / "raw").glob(".ocrbench-stage-*"))
    assert not list(root.glob(".ocrbench-manifest-*.tmp"))


def test_target_replacement_is_detected_and_original_source_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(tmp_path, {"source.png": b"source"})
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments, [("source.png", "scan", "dev")])
    real_revalidate = dataset_module._revalidate_targets
    validation_calls = 0

    def replace_before_validation(
        root_path: Path,
        anchor: Path,
        records: list[dataset_module._BuildRecord],
    ) -> list[str]:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            target = records[0].target
            assert target is not None
            target.unlink()
            target.write_bytes(b"racer")
        return real_revalidate(root_path, anchor, records)

    monkeypatch.setattr(
        dataset_module,
        "_revalidate_targets",
        replace_before_validation,
    )

    with pytest.raises(DatasetError, match="post-rename integrity"):
        build_manifest(root, assignments)

    digest = hashlib.sha256(b"source").hexdigest()
    target = root / "raw" / f"doc-{digest[:12]}.png"
    assert (root / "raw" / "source.png").read_bytes() == b"source"
    assert target.read_bytes() == b"racer"
    assert not (root / "manifest.json").exists()
    assert not list((root / "raw").glob(".ocrbench-stage-*"))


@pytest.mark.parametrize("member_kind", ["raw", "gt"])
def test_load_and_fingerprint_reject_internal_member_symlinks(
    tmp_path: Path, member_kind: str
) -> None:
    root = tmp_path / member_kind
    manifest = make_mini_dataset(root)
    entry = manifest.docs[0]
    if member_kind == "raw":
        member = root / "raw" / entry.file
        alias_target = root / "raw" / "internal-target.bin"
    else:
        member = root / "gt" / f"{entry.doc_id}.json"
        alias_target = root / "gt" / "internal-target.json"
    alias_target.write_bytes(member.read_bytes())
    member.unlink()
    try:
        os.symlink(alias_target, member)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DatasetError, match="symlink or reparse alias"):
        load_manifest(root)
    with pytest.raises(DatasetError, match="symlink or reparse alias"):
        dataset_fingerprint(root, manifest)


def test_casefold_target_sibling_is_self_only_when_file_identity_matches(
    tmp_path: Path,
) -> None:
    content = b"source"
    digest = hashlib.sha256(content).hexdigest()
    target_name = f"doc-{digest[:12]}.png"
    source_name = target_name.upper()
    root = _source_root(tmp_path, {source_name: content})
    sibling = root / "raw" / target_name
    try:
        with sibling.open("xb") as stream:
            stream.write(b"case-only sibling")
    except FileExistsError:
        pytest.skip("the test filesystem is case-insensitive")

    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments, [(source_name, "scan", "dev")])

    with pytest.raises(DatasetError, match="collides with existing"):
        build_manifest(root, assignments)

    assert (root / "raw" / source_name).read_bytes() == content
    assert sibling.read_bytes() == b"case-only sibling"
    assert not (root / "manifest.json").exists()


def test_load_manifest_rejects_logical_manifest_symlink(tmp_path: Path) -> None:
    root = tmp_path / "manifest-alias"
    make_mini_dataset(root)
    manifest_path = root / "manifest.json"
    alias_target = root / "manifest-target.json"
    alias_target.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        os.symlink(alias_target, manifest_path)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DatasetError, match=r"manifest\.json must not be"):
        load_manifest(root)


def test_load_manifest_rejects_nonregular_logical_manifest(tmp_path: Path) -> None:
    root = tmp_path / "manifest-directory"
    make_mini_dataset(root)
    manifest_path = root / "manifest.json"
    manifest_path.unlink()
    manifest_path.mkdir()

    with pytest.raises(
        DatasetError,
        match=r"manifest\.json is not a regular file",
    ):
        load_manifest(root)


def test_casefold_hardlink_sibling_is_not_the_source_directory_entry(
    tmp_path: Path,
) -> None:
    content = b"source"
    digest = hashlib.sha256(content).hexdigest()
    target_name = f"doc-{digest[:12]}.png"
    source_name = target_name.upper()
    root = _source_root(tmp_path, {source_name: content})
    sibling = root / "raw" / target_name
    try:
        os.link(root / "raw" / source_name, sibling)
    except OSError as error:
        pytest.skip(f"case-sensitive hard links are unavailable: {error}")

    assignments = tmp_path / "hardlink-assignments.csv"
    _write_assignments(assignments, [(source_name, "scan", "dev")])

    with pytest.raises(DatasetError, match="collides with existing"):
        build_manifest(root, assignments)

    assert (root / "raw" / source_name).read_bytes() == content
    assert sibling.read_bytes() == content
    assert not (root / "manifest.json").exists()
