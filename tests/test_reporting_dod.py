"""Additional acceptance tests for the T14 reporting deliverables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ocrbench.reporting as reporting_module
from ocrbench.dataset import Manifest, dataset_fingerprint
from ocrbench.metrics import aggregate
from ocrbench.reporting import (
    export_anonymous_summary,
    render_detail_report,
    render_report,
    write_detail_report,
)
from ocrbench.results import RunResults
from ocrbench.schema import OrderDocument
from ocrbench.splitguard import DetailReportAccessError
from tests.test_reporting import _document, _result, _run, _write_run


def _two_document_run() -> RunResults:
    first = _result()
    second = _result(doc_id="doc-1123456789ab", source_kind="photo")
    return _run().model_copy(update={"documents": [first, second]})


def _detail_dataset(root: Path) -> tuple[Path, RunResults]:
    raw_directory = root / "raw"
    gt_directory = root / "gt"
    raw_directory.mkdir(parents=True)
    gt_directory.mkdir()
    raw_content = b"synthetic detail-report image"
    sha256 = hashlib.sha256(raw_content).hexdigest()
    doc_id = f"doc-{sha256[:12]}"
    raw_name = f"{doc_id}.png"
    (raw_directory / raw_name).write_bytes(raw_content)
    (gt_directory / f"{doc_id}.json").write_text(
        _document().model_dump_json(),
        encoding="utf-8",
    )
    manifest = Manifest.model_validate(
        {
            "schema_version": 1,
            "docs": [
                {
                    "doc_id": doc_id,
                    "source_kind": "scan",
                    "file": raw_name,
                    "sha256": sha256,
                    "split": "dev",
                }
            ],
        }
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    fingerprint = dataset_fingerprint(root, manifest)
    base_run = _run()
    run_manifest = base_run.manifest.model_copy(update={"dataset_fingerprint": fingerprint})
    run = base_run.model_copy(
        update={"manifest": run_manifest, "documents": [_result(doc_id=doc_id)]}
    )
    return root, run


def test_render_report_matches_the_public_golden_fixture() -> None:
    run = _run()
    fixture = Path(__file__).parent / "fixtures" / "public" / "golden_report.md"

    assert render_report(run, aggregate(run.documents)) == fixture.read_text(encoding="utf-8")


def test_render_report_is_repeatable_and_document_order_independent() -> None:
    run = _two_document_run()
    reversed_run = run.model_copy(update={"documents": list(reversed(run.documents))})

    first = render_report(run, aggregate(run.documents))
    second = render_report(run, aggregate(run.documents))
    reversed_report = render_report(reversed_run, aggregate(reversed_run.documents))

    assert first == second == reversed_report


def test_anonymous_export_is_document_order_independent() -> None:
    run = _two_document_run()
    reversed_run = run.model_copy(update={"documents": list(reversed(run.documents))})

    first = export_anonymous_summary(run, aggregate(run.documents))
    reversed_payload = export_anonymous_summary(reversed_run, aggregate(reversed_run.documents))

    assert first == reversed_payload


def test_render_report_counts_documents_above_the_30_second_limit() -> None:
    result = _result().model_copy(
        update={
            "timings_ms": {
                "preprocess": 1.0,
                "load": 2.0,
                "infer": 3.0,
                "parse_validate": 4.0,
                "total": 30_001.0,
            }
        }
    )
    run = _run().model_copy(update={"documents": [result]})

    report = render_report(run, aggregate(run.documents))

    assert "Documents above 30,000 ms | 1/1" in report


def test_canaries_are_absent_from_anonymous_export_and_present_in_detail_report() -> None:
    prediction = _document("PREDICTION_CANARY")
    result = _result(prediction=prediction).model_copy(update={"raw_outputs": ["RAW_CANARY"]})
    run = _run().model_copy(update={"documents": [result]})
    ground_truth = _document("GROUND_TRUTH_CANARY")

    anonymous = json.dumps(export_anonymous_summary(run, aggregate(run.documents)))
    detail = render_detail_report(run, {result.doc_id: ground_truth})

    for canary in ("GROUND_TRUTH_CANARY", "PREDICTION_CANARY", "RAW_CANARY"):
        assert canary not in anonymous
        assert canary in detail


def test_write_detail_report_loads_a_local_run_and_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(tmp_path))
    data_root, run = _detail_dataset(tmp_path / "data")
    run_dir = _write_run(tmp_path, run)

    destination = write_detail_report(run_dir, data_root)

    assert destination == run_dir / "report_detail.md"
    assert "Synthetic Customer" in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize("split", ["selection", "final"])
def test_write_detail_report_rejects_blind_splits_before_gt_access_or_write(
    split: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs_root))
    base_run = _run()
    restricted_manifest = base_run.manifest.model_copy(update={"split": split})
    restricted_run = base_run.model_copy(update={"manifest": restricted_manifest})
    run_dir = _write_run(runs_root, restricted_run)
    data_root = tmp_path / "must-not-be-read"

    def fail_gt_access(
        results: RunResults,
        requested_data_root: Path,
    ) -> dict[str, OrderDocument]:
        raise AssertionError(
            f"GT access must not occur for {results.manifest.split}: {requested_data_root}"
        )

    def fail_write(path: Path, content: str) -> None:
        raise AssertionError(f"detail output must not be written: {path}: {content}")

    monkeypatch.setattr(reporting_module, "_ground_truth_for_run", fail_gt_access)
    monkeypatch.setattr(reporting_module, "_atomic_write_text", fail_write)

    with pytest.raises(DetailReportAccessError, match="Development"):
        write_detail_report(run_dir, data_root)

    assert not data_root.exists()
    assert not (run_dir / "report_detail.md").exists()


def test_write_detail_report_rejects_a_run_outside_the_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_root = tmp_path / "configured-runs"
    configured_root.mkdir()
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(configured_root))
    other_root = tmp_path / "other-runs"
    other_root.mkdir()
    data_root, run = _detail_dataset(tmp_path / "data")
    run_dir = _write_run(other_root, run)

    with pytest.raises(ValueError, match="OCRBENCH_RUNS_DIR"):
        write_detail_report(run_dir, data_root)

    assert not (run_dir / "report_detail.md").exists()


def test_write_detail_report_rejects_changed_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs_root))
    data_root, run = _detail_dataset(tmp_path / "data")
    run_dir = _write_run(runs_root, run)
    doc_id = run.documents[0].doc_id
    changed = _document("CHANGED_GROUND_TRUTH_CANARY")
    (data_root / "gt" / f"{doc_id}.json").write_text(
        changed.model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        write_detail_report(run_dir, data_root)

    assert not (run_dir / "report_detail.md").exists()


def test_detail_report_lists_error_tags_and_uses_a_safe_raw_fence() -> None:
    result = _result().model_copy(
        update={
            "score": {
                "exact_match": False,
                "header_correct": {
                    "customer_name": False,
                    "order_no": False,
                    "delivery_date": False,
                },
                "items_exact": False,
                "item_field_counts": {
                    "part_no": [0, 1],
                    "material": [0, 1],
                    "num_pieces": [0, 1],
                },
                "missing_items": 1,
                "extra_items": 0,
                "error_tags": ["missing_item", "not_json"],
            },
            "raw_outputs": ["RAW_CANARY\n````\n"],
        }
    )
    run = _run().model_copy(update={"documents": [result]})

    detail = render_detail_report(run, {result.doc_id: _document()})

    assert "- Error tags: missing_item, not_json" in detail
    assert "`````text\nRAW_CANARY\n````\n\n`````" in detail
