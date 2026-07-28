"""Tests for strict, safe, and metrics-compatible run result persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ocrbench.metrics import aggregate
from ocrbench.parsing import ParseStatus
from ocrbench.results import (
    DocumentResult,
    ResultsError,
    RunManifest,
    RunResults,
    load_run,
    serialize_score,
)
from ocrbench.schema import LineItem, OrderDocument
from ocrbench.scoring import DocumentScore


def _document() -> OrderDocument:
    return OrderDocument(
        customer_name="Synthetic Customer",
        order_no="ORDER-0001",
        delivery_date="2026-08-01",
        items=[
            LineItem(
                part_no="PART-0001",
                material="Synthetic Material",
                num_pieces=1,
            )
        ],
    )


def _score() -> DocumentScore:
    return DocumentScore(
        exact_match=True,
        header_correct={
            "customer_name": True,
            "order_no": True,
            "delivery_date": True,
        },
        items_exact=True,
        item_field_counts={
            "part_no": (1, 1),
            "material": (1, 1),
            "num_pieces": (1, 1),
        },
        missing_items=0,
        extra_items=0,
        error_tags=frozenset(),
    )


def _failure_score(status: ParseStatus = ParseStatus.NOT_JSON) -> DocumentScore:
    return DocumentScore(
        exact_match=False,
        header_correct={
            "customer_name": False,
            "order_no": False,
            "delivery_date": False,
        },
        items_exact=False,
        item_field_counts={
            "part_no": (0, 1),
            "material": (0, 1),
            "num_pieces": (0, 1),
        },
        missing_items=1,
        extra_items=0,
        error_tags=frozenset({"missing_item", status.value}),
    )


def _raw(document: OrderDocument | None = None) -> str:
    selected = document if document is not None else _document()
    return selected.model_dump_json()


def _result(**changes: object) -> DocumentResult:
    data: dict[str, object] = {
        "doc_id": "doc-0123456789ab",
        "source_kind": "scan",
        "first_attempt_status": ParseStatus.OK,
        "final_status": ParseStatus.OK,
        "attempt_count": 1,
        "prediction": _document().model_dump(mode="json"),
        "raw_outputs": [_raw()],
        "score": serialize_score(_score()),
        "needs_review": False,
        "warm": True,
        "timings_ms": {
            "preprocess": 1.0,
            "load": 2.0,
            "infer": 3.0,
            "parse_validate": 4.0,
            "total": 10.0,
        },
        "tokens": {"prompt": 10, "output": 5},
        "tokens_per_sec": 2.5,
        "error": None,
    }
    data.update(changes)
    return DocumentResult.model_validate(data)


def _adapter_failure_result() -> DocumentResult:
    return _result(
        first_attempt_status=ParseStatus.NOT_JSON,
        final_status=ParseStatus.NOT_JSON,
        prediction=None,
        raw_outputs=[],
        score=serialize_score(_failure_score()),
        needs_review=True,
        warm=False,
        tokens={"prompt": None, "output": None},
        tokens_per_sec=None,
        error="Adapter inference failed",
    )


def _manifest(**changes: object) -> RunManifest:
    data: dict[str, object] = {
        "run_id": "unit-test",
        "started_at_utc": "20260724T010203Z",
        "model_tag": "gemma4:12b",
        "model_digest": "sha256:synthetic",
        "prompt_name": "base",
        "prompt_hash": "abcdef012345",
        "split": "dev",
        "dataset_fingerprint": "1" * 64,
        "preprocess_version": "v1",
        "ollama_version": "fake-ollama",
        "seed": 20260721,
        "temperature": 0.0,
        "gpu_fully_loaded": True,
        "participation": "primary",
        "os_info": "Synthetic OS",
        "gpu_info": None,
        "adapter_kind": "fake",
    }
    data.update(changes)
    return RunManifest.model_validate(data)


def _run(**changes: object) -> RunResults:
    data: dict[str, object] = {
        "schema_version": 1,
        "manifest": _manifest().model_dump(mode="json"),
        "documents": [_result().model_dump(mode="json")],
    }
    data.update(changes)
    return RunResults.model_validate_json(json.dumps(data))


def _write_run(root: Path, run: RunResults | None = None) -> tuple[Path, RunResults]:
    selected = run if run is not None else _run()
    manifest = selected.manifest
    directory = root / f"{manifest.started_at_utc}_{manifest.run_id}"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "results.json").write_text(
        selected.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return directory, selected


def test_score_serialization_is_json_safe_lossless_and_aggregate_compatible() -> None:
    result = _result()

    encoded = json.dumps(result.score, allow_nan=False)

    assert "frozenset" not in encoded
    assert result.parsed_score() == _score()
    summary = aggregate([result])
    assert summary.exact_match_count == 1
    assert summary.schema_valid_rate == 1.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attempt_count": "1"}, "integer"),
        ({"source_kind": "fax"}, "scan"),
        ({"tokens_per_sec": float("nan")}, "finite"),
        ({"timings_ms": {"preprocess": 1.0}}, "five required"),
        ({"tokens": {"prompt": 1, "output": 2, "other": 3}}, "exactly"),
        ({"unexpected": True}, "Extra inputs"),
    ],
)
def test_document_result_is_strict_and_rejects_nonfinite_or_extra_data(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _result(**changes)


def test_score_payload_requires_exact_keys_counts_totals_and_sorted_tags() -> None:
    payload = serialize_score(_score())
    payload["header_correct"]["unknown"] = True
    with pytest.raises(ValidationError, match="scoring contract"):
        _result(score=payload)

    payload = serialize_score(_score())
    payload["item_field_counts"]["part_no"] = [2, 1]
    with pytest.raises(ValidationError, match="must not exceed"):
        _result(score=payload)

    payload = serialize_score(_score())
    payload["item_field_counts"]["part_no"] = [1, 2]
    with pytest.raises(ValidationError, match="totals must agree"):
        _result(score=payload)

    payload = serialize_score(_score())
    payload["error_tags"] = ["z", "a", "z"]
    with pytest.raises(ValidationError, match="unique and sorted"):
        _result(score=payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"first_attempt_status": ParseStatus.NOT_JSON},
        {"final_status": ParseStatus.NOT_JSON, "prediction": None},
        {"prediction": _document().model_copy(update={"order_no": "OTHER"}).model_dump()},
        {"raw_outputs": ["not json"]},
        {"needs_review": True},
    ],
)
def test_document_result_rejects_status_prediction_raw_and_review_tampering(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _result(**changes)


def test_adapter_failure_state_requires_exactly_one_missing_response() -> None:
    result = _adapter_failure_result()
    assert result.error is not None

    payload = result.model_dump(mode="python")
    payload["raw_outputs"] = ["invented"]
    with pytest.raises(ValidationError, match="missing response"):
        DocumentResult.model_validate(payload)


def test_parse_failure_cannot_carry_a_successful_score() -> None:
    payload = _adapter_failure_result().model_dump(mode="python")
    payload["score"] = serialize_score(_score())

    with pytest.raises(ValidationError, match="failure contract"):
        DocumentResult.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_name": "../unsafe"},
        {"run_id": "../unsafe"},
        {"started_at_utc": "20260230T010203Z"},
        {"dataset_fingerprint": "A" * 64},
        {"temperature": float("inf")},
        {"gpu_fully_loaded": False, "participation": "primary"},
        {"model_tag": "unapproved:latest"},
        {"extra": True},
    ],
)
def test_run_manifest_is_strict_and_self_consistent(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _manifest(**changes)


def test_run_results_requires_integer_schema_unique_sorted_documents() -> None:
    with pytest.raises(ValidationError, match="integer 1"):
        _run(schema_version="1")
    with pytest.raises(ValidationError, match="duplicate"):
        _run(documents=[_result().model_dump(mode="json")] * 2)
    second = _result(doc_id="doc-000000000000")
    with pytest.raises(ValidationError, match="sorted"):
        _run(
            documents=[
                _result().model_dump(mode="json"),
                second.model_dump(mode="json"),
            ]
        )


def test_load_run_round_trip(tmp_path: Path) -> None:
    directory, expected = _write_run(tmp_path)

    assert load_run(directory) == expected


@pytest.mark.parametrize("missing", ["manifest.json", "results.json"])
def test_load_run_rejects_incomplete_artifacts(tmp_path: Path, missing: str) -> None:
    directory, _ = _write_run(tmp_path)
    (directory / missing).unlink()

    with pytest.raises(ResultsError, match="incomplete"):
        load_run(directory)


def test_load_run_rejects_duplicate_keys_and_schema_mismatch(tmp_path: Path) -> None:
    directory, _ = _write_run(tmp_path)
    results_path = directory / "results.json"
    text = results_path.read_text(encoding="utf-8")
    results_path.write_text(
        text.replace('"schema_version": 1,', '"schema_version": 1,\\n  "schema_version": 1,', 1),
        encoding="utf-8",
    )
    with pytest.raises(ResultsError, match="strict UTF-8 JSON"):
        load_run(directory)

    directory, _ = _write_run(tmp_path / "other")
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResultsError, match="schema_version"):
        load_run(directory)


def test_load_run_rejects_cross_file_or_directory_identity_mismatch(
    tmp_path: Path,
) -> None:
    directory, _ = _write_run(tmp_path)
    manifest_payload: dict[str, Any] = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_payload["model_digest"] = "sha256:other"
    (directory / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    with pytest.raises(ResultsError, match="does not match"):
        load_run(directory)

    clean_directory, _ = _write_run(tmp_path / "clean")
    wrong_name = clean_directory.parent / "wrong-name"
    clean_directory.rename(wrong_name)
    with pytest.raises(ResultsError, match="directory name"):
        load_run(wrong_name)


def test_load_run_rejects_nonregular_and_alias_artifacts(tmp_path: Path) -> None:
    directory, _ = _write_run(tmp_path)
    results_path = directory / "results.json"
    results_path.unlink()
    results_path.mkdir()
    with pytest.raises(ResultsError, match="not a regular file"):
        load_run(directory)

    directory, _ = _write_run(tmp_path / "alias")
    manifest_path = directory / "manifest.json"
    target = directory / "manifest-target.json"
    target.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        os.symlink(target, manifest_path)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(ResultsError, match="symlink or reparse alias"):
        load_run(directory)
