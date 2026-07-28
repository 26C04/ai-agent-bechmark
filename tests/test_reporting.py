"""Contract tests for aggregate, anonymous, and local detailed reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import create_model

from ocrbench.metrics import aggregate
from ocrbench.parsing import ParseStatus
from ocrbench.reporting import (
    assert_no_forbidden_content,
    export_anonymous_summary,
    render_detail_report,
    render_report,
    write_anonymous_summary,
    write_report,
)
from ocrbench.results import DocumentResult, RunManifest, RunResults, serialize_score
from ocrbench.schema import LineItem, OrderDocument
from ocrbench.scoring import DocumentScore


def _document(customer_name: str = "Synthetic Customer") -> OrderDocument:
    return OrderDocument(
        customer_name=customer_name,
        order_no="ORDER-0001",
        delivery_date="2026-08-01",
        items=[LineItem(part_no="PART-0001", material="Synthetic Material", num_pieces=1)],
    )


def _score(exact: bool = True) -> DocumentScore:
    return DocumentScore(
        exact_match=exact,
        header_correct={
            "customer_name": exact,
            "order_no": exact,
            "delivery_date": exact,
        },
        items_exact=exact,
        item_field_counts={
            "part_no": (int(exact), 1),
            "material": (int(exact), 1),
            "num_pieces": (int(exact), 1),
        },
        missing_items=0 if exact else 1,
        extra_items=0,
        error_tags=frozenset() if exact else frozenset({"missing_item", "not_json"}),
    )


def _result(
    *,
    doc_id: str = "doc-0123456789ab",
    source_kind: str = "scan",
    warm: bool = True,
    prediction: OrderDocument | None = None,
) -> DocumentResult:
    selected = prediction if prediction is not None else _document()
    return DocumentResult.model_validate(
        {
            "doc_id": doc_id,
            "source_kind": source_kind,
            "first_attempt_status": ParseStatus.OK,
            "final_status": ParseStatus.OK,
            "attempt_count": 1,
            "prediction": selected.model_dump(mode="json"),
            "raw_outputs": [selected.model_dump_json()],
            "score": serialize_score(_score()),
            "needs_review": False,
            "warm": warm,
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
    )


def _run(*, warm: bool = True) -> RunResults:
    manifest = RunManifest(
        run_id="unit-test",
        started_at_utc="20260724T010203Z",
        model_tag="gemma4:12b",
        model_digest="sha256:synthetic",
        prompt_name="base",
        prompt_hash="abcdef012345",
        split="dev",
        dataset_fingerprint="1" * 64,
        preprocess_version="v1",
        ollama_version="fake-ollama",
        seed=20260721,
        temperature=0.0,
        gpu_fully_loaded=False,
        participation="reference",
        os_info="Synthetic OS",
        gpu_info=None,
        adapter_kind="fake",
    )
    return RunResults(schema_version=1, manifest=manifest, documents=[_result(warm=warm)])


def _write_run(root: Path, run: RunResults) -> Path:
    directory = root / f"{run.manifest.started_at_utc}_{run.manifest.run_id}"
    directory.mkdir()
    (directory / "manifest.json").write_text(run.manifest.model_dump_json(), encoding="utf-8")
    (directory / "results.json").write_text(run.model_dump_json(), encoding="utf-8")
    return directory


def test_render_report_is_aggregate_only_and_formats_metrics() -> None:
    run = _run()

    report = render_report(run, aggregate(run.documents))

    assert "# OCR Benchmark Report" in report
    assert "reference (not eligible for primary comparison)" in report
    assert "fake (test adapter)" in report
    assert "Warm p50 | 10 ms" in report
    assert "Documents above 30,000 ms | 0/1" in report
    assert "Mean prompt tokens/document | 10.0" in report
    assert "Mean output tokens/document | 5.0" in report
    assert "Wilson 95% CI" in report
    assert "Synthetic Customer" not in report
    assert "doc-0123456789ab" not in report


def test_render_report_uses_na_when_no_warm_latency_exists() -> None:
    run = _run(warm=False)

    report = render_report(run, aggregate(run.documents))

    assert "Warm p50 | N/A" in report
    assert "Warm p95 | N/A" in report


def test_write_report_writes_analysis_template_only_once(tmp_path: Path) -> None:
    directory = _write_run(tmp_path, _run())

    assert write_report(directory) == directory / "report.md"
    first_analysis = (directory / "analysis.md").read_text(encoding="utf-8")
    (directory / "analysis.md").write_text("preserve me\n", encoding="utf-8")
    write_report(directory)

    assert first_analysis == "\n".join(
        [
            "# Benchmark Analysis",
            "",
            "## Failure trends",
            "",
            "## Next improvements",
            "",
            "## Anonymous-summary input hash",
            "",
        ]
    )
    assert (directory / "analysis.md").read_text(encoding="utf-8") == "preserve me\n"


def test_anonymous_export_is_allowlisted_and_rejects_canary_leaks(tmp_path: Path) -> None:
    run = _run()
    summary = aggregate(run.documents)

    payload = export_anonymous_summary(run, summary)
    assert set(payload) == {"schema_version", "manifest", "summary", "documents"}
    assert set(payload["manifest"]) == set(type(run.manifest).model_fields)
    assert set(payload["documents"][0]) == {
        "doc_id",
        "source_kind",
        "exact_match",
        "final_status",
        "error_tags",
        "needs_review",
        "total_ms",
        "warm",
    }
    rendered = json.dumps(payload, ensure_ascii=False)
    assert "Synthetic Customer" not in rendered

    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_content({"leak": "Synthetic Customer"}, run)

    directory = _write_run(tmp_path, run)
    destination = tmp_path / "anonymous.json"
    assert write_anonymous_summary(directory, destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["manifest"]["run_id"] == "unit-test"


def test_detail_report_is_explicitly_local_and_contains_full_context() -> None:
    run = _run()
    ground_truth: dict[str, OrderDocument] = {"doc-0123456789ab": _document()}

    report = render_detail_report(run, ground_truth)

    assert "Do not commit it or send it to cloud services" in report
    assert "Synthetic Customer" in report
    assert "Raw model responses" in report
    with pytest.raises(ValueError, match="exactly"):
        render_detail_report(run, {})


def test_anonymous_scanner_detects_json_escaped_sensitive_values() -> None:
    canary = 'PREDICTION_"QUOTED"_CANARY'
    prediction = _document(canary)
    run = _run().model_copy(update={"documents": [_result(prediction=prediction)]})

    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_content({"leak": canary}, run)


def test_anonymous_export_ignores_empty_sensitive_strings() -> None:
    prediction = _document("")
    run = _run().model_copy(update={"documents": [_result(prediction=prediction)]})

    export_anonymous_summary(run, aggregate(run.documents))


def test_manifest_export_does_not_inherit_future_fields() -> None:
    extended_type = create_model(
        "ExtendedManifest",
        private_canary=(str, "PRIVATE_MANIFEST_CANARY"),
        __base__=RunManifest,
    )
    extended = extended_type.model_validate(_run().manifest.model_dump())
    run = _run().model_copy(update={"manifest": extended})

    payload = export_anonymous_summary(run, aggregate(run.documents))

    assert "private_canary" not in payload["manifest"]
    assert "PRIVATE_MANIFEST_CANARY" not in json.dumps(payload)
