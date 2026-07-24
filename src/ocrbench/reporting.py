"""Deterministic human and machine-readable reports for benchmark runs.

The public report deliberately contains aggregate data only.  Per-document
predictions and raw model responses are available exclusively through the
explicitly local detailed report API.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ocrbench import config
from ocrbench.dataset import DatasetError, dataset_fingerprint, load_manifest
from ocrbench.metrics import RunSummary, aggregate, item_field_bootstrap_ci
from ocrbench.results import RunManifest, RunResults, load_run
from ocrbench.schema import OrderDocument

_HEADER_FIELDS = ("customer_name", "order_no", "delivery_date")
_ITEM_FIELDS = ("part_no", "material", "num_pieces")
_TIMING_FIELDS = ("preprocess", "load", "infer", "parse_validate", "total")


def render_report(results: RunResults, summary: RunSummary) -> str:
    """Render the deterministic aggregate report without document-level data."""
    _require_matching_summary(results, summary)
    manifest = results.manifest
    slow_documents = sum(document.timings_ms["total"] > 30_000 for document in results.documents)
    lines = [
        "# OCR Benchmark Report",
        "",
        "## Run identity",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    identity = (
        ("Run ID", manifest.run_id),
        ("Started at (UTC)", manifest.started_at_utc),
        ("Model tag", manifest.model_tag),
        ("Model digest", manifest.model_digest),
        ("Prompt name", manifest.prompt_name),
        ("Prompt hash", manifest.prompt_hash),
        ("Split", manifest.split),
        ("Dataset fingerprint", manifest.dataset_fingerprint),
        ("Preprocess version", manifest.preprocess_version),
        ("Ollama version", manifest.ollama_version),
        ("Seed", str(manifest.seed)),
        ("Temperature", f"{manifest.temperature:g}"),
        ("Participation", _participation_label(manifest.participation)),
        ("GPU fully loaded", _yes_no_unknown(manifest.gpu_fully_loaded)),
        ("Operating system", manifest.os_info),
        ("GPU", manifest.gpu_info or "N/A"),
        ("Adapter kind", _adapter_kind_label(manifest.adapter_kind)),
    )
    lines.extend(_table_row(label, value) for label, value in identity)
    lines.extend(
        [
            "",
            "## Primary metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            _table_row(
                "Exact match",
                (
                    f"{_percent(summary.exact_match_rate)} "
                    f"({summary.exact_match_count}/{summary.n_docs}; Wilson 95% CI "
                    f"{_ci(summary.exact_match_ci)})"
                ),
            ),
            _table_row("Schema valid", _percent(summary.schema_valid_rate)),
            _table_row("First-attempt JSON valid", _percent(summary.first_attempt_json_rate)),
            _table_row(
                "Success after retry",
                _percent_or_na(summary.after_retry_success_rate),
            ),
            _table_row("Needs review", f"{summary.needs_review_count}/{summary.n_docs}"),
            "",
            "## Field accuracy",
            "",
            "| Field | Accuracy | 95% CI |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        _table_row(field, _percent(summary.header_field_accuracy[field]), "N/A")
        for field in _HEADER_FIELDS
    )
    lines.extend(
        _table_row(
            field,
            _percent(summary.item_field_accuracy[field]),
            _ci(item_field_bootstrap_ci(results.documents, field)),
        )
        for field in _ITEM_FIELDS
    )
    lines.extend(
        [
            "",
            "## Source-kind breakdown",
            "",
            "| Source kind | Documents | Exact match | Schema valid |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source_kind, source_summary in sorted(summary.by_source_kind.items()):
        lines.append(
            _table_row(
                source_kind,
                str(source_summary.n_docs),
                _percent(source_summary.exact_match_rate),
                _percent(source_summary.schema_valid_rate),
            )
        )
    lines.extend(
        [
            "",
            "## Error categories",
            "",
            "| Error tag | Documents |",
            "| --- | ---: |",
        ]
    )
    if summary.error_tag_counts:
        lines.extend(
            _table_row(tag, str(summary.error_tag_counts[tag]))
            for tag in sorted(summary.error_tag_counts)
        )
    else:
        lines.append(_table_row("None", "0"))
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            _table_row("Warm p50", _milliseconds_or_na(summary.latency_warm_p50_ms)),
            _table_row("Warm p95", _milliseconds_or_na(summary.latency_warm_p95_ms)),
            _table_row(
                "Documents above 30,000 ms",
                f"{slow_documents}/{summary.n_docs}",
            ),
        ]
    )
    lines.extend(
        _table_row(f"Mean {field}", _milliseconds(summary.mean_timings_ms[field]))
        for field in _TIMING_FIELDS
    )
    lines.extend(
        [
            "",
            "## Tokens",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            _table_row("Prompt tokens", str(summary.token_totals["prompt"])),
            _table_row("Output tokens", str(summary.token_totals["output"])),
            _table_row(
                "Mean prompt tokens/document",
                _number(summary.token_totals["prompt"] / summary.n_docs),
            ),
            _table_row(
                "Mean output tokens/document",
                _number(summary.token_totals["output"] / summary.n_docs),
            ),
            _table_row("Mean tokens/sec", _number_or_na(summary.mean_tokens_per_sec)),
            "",
            "This aggregate report intentionally excludes document IDs, predictions, raw model "
            "responses, and ground-truth values.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    """Load one run, write ``report.md``, and create ``analysis.md`` if absent."""
    directory = Path(run_dir)
    results = load_run(directory)
    summary = aggregate(results.documents)
    report_path = directory / "report.md"
    _atomic_write_text(report_path, render_report(results, summary))
    analysis_path = directory / "analysis.md"
    _atomic_create_text(analysis_path, _analysis_template())
    return report_path


def export_anonymous_summary(results: RunResults, summary: RunSummary) -> dict[str, Any]:
    """Export an allowlisted, collaboration-safe representation of a run.

    The payload includes execution identity, every aggregate metric, and only
    the prescribed document-level status fields.  It never includes model
    predictions, raw model responses, error text, or source/ground-truth paths.
    """
    _require_matching_summary(results, summary)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest": _manifest_payload(results.manifest),
        "summary": _summary_payload(summary),
        "documents": [
            {
                "doc_id": document.doc_id,
                "source_kind": document.source_kind,
                "exact_match": document.parsed_score().exact_match,
                "final_status": document.final_status.value,
                "error_tags": sorted(document.parsed_score().error_tags),
                "needs_review": document.needs_review,
                "total_ms": document.timings_ms["total"],
                "warm": document.warm,
            }
            for document in sorted(results.documents, key=lambda item: item.doc_id)
        ],
    }
    assert_no_forbidden_content(payload, results)
    return payload


def assert_no_forbidden_content(payload: Mapping[str, Any], results: RunResults) -> None:
    """Reject an export if model response or prediction text was leaked."""
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("anonymous export is not strict JSON") from error
    forbidden = _sensitive_strings(results)
    exported_strings = _string_leaves(payload)
    if any(sensitive in exported for sensitive in forbidden for exported in exported_strings):
        raise ValueError("anonymous export contains forbidden prediction or raw-output content")


def write_anonymous_summary(run_dir: Path, out_path: Path) -> Path:
    """Load one run and write its allowlisted anonymous summary as strict JSON."""
    results = load_run(Path(run_dir))
    payload = export_anonymous_summary(results, aggregate(results.documents))
    destination = Path(out_path)
    _atomic_write_text(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return destination


def render_detail_report(results: RunResults, gt_by_doc: Mapping[str, OrderDocument]) -> str:
    """Render the explicitly local report containing GT, predictions, and raw output."""
    expected_ids = {document.doc_id for document in results.documents}
    if set(gt_by_doc) != expected_ids:
        raise ValueError("ground truth must contain exactly the run document IDs")
    lines = [
        "# OCR Benchmark Detailed Report",
        "",
        "CONFIDENTIAL: This local-only report contains ground truth, predictions, and raw model "
        "responses. Do not commit it or send it to cloud services (ADR §5).",
        "",
    ]
    for document in sorted(results.documents, key=lambda item: item.doc_id):
        ground_truth = gt_by_doc[document.doc_id]
        prediction = document.prediction
        lines.extend(
            [
                f"## {document.doc_id}",
                "",
                f"- Source kind: {document.source_kind}",
                f"- Final status: {document.final_status.value}",
                f"- Exact match: {_percent(1.0 if document.parsed_score().exact_match else 0.0)}",
                f"- Needs review: {'yes' if document.needs_review else 'no'}",
                "- Error tags: "
                + (
                    ", ".join(sorted(document.parsed_score().error_tags))
                    if document.parsed_score().error_tags
                    else "none"
                ),
                "",
                "### Field differences",
                "",
                "| Field | Ground truth | Prediction | Match |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(_detail_rows(ground_truth, prediction))
        lines.extend(
            [
                "",
                "### Ground truth",
                "",
                "```json",
                _json_block(ground_truth.model_dump(mode="json")),
                "```",
                "",
                "### Prediction",
                "",
                "```json",
                _json_block(prediction),
                "```",
                "",
                "### Raw model responses",
                "",
            ]
        )
        if document.raw_outputs:
            for attempt, raw_output in enumerate(document.raw_outputs, start=1):
                fence = _text_fence(raw_output)
                lines.extend(
                    [
                        f"Attempt {attempt}:",
                        f"{fence}text",
                        raw_output,
                        fence,
                        "",
                    ]
                )
        else:
            lines.extend(["No raw response was recorded.", ""])
    return "\n".join(lines)


def write_detail_report(run_dir: Path, data_root: Path) -> Path:
    """Load a run and its local GT documents, then write ``report_detail.md``."""
    directory = Path(run_dir)
    results = load_run(directory)
    _require_external_run_directory(directory)
    gt_by_doc = _ground_truth_for_run(results, Path(data_root))
    destination = directory / "report_detail.md"
    _atomic_write_text(destination, render_detail_report(results, gt_by_doc))
    return destination


def _manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    """Copy public manifest fields explicitly so future fields default to private."""
    return {
        "run_id": manifest.run_id,
        "started_at_utc": manifest.started_at_utc,
        "model_tag": manifest.model_tag,
        "model_digest": manifest.model_digest,
        "prompt_name": manifest.prompt_name,
        "prompt_hash": manifest.prompt_hash,
        "split": manifest.split,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "preprocess_version": manifest.preprocess_version,
        "ollama_version": manifest.ollama_version,
        "seed": manifest.seed,
        "temperature": manifest.temperature,
        "gpu_fully_loaded": manifest.gpu_fully_loaded,
        "participation": manifest.participation,
        "os_info": manifest.os_info,
        "gpu_info": manifest.gpu_info,
        "adapter_kind": manifest.adapter_kind,
    }


def _summary_payload(summary: RunSummary) -> dict[str, Any]:
    """Copy aggregate fields explicitly and normalize every nested map."""
    return {
        "n_docs": summary.n_docs,
        "exact_match_count": summary.exact_match_count,
        "exact_match_rate": summary.exact_match_rate,
        "exact_match_ci": list(summary.exact_match_ci),
        "schema_valid_rate": summary.schema_valid_rate,
        "first_attempt_json_rate": summary.first_attempt_json_rate,
        "after_retry_success_rate": summary.after_retry_success_rate,
        "header_field_accuracy": {
            field: summary.header_field_accuracy[field] for field in _HEADER_FIELDS
        },
        "item_field_accuracy": {
            field: summary.item_field_accuracy[field] for field in _ITEM_FIELDS
        },
        "by_source_kind": {
            source_kind: {
                "n_docs": source_summary.n_docs,
                "exact_match_rate": source_summary.exact_match_rate,
                "schema_valid_rate": source_summary.schema_valid_rate,
            }
            for source_kind, source_summary in sorted(summary.by_source_kind.items())
        },
        "error_tag_counts": {
            tag: summary.error_tag_counts[tag] for tag in sorted(summary.error_tag_counts)
        },
        "latency_warm_p50_ms": summary.latency_warm_p50_ms,
        "latency_warm_p95_ms": summary.latency_warm_p95_ms,
        "mean_timings_ms": {field: summary.mean_timings_ms[field] for field in _TIMING_FIELDS},
        "token_totals": {field: summary.token_totals[field] for field in ("prompt", "output")},
        "mean_tokens_per_sec": summary.mean_tokens_per_sec,
        "needs_review_count": summary.needs_review_count,
    }


def _require_matching_summary(results: RunResults, summary: RunSummary) -> None:
    """Reject stale or caller-modified aggregate data."""
    if summary != aggregate(results.documents):
        raise ValueError("summary does not match the supplied run results")


def _analysis_template() -> str:
    return "\n".join(
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


def _detail_rows(ground_truth: OrderDocument, prediction: Mapping[str, Any] | None) -> list[str]:
    predicted = OrderDocument.model_validate(prediction) if prediction is not None else None
    rows: list[str] = []
    for field in _HEADER_FIELDS:
        expected = getattr(ground_truth, field)
        actual = getattr(predicted, field) if predicted is not None else None
        rows.append(
            _table_row(
                field,
                _markdown_value(expected),
                _markdown_value(actual),
                _match(expected, actual),
            )
        )
    max_items = max(len(ground_truth.items), len(predicted.items) if predicted is not None else 0)
    for index in range(max_items):
        gt_item = ground_truth.items[index] if index < len(ground_truth.items) else None
        predicted_item = (
            predicted.items[index]
            if predicted is not None and index < len(predicted.items)
            else None
        )
        for field in _ITEM_FIELDS:
            expected = getattr(gt_item, field) if gt_item is not None else None
            actual = getattr(predicted_item, field) if predicted_item is not None else None
            rows.append(
                _table_row(
                    f"items[{index}].{field}",
                    _markdown_value(expected),
                    _markdown_value(actual),
                    _match(expected, actual),
                )
            )
    return rows


def _ground_truth_for_run(
    results: RunResults,
    data_root: Path,
) -> dict[str, OrderDocument]:
    """Load only GT bound to the run's validated manifest and fingerprint."""
    if _is_link_or_reparse(data_root):
        raise ValueError("dataset root must not be an alias")
    try:
        manifest = load_manifest(data_root)
        before_fingerprint = dataset_fingerprint(data_root, manifest)
    except DatasetError as error:
        raise ValueError("cannot validate the detail-report dataset") from error
    if before_fingerprint != results.manifest.dataset_fingerprint:
        raise ValueError("dataset fingerprint does not match the benchmark run")
    expected_documents = sorted(
        (entry.doc_id, entry.source_kind)
        for entry in manifest.docs
        if entry.split == results.manifest.split
    )
    actual_documents = sorted(
        (document.doc_id, document.source_kind) for document in results.documents
    )
    if expected_documents != actual_documents:
        raise ValueError("dataset split does not match the benchmark run")

    ground_truth = {
        document.doc_id: _load_ground_truth(data_root, document.doc_id)
        for document in results.documents
    }
    try:
        after_manifest = load_manifest(data_root)
        after_fingerprint = dataset_fingerprint(data_root, after_manifest)
    except DatasetError as error:
        raise ValueError("detail-report dataset changed while it was read") from error
    if after_manifest != manifest or after_fingerprint != before_fingerprint:
        raise ValueError("detail-report dataset changed while it was read")
    return ground_truth


def _load_ground_truth(data_root: Path, doc_id: str) -> OrderDocument:
    gt_directory = data_root / "gt"
    path = gt_directory / f"{doc_id}.json"
    try:
        if _is_link_or_reparse(gt_directory) or _is_link_or_reparse(path):
            raise ValueError("ground-truth path is an alias")
        details = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("ground-truth path is not a regular file")
        root = gt_directory.resolve(strict=True)
        if path.resolve(strict=True).parent != root:
            raise ValueError("ground-truth path escapes gt directory")
        with path.open("r", encoding="utf-8", newline="") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
        return OrderDocument.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"ground truth is invalid for {doc_id}") from error


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or path.is_junction():
        return True
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and getattr(details, "st_file_attributes", 0) & reparse_flag)


def _sensitive_strings(results: RunResults) -> set[str]:
    values: set[str] = set()
    for document in results.documents:
        values.update(value for value in document.raw_outputs if value)
        values.update(_string_leaves(document.prediction))
    return values


def _string_leaves(value: object) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    values: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            values.update(_string_leaves(item))
        return values
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            values.update(_string_leaves(item))
        return values
    return set()


def _require_external_run_directory(directory: Path) -> None:
    """Allow sensitive detail output only in the configured external runs root."""
    configured_root = config.runs_dir()
    try:
        if _is_link_or_reparse(configured_root):
            raise ValueError("configured runs root must not be an alias")
        root = configured_root.resolve(strict=True)
        resolved_directory = directory.resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot validate the configured runs root") from error
    if resolved_directory.parent != root:
        raise ValueError("detail reports may only be written under OCRBENCH_RUNS_DIR")
    source_checkout = Path(__file__).resolve().parents[2]
    if resolved_directory.is_relative_to(source_checkout):
        raise ValueError("detail reports must not be written inside the source repository")


def _atomic_create_text(path: Path, content: str) -> None:
    """Atomically create a UTF-8 file without replacing an existing path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".ocrbench-report-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        with suppress(FileExistsError):
            os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a report file with UTF-8 text on the same volume."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".ocrbench-report-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _table_row(*cells: str) -> str:
    return "| " + " | ".join(_escape_markdown(cell) for cell in cells) + " |"


def _escape_markdown(value: str) -> str:
    return (
        value.replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def _markdown_value(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _match(expected: object, actual: object) -> str:
    return "yes" if expected == actual else "no"


def _json_block(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _text_fence(value: str) -> str:
    """Return a Markdown fence that cannot be closed by the supplied text."""
    fence = "````"
    while fence in value:
        fence += "`"
    return fence


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _percent_or_na(value: float | None) -> str:
    return _percent(value) if value is not None else "N/A"


def _ci(interval: tuple[float, float]) -> str:
    return f"{interval[0]:.3f} to {interval[1]:.3f}"


def _milliseconds(value: float) -> str:
    return f"{value:.0f} ms"


def _milliseconds_or_na(value: float | None) -> str:
    return _milliseconds(value) if value is not None else "N/A"


def _number(value: float) -> str:
    return f"{value:.1f}"


def _number_or_na(value: float | None) -> str:
    return _number(value) if value is not None else "N/A"


def _participation_label(value: str) -> str:
    if value == "primary":
        return "primary (eligible benchmark result)"
    return "reference (not eligible for primary comparison)"


def _adapter_kind_label(value: str) -> str:
    return "real (Ollama adapter)" if value == "real" else "fake (test adapter)"


def _yes_no_unknown(value: bool | None) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


__all__ = [
    "assert_no_forbidden_content",
    "export_anonymous_summary",
    "render_detail_report",
    "render_report",
    "write_anonymous_summary",
    "write_detail_report",
    "write_report",
]
