"""Strict, versioned persistence models for benchmark run artifacts."""

from __future__ import annotations

import json
import math
import re
import stat
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.parsing import ParseStatus, parse_model_output
from ocrbench.schema import OrderDocument, compute_needs_review
from ocrbench.scoring import DocumentScore

_DOC_ID_PATTERN = re.compile(r"^doc-[0-9a-f]{12}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PROMPT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
_HASH12_PATTERN = re.compile(r"^[0-9a-f]{12}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEADER_FIELDS = ("customer_name", "order_no", "delivery_date")
_ITEM_FIELDS = ("part_no", "material", "num_pieces")
_TIMING_FIELDS = ("preprocess", "load", "infer", "parse_validate", "total")
_TOKEN_FIELDS = ("prompt", "output")


class ResultsError(OcrBenchError):
    """Raised when persisted run artifacts are missing, unsafe, or invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _ScorePayload(_StrictModel):
    exact_match: bool
    header_correct: dict[str, bool]
    items_exact: bool
    item_field_counts: dict[str, list[Annotated[int, Field(ge=0)]]]
    missing_items: Annotated[int, Field(ge=0)]
    extra_items: Annotated[int, Field(ge=0)]
    error_tags: list[str]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if set(self.header_correct) != set(_HEADER_FIELDS):
            raise ValueError("header_correct fields do not match the scoring contract")
        if set(self.item_field_counts) != set(_ITEM_FIELDS):
            raise ValueError("item_field_counts fields do not match the scoring contract")
        totals: set[int] = set()
        for field, counts in self.item_field_counts.items():
            if len(counts) != 2:
                raise ValueError(f"item_field_counts[{field!r}] must contain two integers")
            if counts[0] > counts[1]:
                raise ValueError(
                    f"item_field_counts[{field!r}] correct count must not exceed total"
                )
            totals.add(counts[1])
        if len(totals) != 1:
            raise ValueError("all item field totals must agree")
        expected_exact = all(self.header_correct.values()) and self.items_exact
        if self.exact_match != expected_exact:
            raise ValueError("exact_match is inconsistent with header and item exactness")
        if self.items_exact and (self.missing_items != 0 or self.extra_items != 0):
            raise ValueError("items_exact requires zero missing and extra items")
        if self.error_tags != sorted(set(self.error_tags)):
            raise ValueError("error_tags must be unique and sorted")
        return self


def serialize_score(score: DocumentScore) -> dict[str, Any]:
    """Return a stable, explicitly JSON-safe representation of ``DocumentScore``."""
    payload = {
        "exact_match": score.exact_match,
        "header_correct": {field: score.header_correct[field] for field in _HEADER_FIELDS},
        "items_exact": score.items_exact,
        "item_field_counts": {
            field: [
                score.item_field_counts[field][0],
                score.item_field_counts[field][1],
            ]
            for field in _ITEM_FIELDS
        },
        "missing_items": score.missing_items,
        "extra_items": score.extra_items,
        "error_tags": sorted(score.error_tags),
    }
    return _ScorePayload.model_validate(payload).model_dump(mode="json")


def _restore_score(value: Mapping[str, Any]) -> DocumentScore:
    payload = _ScorePayload.model_validate(dict(value))
    return DocumentScore(
        exact_match=payload.exact_match,
        header_correct={field: payload.header_correct[field] for field in _HEADER_FIELDS},
        items_exact=payload.items_exact,
        item_field_counts={
            field: (
                payload.item_field_counts[field][0],
                payload.item_field_counts[field][1],
            )
            for field in _ITEM_FIELDS
        },
        missing_items=payload.missing_items,
        extra_items=payload.extra_items,
        error_tags=frozenset(payload.error_tags),
    )


class DocumentResult(_StrictModel):
    """One document's persisted inference, validation, score, and timing data."""

    doc_id: str
    source_kind: Literal["scan", "photo"]
    first_attempt_status: ParseStatus
    final_status: ParseStatus
    attempt_count: Annotated[int, Field(ge=1, le=2)]
    prediction: dict[str, Any] | None
    raw_outputs: list[str]
    score: dict[str, Any]
    needs_review: bool
    warm: bool
    timings_ms: dict[str, float]
    tokens: dict[str, int | None]
    tokens_per_sec: Annotated[float, Field(ge=0.0)] | None
    error: Annotated[str, Field(min_length=1, max_length=500)] | None

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if _DOC_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("doc_id must use the anonymous dataset ID format")
        return value

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        document = OrderDocument.model_validate(value)
        return document.model_dump(mode="json")

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _ScorePayload.model_validate(value).model_dump(mode="json")

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(_TIMING_FIELDS):
            raise ValueError("timings_ms must contain exactly the five required timing fields")
        for field, timing in value.items():
            if not math.isfinite(timing) or timing < 0.0:
                raise ValueError(f"timings_ms[{field!r}] must be finite and non-negative")
        if value["total"] < value["preprocess"]:
            raise ValueError("total timing must include preprocessing")
        return {field: value[field] for field in _TIMING_FIELDS}

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, value: dict[str, int | None]) -> dict[str, int | None]:
        if set(value) != set(_TOKEN_FIELDS):
            raise ValueError("tokens must contain exactly prompt and output")
        for field, count in value.items():
            if count is not None and count < 0:
                raise ValueError(f"tokens[{field!r}] must be non-negative")
        return {field: value[field] for field in _TOKEN_FIELDS}

    @model_validator(mode="after")
    def validate_attempt_consistency(self) -> Self:
        if self.attempt_count == 1 and self.final_status is not self.first_attempt_status:
            raise ValueError("one-attempt results must have identical first and final statuses")
        if self.attempt_count == 2 and self.first_attempt_status is ParseStatus.OK:
            raise ValueError("a successful first attempt must not be retried")

        parsed_outputs = [parse_model_output(raw) for raw in self.raw_outputs]
        if parsed_outputs and parsed_outputs[0].status is not self.first_attempt_status:
            raise ValueError("first_attempt_status does not match the first raw output")

        if self.error is None:
            if len(parsed_outputs) != self.attempt_count:
                raise ValueError("every completed adapter attempt must have one raw output")
            final_output = parsed_outputs[-1]
            if final_output.status is not self.final_status:
                raise ValueError("final_status does not match the final raw output")
            expected_prediction = (
                final_output.document.model_dump(mode="json")
                if final_output.document is not None
                else None
            )
            if self.prediction != expected_prediction:
                raise ValueError("prediction does not match the final validated raw output")
        else:
            if len(parsed_outputs) != self.attempt_count - 1:
                raise ValueError("adapter errors must correspond to exactly one missing response")
            if self.final_status is not ParseStatus.NOT_JSON or self.prediction is not None:
                raise ValueError("adapter errors require a synthetic failed final result")
            if self.attempt_count == 1 and self.first_attempt_status is not ParseStatus.NOT_JSON:
                raise ValueError("a first-call adapter error uses synthetic not_json status")
            if (
                self.tokens["prompt"] is not None
                or self.tokens["output"] is not None
                or self.tokens_per_sec is not None
                or self.warm
            ):
                raise ValueError("adapter-error token, rate, and warm metadata must be unknown")

        if self.final_status is ParseStatus.OK and self.prediction is None:
            raise ValueError("a successful final status requires a validated prediction")
        if self.final_status is not ParseStatus.OK and self.prediction is not None:
            raise ValueError("a failed final status cannot contain a prediction")
        parsed_prediction = (
            OrderDocument.model_validate(self.prediction) if self.prediction is not None else None
        )
        if self.needs_review != compute_needs_review(parsed_prediction):
            raise ValueError("needs_review does not match the validated prediction")
        if self.final_status is not ParseStatus.OK:
            self._validate_failure_score()
        return self

    def _validate_failure_score(self) -> None:
        score = self.parsed_score()
        totals = {counts[1] for counts in score.item_field_counts.values()}
        total = next(iter(totals))
        expected_tags = {self.final_status.value}
        if total > 0:
            expected_tags.add("missing_item")
        if (
            score.exact_match
            or score.items_exact
            or any(score.header_correct.values())
            or any(correct != 0 for correct, _ in score.item_field_counts.values())
            or score.missing_items != total
            or score.extra_items != 0
            or score.error_tags != frozenset(expected_tags)
        ):
            raise ValueError("failed parse score does not match the T07 failure contract")

    def parsed_score(self) -> DocumentScore:
        """Restore the persisted score for direct use by ``metrics.aggregate``."""
        return _restore_score(self.score)


class RunManifest(_StrictModel):
    """Immutable execution and environment identity for one benchmark run."""

    run_id: str
    started_at_utc: str
    model_tag: str
    model_digest: str
    prompt_name: str
    prompt_hash: str
    split: Literal["dev", "selection", "final"]
    dataset_fingerprint: str
    preprocess_version: str
    ollama_version: str
    seed: int
    temperature: float
    gpu_fully_loaded: bool | None
    participation: Literal["primary", "reference"]
    os_info: str
    gpu_info: str | None
    adapter_kind: Literal["real", "fake"]

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("run_id must be 1-64 ASCII letters, digits, underscores, or hyphens")
        return value

    @field_validator("started_at_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if _TIMESTAMP_PATTERN.fullmatch(value) is None:
            raise ValueError("started_at_utc must use YYYYMMDDTHHMMSSZ")
        try:
            datetime.strptime(value, "%Y%m%dT%H%M%SZ")
        except ValueError as error:
            raise ValueError("started_at_utc must be a real UTC timestamp") from error
        return value

    @field_validator("model_tag")
    @classmethod
    def validate_model_tag(cls, value: str) -> str:
        if value not in config.ALLOWED_MODELS:
            raise ValueError("model_tag is not in the benchmark allowlist")
        return value

    @field_validator("model_digest", "preprocess_version")
    @classmethod
    def validate_nonempty_identity(cls, value: str) -> str:
        if not value:
            raise ValueError("identity fields must not be empty")
        return value

    @field_validator("prompt_name")
    @classmethod
    def validate_prompt_name(cls, value: str) -> str:
        if _PROMPT_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("prompt_name does not match the T13 registry name contract")
        return value

    @field_validator("prompt_hash")
    @classmethod
    def validate_prompt_hash(cls, value: str) -> str:
        if _HASH12_PATTERN.fullmatch(value) is None:
            raise ValueError("prompt_hash must contain exactly 12 lowercase hex digits")
        return value

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_dataset_fingerprint(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("dataset_fingerprint must contain 64 lowercase hex digits")
        return value

    @model_validator(mode="after")
    def validate_execution_contract(self) -> Self:
        expected = "primary" if self.gpu_fully_loaded is True else "reference"
        if self.participation != expected:
            raise ValueError(
                "participation must be primary only when GPU placement is explicitly true"
            )
        if self.seed != config.SEED:
            raise ValueError("seed does not match the fixed benchmark seed")
        if self.temperature != 0.0:
            raise ValueError("temperature must be zero for deterministic execution")
        return self


class RunResults(_StrictModel):
    """Versioned metrics source of truth for one complete benchmark run."""

    schema_version: Literal[1]
    manifest: RunManifest
    documents: list[DocumentResult]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, value: list[DocumentResult]) -> list[DocumentResult]:
        if not value:
            raise ValueError("documents must not be empty")
        ids = [document.doc_id for document in value]
        if len(ids) != len(set(ids)):
            raise ValueError("documents must not contain duplicate doc_id values")
        if ids != sorted(ids):
            raise ValueError("documents must be sorted by doc_id")
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


def _regular_artifact(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    try:
        if _is_link_or_reparse(path):
            raise ResultsError(f"{name} must not be a symlink or reparse alias")
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ResultsError(f"run is incomplete: {name} is missing") from error
    except ResultsError:
        raise
    except OSError as error:
        raise ResultsError(f"cannot inspect {name}") from error
    if not stat.S_ISREG(details.st_mode):
        raise ResultsError(f"{name} is not a regular file")
    try:
        resolved = path.resolve(strict=True)
        root = run_dir.resolve(strict=True)
    except OSError as error:
        raise ResultsError(f"cannot resolve {name}") from error
    if resolved.parent != root:
        raise ResultsError(f"{name} escapes the run directory")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> object:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ResultsError(f"{label} is not valid strict UTF-8 JSON") from error


def _schema_version(document: object, label: str) -> None:
    if not isinstance(document, dict):
        raise ResultsError(f"{label} must contain a JSON object")
    mapping = cast(dict[str, object], document)
    if "schema_version" not in mapping:
        raise ResultsError(f"{label} schema_version is missing")
    if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
        raise ResultsError(f"{label} schema_version must be the integer 1")


def _strict_json(document: object) -> str:
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except TypeError, ValueError:
        raise ResultsError("run artifacts contain values that are not strict JSON") from None


def load_run(run_dir: Path) -> RunResults:
    """Load a complete run after strict path, schema, and cross-file checks."""
    directory = Path(run_dir)
    try:
        if _is_link_or_reparse(directory):
            raise ResultsError("run directory must not be a symlink or reparse alias")
        details = directory.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ResultsError("run directory does not exist") from error
    except ResultsError:
        raise
    except OSError as error:
        raise ResultsError("cannot inspect run directory") from error
    if not stat.S_ISDIR(details.st_mode):
        raise ResultsError("run path is not a directory")

    manifest_document = _read_json(_regular_artifact(directory, "manifest.json"), "manifest.json")
    results_document = _read_json(_regular_artifact(directory, "results.json"), "results.json")
    _schema_version(results_document, "results.json")

    try:
        manifest = RunManifest.model_validate_json(_strict_json(manifest_document))
        results = RunResults.model_validate_json(_strict_json(results_document))
    except ValidationError as error:
        raise ResultsError("run artifacts do not match the strict results schema") from error

    if results.manifest != manifest:
        raise ResultsError("manifest.json does not match results.json manifest")
    expected_name = f"{manifest.started_at_utc}_{manifest.run_id}"
    if directory.name != expected_name:
        raise ResultsError("run directory name does not match its manifest identity")
    return results


__all__ = [
    "DocumentResult",
    "ResultsError",
    "RunManifest",
    "RunResults",
    "load_run",
    "serialize_score",
]
