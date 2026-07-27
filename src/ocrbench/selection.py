"""Deterministic prompt-candidate adoption and improvement-loop control."""

from __future__ import annotations

import contextlib
import json
import math
import os
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ocrbench.metrics import RunSummary
from ocrbench.prompts import PromptVersion, activate, get_active

MAX_CANDIDATES: Final[int] = 10
MAX_CONSECUTIVE_NO_IMPROVE: Final[int] = 3
MAX_CONSECUTIVE_ERRORS: Final[int] = 3
TIME_LIMIT: Final[timedelta] = timedelta(hours=2)

_REQUIRED_HEADER_FIELD = "order_no"
_REQUIRED_ITEM_FIELDS: Final[tuple[str, str]] = ("part_no", "num_pieces")


@dataclass(frozen=True)
class CandidateEval:
    """One prompt version and its aggregate benchmark result."""

    prompt: PromptVersion
    summary: RunSummary


@dataclass(frozen=True)
class AdoptionDecision:
    """The complete, auditable result of applying the adoption policy."""

    adopted: bool
    reasons: tuple[str, ...]


class LoopState(BaseModel):
    """Persisted state for a deterministic prompt-improvement loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    started_at_utc: str
    candidates_tried: int = Field(ge=0)
    consecutive_no_improve: int = Field(ge=0)
    consecutive_errors: int = Field(ge=0)
    best_exact_match_rate: float = Field(ge=0.0, le=1.0)
    baseline_prompt_hash: str

    @field_validator("started_at_utc")
    @classmethod
    def _validate_started_at_utc(cls, value: str) -> str:
        """Require an unambiguous UTC ISO-8601 timestamp."""
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("started_at_utc must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("started_at_utc must use UTC")
        return value

    @field_validator("best_exact_match_rate")
    @classmethod
    def _validate_finite_rate(cls, value: float) -> float:
        """Reject non-finite values that would make stopping non-deterministic."""
        if not math.isfinite(value):
            raise ValueError("best_exact_match_rate must be finite")
        return value

    @field_validator("baseline_prompt_hash")
    @classmethod
    def _validate_prompt_hash(cls, value: str) -> str:
        """Require the content-addressed hash format produced by the registry."""
        if len(value) != 12 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("baseline_prompt_hash must be a 12-character lowercase hex hash")
        return value


class StopReason(StrEnum):
    """The bounded set of improvement-loop termination causes."""

    MAX_CANDIDATES = "max_candidates"
    NO_IMPROVEMENT = "no_improvement"
    TIME_LIMIT = "time_limit"
    PERFECT = "perfect"
    ERRORS = "errors"


def is_adoptable(baseline: CandidateEval, candidate: CandidateEval) -> AdoptionDecision:
    """Evaluate every ADR §8 Development adoption condition without short-circuiting."""
    _validate_candidate(baseline)
    _validate_candidate(candidate)
    _require_matching_document_counts(baseline, candidate)

    baseline_invalid = _invalid_json_count(baseline.summary)
    candidate_invalid = _invalid_json_count(candidate.summary)
    json_validity_ok = candidate_invalid <= baseline_invalid

    required_exact_count = baseline.summary.exact_match_count + 1
    exact_match_ok = candidate.summary.exact_match_count >= required_exact_count

    accuracy_results = [
        (
            "order_no header accuracy",
            candidate.summary.header_field_accuracy[_REQUIRED_HEADER_FIELD],
            baseline.summary.header_field_accuracy[_REQUIRED_HEADER_FIELD],
        ),
        *[
            (
                f"{field} item accuracy",
                candidate.summary.item_field_accuracy[field],
                baseline.summary.item_field_accuracy[field],
            )
            for field in _REQUIRED_ITEM_FIELDS
        ],
    ]
    accuracy_ok = all(
        candidate_value >= baseline_value for _, candidate_value, baseline_value in accuracy_results
    )

    reasons = (
        _reason(
            json_validity_ok,
            "invalid JSON count",
            candidate_invalid,
            "<=",
            baseline_invalid,
        ),
        _reason(
            exact_match_ok,
            "exact document matches",
            candidate.summary.exact_match_count,
            ">=",
            required_exact_count,
        ),
        *(
            _reason(candidate_value >= baseline_value, label, candidate_value, ">=", baseline_value)
            for label, candidate_value, baseline_value in accuracy_results
        ),
    )
    return AdoptionDecision(json_validity_ok and exact_match_ok and accuracy_ok, reasons)


def pick_best(candidates: Sequence[CandidateEval]) -> CandidateEval:
    """Select one aggregate-only candidate with the ADR's complete stable tie-breaker."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    expected_documents = candidates[0].summary.n_docs
    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.summary.n_docs != expected_documents:
            raise ValueError("all candidates must contain the same number of documents")
    return min(candidates, key=_selection_key)


def should_stop(state: LoopState, *, now_utc: datetime) -> StopReason | None:
    """Return the first applicable ADR §8 stopping condition, if any."""
    now = _require_utc(now_utc, "now_utc")
    started = _parse_utc_timestamp(state.started_at_utc)
    if now < started:
        raise ValueError("now_utc must not precede started_at_utc")
    if state.candidates_tried >= MAX_CANDIDATES:
        return StopReason.MAX_CANDIDATES
    if state.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        return StopReason.ERRORS
    if state.consecutive_no_improve >= MAX_CONSECUTIVE_NO_IMPROVE:
        return StopReason.NO_IMPROVEMENT
    if now - started >= TIME_LIMIT:
        return StopReason.TIME_LIMIT
    if state.best_exact_match_rate >= 1.0:
        return StopReason.PERFECT
    return None


def update_after_candidate(
    state: LoopState,
    decision: AdoptionDecision,
    *,
    exact_match_rate: float | None = None,
) -> LoopState:
    """Record a completed candidate evaluation and reset its error streak."""
    if exact_match_rate is not None and (
        not math.isfinite(exact_match_rate) or not 0.0 <= exact_match_rate <= 1.0
    ):
        raise ValueError("exact_match_rate must be finite and in [0, 1]")
    return state.model_copy(
        update={
            "candidates_tried": state.candidates_tried + 1,
            "consecutive_no_improve": 0 if decision.adopted else state.consecutive_no_improve + 1,
            "consecutive_errors": 0,
            "best_exact_match_rate": (
                state.best_exact_match_rate
                if exact_match_rate is None
                else max(state.best_exact_match_rate, exact_match_rate)
            ),
        }
    )


def update_after_error(state: LoopState) -> LoopState:
    """Record a failed candidate execution as both an error and no improvement."""
    return state.model_copy(
        update={
            "candidates_tried": state.candidates_tried + 1,
            "consecutive_no_improve": state.consecutive_no_improve + 1,
            "consecutive_errors": state.consecutive_errors + 1,
        }
    )


def apply_adoption(
    registry_root: Path,
    name: str,
    baseline: CandidateEval,
    candidates: Sequence[CandidateEval],
) -> PromptVersion:
    """Activate the strongest adoptable candidate, otherwise preserve the active prompt."""
    if baseline.prompt.name != name:
        raise ValueError("baseline prompt name does not match name")
    _require_matching_active_baseline(registry_root, name, baseline.prompt)

    adoptable: list[CandidateEval] = []
    for candidate in candidates:
        if candidate.prompt.name != name:
            raise ValueError("candidate prompt name does not match name")
        if is_adoptable(baseline, candidate).adopted:
            adoptable.append(candidate)
    if not adoptable:
        return baseline.prompt

    selected = pick_best(adoptable)
    activate(registry_root, name, selected.prompt.hash, note="automatic adoption")
    return get_active(registry_root, name)


def load_loop_state(path: Path) -> LoopState:
    """Load a strict JSON loop-state file without accepting duplicate object keys."""
    source = _regular_file(Path(path), "loop state")
    try:
        text = source.read_text(encoding="utf-8")
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        return LoopState.model_validate(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"invalid loop state {source}: {error}") from error


def save_loop_state(path: Path, state: LoopState) -> None:
    """Atomically persist a validated loop state as canonical UTF-8 JSON."""
    destination = Path(path)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        validated = LoopState.model_validate(state.model_dump(mode="python"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_directory_chain(destination.parent)
        _reject_nonregular_destination(destination)
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                validated.model_dump(mode="json"),
                stream,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ValueError(f"cannot save loop state {destination}: {error}") from error


def _selection_key(candidate: CandidateEval) -> tuple[int, int, float, int, float, str]:
    """Return the total ordering key for blind selection over aggregate metrics only."""
    summary = candidate.summary
    accuracy_sum = (
        summary.header_field_accuracy[_REQUIRED_HEADER_FIELD]
        + summary.item_field_accuracy[_REQUIRED_ITEM_FIELDS[0]]
        + summary.item_field_accuracy[_REQUIRED_ITEM_FIELDS[1]]
    )
    return (
        -summary.exact_match_count,
        _invalid_json_count(summary),
        -accuracy_sum,
        summary.token_totals["output"],
        math.inf if summary.latency_warm_p50_ms is None else summary.latency_warm_p50_ms,
        candidate.prompt.hash,
    )


def _invalid_json_count(summary: RunSummary) -> int:
    """Derive the ADR JSON-invalid document count from the aggregate schema rate."""
    valid_count = round(summary.n_docs * summary.schema_valid_rate)
    if not math.isclose(summary.schema_valid_rate, valid_count / summary.n_docs, abs_tol=1e-12):
        raise ValueError("schema_valid_rate must correspond to a whole-document count")
    return summary.n_docs - valid_count


def _validate_candidate(candidate: CandidateEval) -> None:
    """Reject malformed aggregate inputs that could defeat a total ordering."""
    summary = candidate.summary
    if (
        isinstance(summary.n_docs, bool)
        or not isinstance(summary.n_docs, int)
        or summary.n_docs < 1
    ):
        raise ValueError("n_docs must be a positive integer")
    if (
        isinstance(summary.exact_match_count, bool)
        or not isinstance(summary.exact_match_count, int)
        or not 0 <= summary.exact_match_count <= summary.n_docs
    ):
        raise ValueError("exact_match_count must be an integer in [0, n_docs]")
    _invalid_json_count(summary)
    for mapping, field in (
        (summary.header_field_accuracy, _REQUIRED_HEADER_FIELD),
        *((summary.item_field_accuracy, field) for field in _REQUIRED_ITEM_FIELDS),
    ):
        try:
            accuracy = mapping[field]
        except KeyError as error:
            raise ValueError(f"missing required field accuracy {field!r}") from error
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"{field} accuracy must be finite and in [0, 1]")
    try:
        output_tokens = summary.token_totals["output"]
    except KeyError as error:
        raise ValueError("missing output token total") from error
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError("output token total must be a non-negative integer")
    warm_p50 = summary.latency_warm_p50_ms
    if warm_p50 is not None and (not math.isfinite(warm_p50) or warm_p50 < 0.0):
        raise ValueError("warm p50 must be finite and non-negative when present")
    if len(candidate.prompt.hash) != 12 or any(
        character not in "0123456789abcdef" for character in candidate.prompt.hash
    ):
        raise ValueError("prompt hash must be a 12-character lowercase hex hash")


def _require_matching_document_counts(baseline: CandidateEval, candidate: CandidateEval) -> None:
    """Ensure an adoption comparison covers the same fixed Development split."""
    if baseline.summary.n_docs != candidate.summary.n_docs:
        raise ValueError("baseline and candidate must contain the same number of documents")


def _reason(
    passed: bool, label: str, actual: int | float, operator: str, required: int | float
) -> str:
    """Render one stable decision reason for audit logs and callers."""
    status = "passed" if passed else "failed"
    return f"{label}: {status} ({actual} {operator} {required})"


def _parse_utc_timestamp(value: str) -> datetime:
    """Parse the LoopState timestamp after its model-level validation."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _require_utc(value: datetime, label: str) -> datetime:
    """Require a timezone-aware UTC value supplied by the caller."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must use UTC")
    return value.astimezone(UTC)


def _require_matching_active_baseline(
    registry_root: Path, name: str, baseline: PromptVersion
) -> None:
    """Refuse to mutate a registry when the caller supplied a stale baseline."""
    active = get_active(registry_root, name)
    if active != baseline:
        raise ValueError("baseline prompt is not the registry's active prompt")


def _regular_file(path: Path, label: str) -> Path:
    """Return a regular non-alias file or raise a clear persistence error."""
    try:
        if _is_link_or_reparse(path):
            raise ValueError(f"{label} must not be a symlink or reparse alias: {path}")
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise ValueError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether a path is a symlink, junction, or Windows reparse alias."""
    if path.is_symlink() or path.is_junction():
        return True
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(details, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _require_safe_directory_chain(path: Path) -> None:
    """Reject aliases and non-directories in the destination's existing path chain."""
    current = path.absolute()
    while True:
        if _is_link_or_reparse(current):
            raise ValueError(
                f"loop state parent must not use a symlink or reparse alias: {current}"
            )
        try:
            details = current.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"cannot inspect loop state parent {current}: {error}") from error
        if not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"loop state parent is not a directory: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _reject_nonregular_destination(path: Path) -> None:
    """Reject aliases and special files before atomically replacing a state file."""
    if not path.exists() and not path.is_symlink():
        return
    _regular_file(path, "loop state")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Preserve JSON's object-key uniqueness contract during state loading."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
