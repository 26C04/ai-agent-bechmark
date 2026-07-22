"""Deterministic, pure-Python metrics for OCR benchmark results."""

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from ocrbench import config
from ocrbench.normalize import strip_outer_whitespace
from ocrbench.parsing import ParseStatus
from ocrbench.schema import OrderDocument
from ocrbench.scoring import DocumentScore

_HEADER_FIELDS = ("customer_name", "order_no", "delivery_date")
_ITEM_FIELDS = ("part_no", "material", "num_pieces")
_SOURCE_KINDS = ("scan", "photo")
_TIMING_FIELDS = ("preprocess", "load", "infer", "parse_validate", "total")
_TOKEN_FIELDS = ("prompt", "output")


class DocResultLike(Protocol):
    """Read-only result contract shared with T12's persisted document model.

    T12 persists ``score`` as a JSON-compatible mapping. This protocol uses a
    restoration accessor so its ``DocumentResult`` can be passed directly to
    these metrics without an adapter or wrapper.
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def source_kind(self) -> str: ...

    @property
    def first_attempt_status(self) -> ParseStatus: ...

    @property
    def final_status(self) -> ParseStatus: ...

    @property
    def attempt_count(self) -> int: ...

    @property
    def prediction(self) -> Mapping[str, object] | None: ...

    @property
    def warm(self) -> bool: ...

    @property
    def timings_ms(self) -> Mapping[str, float]: ...

    @property
    def tokens(self) -> Mapping[str, int | None]: ...

    @property
    def tokens_per_sec(self) -> float | None: ...

    @property
    def needs_review(self) -> bool: ...

    def parsed_score(self) -> DocumentScore:
        """Restore and return the persisted document score."""


@dataclass(frozen=True)
class SourceKindSummary:
    """Accuracy summary for one input source kind."""

    n_docs: int
    exact_match_rate: float
    schema_valid_rate: float


@dataclass(frozen=True)
class RunSummary:
    """All deterministic aggregate metrics for one benchmark run."""

    n_docs: int
    exact_match_count: int
    exact_match_rate: float
    exact_match_ci: tuple[float, float]
    schema_valid_rate: float
    first_attempt_json_rate: float
    after_retry_success_rate: float | None
    header_field_accuracy: dict[str, float]
    item_field_accuracy: dict[str, float]
    by_source_kind: dict[str, SourceKindSummary]
    error_tag_counts: dict[str, int]
    latency_warm_p50_ms: float | None
    latency_warm_p95_ms: float | None
    mean_timings_ms: dict[str, float]
    token_totals: dict[str, int]
    mean_tokens_per_sec: float | None
    needs_review_count: int


@dataclass(frozen=True)
class PairedComparison:
    """Exact-match contingency counts for two paired runs."""

    n_docs: int
    a_only: int
    b_only: int
    both: int
    neither: int
    mcnemar_p_value: float


@dataclass(frozen=True)
class ReproSummary:
    """Prediction stability and per-run accuracy for exactly three runs."""

    n_docs: int
    identical_prediction_rate: float
    exact_match_rates: tuple[float, ...]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval after validating binomial inputs."""
    _require_int("successes", successes, minimum=0)
    _require_int("n", n, minimum=1)
    if successes > n:
        raise ValueError("successes must not exceed n")
    _require_number("z", z, positive=True)
    proportion = successes / n
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2 * n)) / denominator
    half = (
        z / denominator * math.sqrt(proportion * (1.0 - proportion) / n + z_squared / (4 * n * n))
    )
    return (max(0.0, center - half), min(1.0, center + half))


def percentile(values: Sequence[float], q: float) -> float:
    """Return a nearest-rank percentile; reject empty or invalid inputs."""
    if not values:
        raise ValueError("values must not be empty")
    validated_q = _require_number("q", q)
    if validated_q <= 0.0 or validated_q > 100.0:
        raise ValueError("q must be in (0, 100]")
    ordered = sorted(_require_number("value", value) for value in values)
    numerator, denominator = Decimal(str(validated_q)).as_integer_ratio()
    rank_denominator = denominator * 100
    rank = (numerator * len(ordered) + rank_denominator - 1) // rank_denominator
    return ordered[rank - 1]


def bootstrap_ci(
    per_doc_values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float],
    n_boot: int = 2000,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Return a deterministic document-resampled percentile 95% interval."""
    if not per_doc_values:
        raise ValueError("per_doc_values must not be empty")
    _require_int("n_boot", n_boot, minimum=1)
    _require_int("seed", seed)
    values = [_require_number("per-document value", value) for value in per_doc_values]
    generator = random.Random(seed)
    sample_size = len(values)
    statistics: list[float] = []
    for _ in range(n_boot):
        sample = [values[generator.randrange(sample_size)] for _ in range(sample_size)]
        statistics.append(_require_number("bootstrap statistic", stat_fn(sample)))
    return (percentile(statistics, 2.5), percentile(statistics, 97.5))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant counts."""
    _require_int("a_only", a_only, minimum=0)
    _require_int("b_only", b_only, minimum=0)
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    probability = (2 * tail) / (1 << discordant)
    return min(1.0, probability)


def aggregate(results: Sequence[DocResultLike]) -> RunSummary:
    """Aggregate a non-empty run independent of document input order.

    A run without a warm document reports ``None`` for both warm percentiles
    because there is no measured warm latency distribution.
    """
    ordered = _validated_results(results)
    scores = [result.parsed_score() for result in ordered]
    n_docs = len(ordered)
    exact_count = sum(score.exact_match for score in scores)
    retries = [result for result in ordered if result.attempt_count == 2]
    warm_latencies = [result.timings_ms["total"] for result in ordered if result.warm]
    token_rates = [result.tokens_per_sec for result in ordered if result.tokens_per_sec is not None]
    tag_counts = Counter(tag for score in scores for tag in score.error_tags)
    return RunSummary(
        n_docs=n_docs,
        exact_match_count=exact_count,
        exact_match_rate=exact_count / n_docs,
        exact_match_ci=wilson_interval(exact_count, n_docs),
        schema_valid_rate=sum(result.final_status is ParseStatus.OK for result in ordered) / n_docs,
        first_attempt_json_rate=sum(
            result.first_attempt_status is ParseStatus.OK for result in ordered
        )
        / n_docs,
        after_retry_success_rate=(
            sum(result.final_status is ParseStatus.OK for result in retries) / len(retries)
            if retries
            else None
        ),
        header_field_accuracy={
            field: sum(score.header_correct[field] for score in scores) / n_docs
            for field in _HEADER_FIELDS
        },
        item_field_accuracy={field: _item_accuracy(scores, field) for field in _ITEM_FIELDS},
        by_source_kind=_source_summaries(ordered),
        error_tag_counts=dict(sorted(tag_counts.items())),
        latency_warm_p50_ms=(percentile(warm_latencies, 50.0) if warm_latencies else None),
        latency_warm_p95_ms=(percentile(warm_latencies, 95.0) if warm_latencies else None),
        mean_timings_ms={
            field: sum(result.timings_ms[field] for result in ordered) / n_docs
            for field in _TIMING_FIELDS
        },
        token_totals={
            field: sum(result.tokens[field] or 0 for result in ordered) for field in _TOKEN_FIELDS
        },
        mean_tokens_per_sec=(sum(token_rates) / len(token_rates) if token_rates else None),
        needs_review_count=sum(result.needs_review for result in ordered),
    )


def item_field_bootstrap_ci(results: Sequence[DocResultLike], field: str) -> tuple[float, float]:
    """Return a fixed-seed, document-unit bootstrap CI for one item field."""
    if field not in _ITEM_FIELDS:
        raise ValueError(f"unknown item field: {field}")
    ordered = _validated_results(results)
    counts = [result.parsed_score().item_field_counts[field] for result in ordered]

    def sampled_accuracy(indices: Sequence[float]) -> float:
        sampled = [counts[int(index)] for index in indices]
        total = sum(count[1] for count in sampled)
        return sum(count[0] for count in sampled) / total if total else 0.0

    return bootstrap_ci([float(index) for index in range(len(counts))], sampled_accuracy)


def paired_comparison(a: Sequence[DocResultLike], b: Sequence[DocResultLike]) -> PairedComparison:
    """Compare exact matches after a strict one-to-one document ID join."""
    a_by_id = _validated_by_id(a)
    b_by_id = _validated_by_id(b)
    if a_by_id.keys() != b_by_id.keys():
        raise ValueError("runs must contain the same doc_id set")
    a_only = b_only = both = neither = 0
    for doc_id in sorted(a_by_id):
        a_exact = a_by_id[doc_id].parsed_score().exact_match
        b_exact = b_by_id[doc_id].parsed_score().exact_match
        if a_exact and b_exact:
            both += 1
        elif a_exact:
            a_only += 1
        elif b_exact:
            b_only += 1
        else:
            neither += 1
    return PairedComparison(
        len(a_by_id), a_only, b_only, both, neither, mcnemar_exact(a_only, b_only)
    )


def reproducibility(runs: Sequence[Sequence[DocResultLike]]) -> ReproSummary:
    """Measure normalized prediction identity across exactly three runs.

    Each validated ``OrderDocument`` dump has its string fields outer-trimmed
    as in T05 and is serialized with sorted object keys. Item list order is
    preserved. Any parse failure makes the document non-identical, even when
    the failure status is the same in all runs.
    """
    if len(runs) != 3:
        raise ValueError("reproducibility requires exactly three runs")
    indexed = [_validated_by_id(run) for run in runs]
    expected_ids = indexed[0].keys()
    if any(run.keys() != expected_ids for run in indexed[1:]):
        raise ValueError("runs must contain the same doc_id set")
    identical = 0
    for doc_id in sorted(expected_ids):
        documents = [run[doc_id] for run in indexed]
        if any(
            result.final_status is not ParseStatus.OK or result.prediction is None
            for result in documents
        ):
            continue
        if len({_serialized_prediction(result.prediction) for result in documents}) == 1:
            identical += 1
    n_docs = len(expected_ids)
    return ReproSummary(
        n_docs=n_docs,
        identical_prediction_rate=identical / n_docs,
        exact_match_rates=tuple(
            sum(result.parsed_score().exact_match for result in run.values()) / n_docs
            for run in indexed
        ),
    )


def _source_summaries(results: Sequence[DocResultLike]) -> dict[str, SourceKindSummary]:
    summaries: dict[str, SourceKindSummary] = {}
    for source_kind in _SOURCE_KINDS:
        matching = [result for result in results if result.source_kind == source_kind]
        if not matching:
            continue
        summaries[source_kind] = SourceKindSummary(
            n_docs=len(matching),
            exact_match_rate=sum(result.parsed_score().exact_match for result in matching)
            / len(matching),
            schema_valid_rate=sum(result.final_status is ParseStatus.OK for result in matching)
            / len(matching),
        )
    return summaries


def _validated_results(results: Sequence[DocResultLike]) -> list[DocResultLike]:
    by_id = _validated_by_id(results)
    return [by_id[doc_id] for doc_id in sorted(by_id)]


def _validated_by_id(results: Sequence[DocResultLike]) -> dict[str, DocResultLike]:
    if not results:
        raise ValueError("results must not be empty")
    by_id: dict[str, DocResultLike] = {}
    for result in results:
        if not result.doc_id:
            raise ValueError("doc_id must not be empty")
        if result.doc_id in by_id:
            raise ValueError(f"duplicate doc_id: {result.doc_id}")
        if result.source_kind not in _SOURCE_KINDS:
            raise ValueError(f"invalid source_kind: {result.source_kind}")
        if result.first_attempt_status not in ParseStatus or result.final_status not in ParseStatus:
            raise ValueError("invalid parse status")
        if isinstance(result.attempt_count, bool) or result.attempt_count not in (1, 2):
            raise ValueError("attempt_count must be 1 or 2")
        if not isinstance(result.warm, bool) or not isinstance(result.needs_review, bool):
            raise ValueError("warm and needs_review must be bool")
        _validate_score(result.parsed_score())
        _validate_timings(result.timings_ms)
        _validate_tokens(result.tokens)
        if result.tokens_per_sec is not None:
            _require_number("tokens_per_sec", result.tokens_per_sec, minimum=0.0)
        by_id[result.doc_id] = result
    return by_id


def _validate_score(score: DocumentScore) -> None:
    if not isinstance(score, DocumentScore):
        raise ValueError("parsed_score() must return DocumentScore")
    if set(score.header_correct) != set(_HEADER_FIELDS):
        raise ValueError("header_correct fields do not match the contract")
    if set(score.item_field_counts) != set(_ITEM_FIELDS):
        raise ValueError("item_field_counts fields do not match the contract")
    for correct, total in score.item_field_counts.values():
        _require_int("item correct count", correct, minimum=0)
        _require_int("item total count", total, minimum=0)
        if correct > total:
            raise ValueError("item correct count must not exceed total")


def _validate_timings(timings: Mapping[str, float]) -> None:
    if not set(_TIMING_FIELDS).issubset(timings):
        raise ValueError("timings_ms is missing a required field")
    for field in _TIMING_FIELDS:
        _require_number(f"timings_ms[{field}]", timings[field], minimum=0.0)


def _validate_tokens(tokens: Mapping[str, int | None]) -> None:
    if not set(_TOKEN_FIELDS).issubset(tokens):
        raise ValueError("tokens is missing a required field")
    for field in _TOKEN_FIELDS:
        value = tokens[field]
        if value is not None:
            _require_int(f"tokens[{field}]", value, minimum=0)


def _item_accuracy(scores: Sequence[DocumentScore], field: str) -> float:
    counts = [score.item_field_counts[field] for score in scores]
    total = sum(count[1] for count in counts)
    return sum(count[0] for count in counts) / total if total else 0.0


def _serialized_prediction(prediction: Mapping[str, object] | None) -> str:
    if prediction is None:
        raise ValueError("successful prediction must not be None")
    try:
        document = OrderDocument.model_validate(dict(prediction))
    except Exception as error:
        raise ValueError("prediction is not a valid OrderDocument dump") from error
    normalized = document.model_dump()
    for field in _HEADER_FIELDS:
        value = normalized[field]
        if isinstance(value, str):
            normalized[field] = strip_outer_whitespace(value)
    for item in normalized["items"]:
        for field in ("part_no", "material"):
            value = item[field]
            if isinstance(value, str):
                item[field] = strip_outer_whitespace(value)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _require_int(name: str, value: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_number(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted
