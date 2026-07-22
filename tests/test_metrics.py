"""Known-value and contract tests for deterministic benchmark metrics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import localcontext
from typing import Any, Literal

import pytest

from ocrbench.metrics import (
    DocResultLike,
    PairedComparison,
    ReproSummary,
    RunSummary,
    SourceKindSummary,
    aggregate,
    bootstrap_ci,
    item_field_bootstrap_ci,
    mcnemar_exact,
    paired_comparison,
    percentile,
    reproducibility,
    wilson_interval,
)
from ocrbench.parsing import ParseStatus
from ocrbench.scoring import DocumentScore


def _timings() -> dict[str, float]:
    return {
        "preprocess": 1.0,
        "load": 2.0,
        "infer": 3.0,
        "parse_validate": 4.0,
        "total": 10.0,
    }


@dataclass
class FakeResult:
    """T12-shaped local result with a score restoration accessor."""

    doc_id: str
    source_kind: str
    _score: DocumentScore
    first_attempt_status: ParseStatus = ParseStatus.OK
    final_status: ParseStatus = ParseStatus.OK
    attempt_count: int = 1
    prediction: Mapping[str, object] | None = None
    warm: bool = True
    timings_ms: Mapping[str, float] = field(default_factory=_timings)
    tokens: Mapping[str, int | None] = field(default_factory=lambda: {"prompt": 10, "output": 5})
    tokens_per_sec: float | None = None
    needs_review: bool = False

    def parsed_score(self) -> DocumentScore:
        """Restore the in-memory score exactly as T12's accessor will."""
        return self._score


@dataclass
class T12ConcreteResult:
    """Concrete dict/Literal result shape matching T12's planned model."""

    doc_id: str
    source_kind: Literal["scan", "photo"]
    _score: DocumentScore
    first_attempt_status: ParseStatus = ParseStatus.OK
    final_status: ParseStatus = ParseStatus.OK
    attempt_count: int = 1
    prediction: dict[str, Any] | None = None
    warm: bool = True
    timings_ms: dict[str, float] = field(default_factory=_timings)
    tokens: dict[str, int | None] = field(default_factory=lambda: {"prompt": 10, "output": 5})
    tokens_per_sec: float | None = 1.0
    needs_review: bool = False

    def parsed_score(self) -> DocumentScore:
        return self._score


def _score(
    *,
    exact: bool,
    headers: tuple[bool, bool, bool] = (True, True, True),
    item_counts: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (1, 1),
        (1, 1),
        (1, 1),
    ),
    tags: frozenset[str] = frozenset(),
) -> DocumentScore:
    customer, order, delivery = headers
    part, material, quantity = item_counts
    return DocumentScore(
        exact_match=exact,
        header_correct={
            "customer_name": customer,
            "order_no": order,
            "delivery_date": delivery,
        },
        items_exact=exact,
        item_field_counts={
            "part_no": part,
            "material": material,
            "num_pieces": quantity,
        },
        missing_items=0,
        extra_items=0,
        error_tags=tags,
    )


def _prediction(
    *, part_no: str = "P-1", material: str = "steel", reverse: bool = False
) -> dict[str, object]:
    items: list[dict[str, object]] = [
        {"part_no": part_no, "material": material, "num_pieces": 1},
        {"part_no": "P-2", "material": "resin", "num_pieces": 2},
    ]
    if reverse:
        items.reverse()
    return {
        "customer_name": "ACME",
        "order_no": "O-1",
        "delivery_date": "2026-07-22",
        "items": items,
    }


def _aggregate_fixture() -> list[FakeResult]:
    exact_two = _score(exact=True, item_counts=((2, 2), (2, 2), (2, 2)))
    partial = _score(
        exact=False,
        headers=(False, True, True),
        item_counts=((1, 2), (2, 2), (1, 2)),
        tags=frozenset({"char_substitution", "count_mismatch"}),
    )
    failure = _score(
        exact=False,
        headers=(False, False, False),
        item_counts=((0, 1), (0, 1), (0, 1)),
        tags=frozenset({"missing_item", "not_json"}),
    )
    scores = [
        exact_two,
        _score(exact=True),
        partial,
        _score(exact=True),
        exact_two,
        failure,
    ]
    results: list[FakeResult] = []
    for index, score in enumerate(scores, start=1):
        results.append(
            FakeResult(
                doc_id=f"doc-{index}",
                source_kind="scan" if index <= 3 else "photo",
                _score=score,
                first_attempt_status=(ParseStatus.NOT_JSON if index in (2, 6) else ParseStatus.OK),
                final_status=ParseStatus.NOT_JSON if index == 6 else ParseStatus.OK,
                attempt_count=2 if index in (2, 6) else 1,
                prediction=_prediction() if index != 6 else None,
                warm=index != 6,
                timings_ms={
                    "preprocess": float(index),
                    "load": float(2 * index),
                    "infer": float(3 * index),
                    "parse_validate": float(4 * index),
                    "total": float(10 * index),
                },
                tokens={
                    "prompt": 10 * index if index != 6 else None,
                    "output": 5 * index if index != 6 else None,
                },
                tokens_per_sec=float(index) if index != 6 else None,
                needs_review=index in (3, 6),
            )
        )
    return results


def test_protocol_requires_score_restoration_accessor() -> None:
    result: DocResultLike = _aggregate_fixture()[0]
    assert isinstance(result.parsed_score(), DocumentScore)


def test_t12_concrete_types_are_protocol_compatible_at_typecheck_and_runtime() -> None:
    concrete = T12ConcreteResult("doc-t12", "scan", _score(exact=True))
    result: DocResultLike = concrete
    assert aggregate([result]).exact_match_rate == 1.0


def test_statistical_primitives_known_values() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=0.0001)
    assert high == pytest.approx(0.9433, abs=0.0001)
    values = [float(value) for value in range(1, 11)]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 95) == 10.0
    assert percentile([float(value) for value in range(1, 26)], 28) == 7.0
    assert mcnemar_exact(5, 1) == pytest.approx(0.21875)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(550, 550) == 1.0
    assert 0.0 <= mcnemar_exact(1090, 10) <= 1.0


def test_percentile_is_independent_of_decimal_context_precision() -> None:
    with localcontext() as context:
        context.prec = 1
        assert percentile([1.0, 2.0, 3.0], 34.0) == 2.0


def test_bootstrap_is_seeded_and_deterministic() -> None:
    values = [0.0, 0.0, 0.0, 1.0, 5.0]

    def mean(sample: Sequence[float]) -> float:
        return sum(sample) / len(sample)

    first = bootstrap_ci(values, mean, n_boot=101, seed=7)
    assert bootstrap_ci(values, mean, n_boot=101, seed=7) == first
    assert bootstrap_ci(values, mean, n_boot=101, seed=8) != first


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (wilson_interval, (-1, 10)),
        (wilson_interval, (11, 10)),
        (wilson_interval, (1, 0)),
        (wilson_interval, (True, 1)),
        (percentile, ([], 50.0)),
        (percentile, ([1.0], 0.0)),
        (percentile, ([1.0], 101.0)),
        (percentile, ([float("nan")], 50.0)),
        (mcnemar_exact, (-1, 0)),
        (mcnemar_exact, (0, -1)),
        (mcnemar_exact, (True, 0)),
    ],
)
def test_primitives_reject_invalid_inputs(operation: object, args: tuple[object, ...]) -> None:
    assert callable(operation)
    with pytest.raises(ValueError):
        operation(*args)


def test_bootstrap_rejects_invalid_inputs_and_statistic() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([], lambda sample: sample[0])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], lambda sample: sample[0], n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci([float("inf")], lambda sample: sample[0])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], lambda _: float("nan"), n_boot=1)


def test_aggregate_populates_every_field_and_is_order_independent() -> None:
    results = _aggregate_fixture()
    expected = RunSummary(
        n_docs=6,
        exact_match_count=4,
        exact_match_rate=4 / 6,
        exact_match_ci=wilson_interval(4, 6),
        schema_valid_rate=5 / 6,
        first_attempt_json_rate=4 / 6,
        after_retry_success_rate=0.5,
        header_field_accuracy={
            "customer_name": 4 / 6,
            "delivery_date": 5 / 6,
            "order_no": 5 / 6,
        },
        item_field_accuracy={
            "material": 8 / 9,
            "num_pieces": 7 / 9,
            "part_no": 7 / 9,
        },
        by_source_kind={
            "photo": SourceKindSummary(3, 2 / 3, 2 / 3),
            "scan": SourceKindSummary(3, 2 / 3, 1.0),
        },
        error_tag_counts={
            "char_substitution": 1,
            "count_mismatch": 1,
            "missing_item": 1,
            "not_json": 1,
        },
        latency_warm_p50_ms=30.0,
        latency_warm_p95_ms=50.0,
        mean_timings_ms={
            "infer": 10.5,
            "load": 7.0,
            "parse_validate": 14.0,
            "preprocess": 3.5,
            "total": 35.0,
        },
        token_totals={"output": 75, "prompt": 150},
        mean_tokens_per_sec=3.0,
        needs_review_count=2,
    )
    assert aggregate(results) == expected
    assert aggregate(list(reversed(results))) == expected


def test_aggregate_retry_denominator_zero_is_none() -> None:
    assert aggregate([_aggregate_fixture()[0]]).after_retry_success_rate is None


def test_aggregate_accepts_reordered_score_mappings_and_keeps_output_order() -> None:
    result = _aggregate_fixture()[0]
    score = result.parsed_score()
    reordered = replace(
        score,
        header_correct=dict(reversed(tuple(score.header_correct.items()))),
        item_field_counts=dict(reversed(tuple(score.item_field_counts.items()))),
    )
    summary = aggregate([replace(result, _score=reordered)])
    assert tuple(summary.header_field_accuracy) == (
        "customer_name",
        "order_no",
        "delivery_date",
    )
    assert tuple(summary.item_field_accuracy) == ("part_no", "material", "num_pieces")


def test_aggregate_cold_only_run_has_no_warm_percentiles() -> None:
    results = [replace(result, warm=False) for result in _aggregate_fixture()]
    summary = aggregate(results)
    assert summary.latency_warm_p50_ms is None
    assert summary.latency_warm_p95_ms is None


@pytest.mark.parametrize("case", ["empty", "duplicate", "source"])
def test_aggregate_rejects_undefined_or_invalid_runs(case: str) -> None:
    results = _aggregate_fixture()
    if case == "empty":
        invalid: list[FakeResult] = []
    elif case == "duplicate":
        invalid = [results[0], results[0]]
    else:
        invalid = [replace(results[0], source_kind="fax")]
    with pytest.raises(ValueError):
        aggregate(invalid)


def test_item_field_bootstrap_is_document_level_and_validates_field() -> None:
    results = _aggregate_fixture()
    interval = item_field_bootstrap_ci(results, "part_no")
    assert interval == item_field_bootstrap_ci(list(reversed(results)), "part_no")
    assert 0.0 <= interval[0] <= interval[1] <= 1.0
    with pytest.raises(ValueError, match="unknown item field"):
        item_field_bootstrap_ci(results, "price")


def test_paired_comparison_matches_by_doc_id() -> None:
    exact, wrong = _score(exact=True), _score(exact=False)
    a = [
        FakeResult("d1", "scan", exact),
        FakeResult("d2", "photo", exact),
        FakeResult("d3", "scan", wrong),
        FakeResult("d4", "photo", wrong),
    ]
    b = [
        FakeResult("d4", "photo", wrong),
        FakeResult("d3", "scan", exact),
        FakeResult("d2", "photo", wrong),
        FakeResult("d1", "scan", exact),
    ]
    assert paired_comparison(a, b) == PairedComparison(4, 1, 1, 1, 1, 1.0)


def test_paired_comparison_rejects_mismatched_or_duplicate_ids() -> None:
    result = _aggregate_fixture()[0]
    with pytest.raises(ValueError, match="same doc_id set"):
        paired_comparison([result], [replace(result, doc_id="other")])
    with pytest.raises(ValueError, match="duplicate doc_id"):
        paired_comparison([result, result], [result])


def test_reproducibility_trims_strings_preserves_lists_and_rejects_failures() -> None:
    exact, wrong = _score(exact=True), _score(exact=False)
    base = _prediction()
    padded = _prediction(part_no=" P-1 ", material=" steel ")
    changed = _prediction(part_no="P-X")
    failure = FakeResult("d3", "scan", wrong, final_status=ParseStatus.NOT_JSON, prediction=None)
    run_1 = [
        FakeResult("d1", "scan", exact, prediction=base),
        FakeResult("d2", "photo", wrong, prediction=base),
        failure,
    ]
    run_2 = [
        failure,
        FakeResult("d2", "photo", exact, prediction=base),
        FakeResult("d1", "scan", exact, prediction=padded),
    ]
    run_3 = [
        FakeResult("d2", "photo", exact, prediction=changed),
        FakeResult("d1", "scan", exact, prediction=base),
        failure,
    ]
    assert reproducibility([run_1, run_2, run_3]) == ReproSummary(
        n_docs=3,
        identical_prediction_rate=1 / 3,
        exact_match_rates=(1 / 3, 2 / 3, 2 / 3),
    )
    run_3[1] = replace(run_3[1], prediction=_prediction(reverse=True))
    assert reproducibility([run_1, run_2, run_3]).identical_prediction_rate == 0.0


@pytest.mark.parametrize("case", ["count", "set", "duplicate", "empty"])
def test_reproducibility_rejects_invalid_run_shapes(case: str) -> None:
    result = FakeResult("d1", "scan", _score(exact=True), prediction=_prediction())
    if case == "count":
        runs = [[result], [result]]
    elif case == "set":
        runs = [[result], [result], [replace(result, doc_id="d2")]]
    elif case == "duplicate":
        runs = [[result, result], [result], [result]]
    else:
        runs = [[], [], []]
    with pytest.raises(ValueError):
        reproducibility(runs)


def test_reproducibility_rejects_unvalidated_prediction_dump() -> None:
    result = FakeResult(
        "d1", "scan", _score(exact=True), prediction={"customer_name": "incomplete"}
    )
    with pytest.raises(ValueError, match="valid OrderDocument"):
        reproducibility([[result], [result], [result]])
