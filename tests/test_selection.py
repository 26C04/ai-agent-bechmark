"""Tests for deterministic prompt adoption, blind selection, and loop stopping."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ocrbench.metrics import RunSummary
from ocrbench.prompts import PromptVersion, activate, add_prompt, get_active
from ocrbench.selection import (
    MAX_CANDIDATES,
    MAX_CONSECUTIVE_ERRORS,
    MAX_CONSECUTIVE_NO_IMPROVE,
    TIME_LIMIT,
    AdoptionDecision,
    CandidateEval,
    LoopState,
    StopReason,
    apply_adoption,
    is_adoptable,
    load_loop_state,
    pick_best,
    save_loop_state,
    should_stop,
    update_after_candidate,
    update_after_error,
)


def _summary(
    *,
    n_docs: int = 10,
    exact_matches: int = 4,
    valid_json: int = 8,
    order_no: float = 0.8,
    part_no: float = 0.8,
    num_pieces: float = 0.8,
    output_tokens: int = 100,
    warm_p50: float | None = 20.0,
) -> RunSummary:
    return RunSummary(
        n_docs=n_docs,
        exact_match_count=exact_matches,
        exact_match_rate=exact_matches / n_docs,
        exact_match_ci=(0.0, 1.0),
        schema_valid_rate=valid_json / n_docs,
        first_attempt_json_rate=valid_json / n_docs,
        after_retry_success_rate=None,
        header_field_accuracy={
            "customer_name": 0.0,
            "order_no": order_no,
            "delivery_date": 0.0,
        },
        item_field_accuracy={
            "part_no": part_no,
            "material": 0.0,
            "num_pieces": num_pieces,
        },
        by_source_kind={},
        error_tag_counts={},
        latency_warm_p50_ms=warm_p50,
        latency_warm_p95_ms=warm_p50,
        mean_timings_ms={
            "preprocess": 0.0,
            "load": 0.0,
            "infer": 0.0,
            "parse_validate": 0.0,
            "total": 0.0,
        },
        token_totals={"prompt": 10, "output": output_tokens},
        mean_tokens_per_sec=None,
        needs_review_count=0,
    )


def _candidate(hash12: str, summary: RunSummary) -> CandidateEval:
    return CandidateEval(PromptVersion("base", hash12, "text"), summary)


def _state(**updates: object) -> LoopState:
    values: dict[str, object] = {
        "started_at_utc": "2026-07-24T00:00:00+00:00",
        "candidates_tried": 0,
        "consecutive_no_improve": 0,
        "consecutive_errors": 0,
        "best_exact_match_rate": 0.5,
        "baseline_prompt_hash": "a" * 12,
    }
    values.update(updates)
    return LoopState.model_validate(values)


@pytest.mark.parametrize(
    ("candidate_summary", "failed_reason"),
    [
        (_summary(valid_json=7, exact_matches=5), "invalid JSON count: failed"),
        (_summary(exact_matches=4), "exact document matches: failed"),
        (_summary(exact_matches=5, order_no=0.7), "order_no header accuracy: failed"),
        (_summary(exact_matches=5, part_no=0.7), "part_no item accuracy: failed"),
        (_summary(exact_matches=5, num_pieces=0.7), "num_pieces item accuracy: failed"),
    ],
)
def test_adoption_rejects_each_adr_violation(
    candidate_summary: RunSummary, failed_reason: str
) -> None:
    decision = is_adoptable(
        _candidate("a" * 12, _summary()), _candidate("b" * 12, candidate_summary)
    )

    assert not decision.adopted
    assert any(failed_reason in reason for reason in decision.reasons)
    assert len(decision.reasons) == 5


def test_adoption_requires_one_new_exact_match_and_keeps_all_reasons() -> None:
    baseline = _candidate("a" * 12, _summary())
    candidate = _candidate("b" * 12, _summary(exact_matches=5))

    decision = is_adoptable(baseline, candidate)

    assert decision.adopted
    assert all("passed" in reason for reason in decision.reasons)


def test_adoption_requires_comparable_fixed_split() -> None:
    with pytest.raises(ValueError, match="same number"):
        is_adoptable(_candidate("a" * 12, _summary()), _candidate("b" * 12, _summary(n_docs=8)))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_summary(exact_matches=5), _summary(exact_matches=4)),
        (_summary(valid_json=9), _summary(valid_json=8)),
        (_summary(order_no=0.9), _summary(order_no=0.8)),
        (_summary(output_tokens=90), _summary(output_tokens=100)),
        (_summary(warm_p50=10.0), _summary(warm_p50=20.0)),
    ],
)
def test_pick_best_applies_each_metric_tie_breaker(left: RunSummary, right: RunSummary) -> None:
    first = _candidate("b" * 12, left)
    second = _candidate("a" * 12, right)

    assert pick_best([second, first]) == first


def test_pick_best_ranks_missing_warm_p50_below_a_measured_value() -> None:
    missing = _candidate("a" * 12, _summary(warm_p50=None))
    measured = _candidate("b" * 12, _summary(warm_p50=1000.0))

    assert pick_best([missing, measured]) == measured


def test_pick_best_uses_prompt_hash_as_the_final_stable_tie_breaker() -> None:
    first = _candidate("b" * 12, _summary())
    second = _candidate("a" * 12, _summary())

    assert pick_best([first, second]) == second
    with pytest.raises(ValueError, match="must not be empty"):
        pick_best([])


@pytest.mark.parametrize(
    ("state", "now", "expected"),
    [
        (_state(candidates_tried=MAX_CANDIDATES - 1), timedelta(), None),
        (_state(candidates_tried=MAX_CANDIDATES), timedelta(), StopReason.MAX_CANDIDATES),
        (_state(consecutive_no_improve=MAX_CONSECUTIVE_NO_IMPROVE - 1), timedelta(), None),
        (
            _state(consecutive_no_improve=MAX_CONSECUTIVE_NO_IMPROVE),
            timedelta(),
            StopReason.NO_IMPROVEMENT,
        ),
        (_state(), TIME_LIMIT - timedelta(microseconds=1), None),
        (_state(), TIME_LIMIT, StopReason.TIME_LIMIT),
        (_state(best_exact_match_rate=0.999), timedelta(), None),
        (_state(best_exact_match_rate=1.0), timedelta(), StopReason.PERFECT),
        (_state(consecutive_errors=MAX_CONSECUTIVE_ERRORS - 1), timedelta(), None),
        (_state(consecutive_errors=MAX_CONSECUTIVE_ERRORS), timedelta(), StopReason.ERRORS),
    ],
)
def test_should_stop_enforces_each_boundary(
    state: LoopState, now: timedelta, expected: StopReason | None
) -> None:
    started = datetime(2026, 7, 24, tzinfo=UTC)

    assert should_stop(state, now_utc=started + now) == expected


def test_should_stop_rejects_non_utc_or_backwards_clocks() -> None:
    with pytest.raises(ValueError, match="UTC"):
        should_stop(_state(), now_utc=datetime(2026, 7, 24))
    with pytest.raises(ValueError, match="must not precede"):
        should_stop(_state(), now_utc=datetime(2026, 7, 23, 23, tzinfo=UTC))


def test_loop_state_persists_strictly_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "loop-state.json"
    state = _state()

    save_loop_state(path, state)

    assert load_loop_state(path) == state
    path.write_text(
        (
            '{"started_at_utc":"2026-07-24T00:00:00+00:00",'
            '"started_at_utc":"2026-07-24T00:00:00+00:00"}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_loop_state(path)


def test_loop_state_model_rejects_extra_non_utc_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        _state(unexpected=True)
    with pytest.raises(ValueError, match="UTC"):
        _state(started_at_utc="2026-07-24T00:00:00+09:00")
    with pytest.raises(ValueError, match="less than or equal to 1"):
        _state(best_exact_match_rate=float("nan"))


def test_state_updates_count_candidates_and_reset_the_relevant_streaks() -> None:
    state = _state(consecutive_errors=2, consecutive_no_improve=2)

    adopted = update_after_candidate(state, AdoptionDecision(True, ()))
    rejected = update_after_candidate(state, AdoptionDecision(False, ()))
    failed = update_after_error(state)

    assert (
        adopted.candidates_tried,
        adopted.consecutive_no_improve,
        adopted.consecutive_errors,
    ) == (1, 0, 0)
    assert (
        rejected.candidates_tried,
        rejected.consecutive_no_improve,
        rejected.consecutive_errors,
    ) == (1, 3, 0)
    assert (
        failed.candidates_tried,
        failed.consecutive_no_improve,
        failed.consecutive_errors,
    ) == (1, 3, 3)


def test_apply_adoption_activates_best_adoptable_candidate_and_records_history(
    tmp_path: Path,
) -> None:
    baseline_prompt = add_prompt(tmp_path, "base", "baseline")
    activate(tmp_path, "base", baseline_prompt.hash)
    weaker = add_prompt(tmp_path, "base", "weaker")
    stronger = add_prompt(tmp_path, "base", "stronger")
    baseline = CandidateEval(baseline_prompt, _summary())
    candidates = [
        CandidateEval(weaker, _summary(exact_matches=5, output_tokens=200)),
        CandidateEval(stronger, _summary(exact_matches=5, output_tokens=100)),
    ]

    adopted = apply_adoption(tmp_path, "base", baseline, candidates)

    assert adopted == stronger
    assert get_active(tmp_path, "base") == stronger
    history = [
        json.loads(line) for line in (tmp_path / "base" / "history.jsonl").read_text().splitlines()
    ]
    assert history[-1]["action"] == "activate"
    assert history[-1]["hash"] == stronger.hash


def test_apply_adoption_keeps_the_existing_active_prompt_when_none_qualify(tmp_path: Path) -> None:
    baseline_prompt = add_prompt(tmp_path, "base", "baseline")
    activate(tmp_path, "base", baseline_prompt.hash)
    rejected = add_prompt(tmp_path, "base", "rejected")
    baseline = CandidateEval(baseline_prompt, _summary())
    history_before = (tmp_path / "base" / "history.jsonl").read_text(encoding="utf-8")

    result = apply_adoption(tmp_path, "base", baseline, [CandidateEval(rejected, _summary())])

    assert result == baseline_prompt
    assert get_active(tmp_path, "base") == baseline_prompt
    assert (tmp_path / "base" / "history.jsonl").read_text(encoding="utf-8") == history_before
