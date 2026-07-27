"""Regression tests for T15 review findings."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ocrbench.metrics import RunSummary
from ocrbench.prompts import PromptVersion
from ocrbench.selection import (
    MAX_CONSECUTIVE_ERRORS,
    AdoptionDecision,
    CandidateEval,
    LoopState,
    StopReason,
    pick_best,
    save_loop_state,
    should_stop,
    update_after_candidate,
    update_after_error,
)


def _summary(
    *,
    n_docs: int = 10,
    order_no: float = 0.8,
) -> RunSummary:
    """Build a complete aggregate result with controllable review fields."""
    return RunSummary(
        n_docs=n_docs,
        exact_match_count=4,
        exact_match_rate=4 / n_docs,
        exact_match_ci=(0.0, 1.0),
        schema_valid_rate=8 / n_docs,
        first_attempt_json_rate=8 / n_docs,
        after_retry_success_rate=None,
        header_field_accuracy={
            "customer_name": 0.0,
            "order_no": order_no,
            "delivery_date": 0.0,
        },
        item_field_accuracy={"part_no": 0.8, "material": 0.0, "num_pieces": 0.8},
        by_source_kind={},
        error_tag_counts={},
        latency_warm_p50_ms=20.0,
        latency_warm_p95_ms=20.0,
        mean_timings_ms={
            "preprocess": 0.0,
            "load": 0.0,
            "infer": 0.0,
            "parse_validate": 0.0,
            "total": 0.0,
        },
        token_totals={"prompt": 10, "output": 100},
        mean_tokens_per_sec=None,
        needs_review_count=0,
    )


def _candidate(hash12: str, summary: RunSummary) -> CandidateEval:
    """Wrap an aggregate result in a valid prompt reference."""
    return CandidateEval(PromptVersion("base", hash12, "text"), summary)


def _state() -> LoopState:
    """Build a fresh persisted loop state."""
    return LoopState(
        started_at_utc="2026-07-24T00:00:00+00:00",
        candidates_tried=0,
        consecutive_no_improve=0,
        consecutive_errors=0,
        best_exact_match_rate=0.5,
        baseline_prompt_hash="a" * 12,
    )


def test_three_execution_errors_reach_the_error_stop_reason() -> None:
    """Errors must not be masked by the no-improvement stop condition."""
    state = _state()
    for _ in range(MAX_CONSECUTIVE_ERRORS):
        state = update_after_error(state)

    assert should_stop(state, now_utc=datetime(2026, 7, 24, tzinfo=UTC)) is StopReason.ERRORS


def test_candidate_update_advances_the_best_rate_and_can_reach_perfect() -> None:
    """The public update path must be able to trigger the perfect stop."""
    state = update_after_candidate(
        _state(),
        AdoptionDecision(True, ()),
        exact_match_rate=1.0,
    )

    assert state.best_exact_match_rate == 1.0
    assert should_stop(state, now_utc=datetime(2026, 7, 24, tzinfo=UTC)) is StopReason.PERFECT
    lower = update_after_candidate(
        state,
        AdoptionDecision(False, ()),
        exact_match_rate=0.5,
    )
    assert lower.best_exact_match_rate == 1.0
    with pytest.raises(ValueError, match="finite"):
        update_after_candidate(
            state,
            AdoptionDecision(False, ()),
            exact_match_rate=float("nan"),
        )


def test_pick_best_rejects_incomparable_or_nonfinite_aggregates() -> None:
    """Blind selection must remain a total ordering over one fixed split."""
    with pytest.raises(ValueError, match="same number"):
        pick_best(
            [
                _candidate("a" * 12, _summary(n_docs=10)),
                _candidate("b" * 12, _summary(n_docs=20)),
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        pick_best([_candidate("a" * 12, _summary(order_no=float("nan")))])


def test_save_revalidates_constructed_state(tmp_path: Path) -> None:
    """Low-level Pydantic construction must not persist non-standard JSON."""
    invalid = LoopState.model_construct(
        started_at_utc="2026-07-24T00:00:00+00:00",
        candidates_tried=0,
        consecutive_no_improve=0,
        consecutive_errors=0,
        best_exact_match_rate=float("nan"),
        baseline_prompt_hash="a" * 12,
    )

    with pytest.raises(ValueError, match="cannot save loop state"):
        save_loop_state(tmp_path / "state.json", invalid)


def test_save_rejects_an_aliased_parent(tmp_path: Path) -> None:
    """State publication must not escape through a directory alias."""
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    try:
        os.symlink(real_parent, alias, target_is_directory=True)
    except NotImplementedError, OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ValueError, match="symlink or reparse alias"):
        save_loop_state(alias / "state.json", _state())
    assert not (real_parent / "state.json").exists()
