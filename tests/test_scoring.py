"""Tests for deterministic document scoring."""

from collections.abc import Mapping

import pytest

from ocrbench.parsing import ParseResult, ParseStatus
from ocrbench.schema import LineItem, OrderDocument
from ocrbench.scoring import DocumentScore, score_document


def _item(part_no: str | None, material: str | None, count: int | None) -> LineItem:
    return LineItem(part_no=part_no, material=material, num_pieces=count)


def _counts(
    part_no: int, material: int, num_pieces: int, *, total: int = 1
) -> dict[str, tuple[int, int]]:
    return {
        "part_no": (part_no, total),
        "material": (material, total),
        "num_pieces": (num_pieces, total),
    }


@pytest.mark.parametrize("variant", ["same", "reordered", "trimmed"])
def test_exact_match_ignores_order_and_outer_whitespace(variant: str) -> None:
    gt = _gt()
    pred = gt
    if variant == "reordered":
        pred = _doc(items=list(reversed(gt.items)))
    elif variant == "trimmed":
        pred = OrderDocument(
            customer_name="\u3000CUSTOMER ",
            order_no=" ORDER-1\t",
            delivery_date="2026-07-01",
            items=[_item(" A ", "\u3000STEEL", 10), _item("B\n", "RESIN ", 20)],
        )

    score = score_document(gt, _ok(pred))

    assert score.exact_match is True
    assert score.header_correct == _header_bools(True)
    assert score.items_exact is True
    assert score.item_field_counts == _counts(2, 2, 2, total=2)
    assert score.missing_items == score.extra_items == 0
    assert score.error_tags == frozenset()


@pytest.mark.parametrize(
    ("changes", "tag"),
    [
        ({"customer_name": "OTHER"}, "char_substitution"),
        ({"order_no": "ORDER-2"}, "char_substitution"),
        ({"delivery_date": "2026-07-02"}, "date_error"),
    ],
)
def test_header_errors_are_classified(changes: Mapping[str, object], tag: str) -> None:
    pred = _gt().model_copy(update=changes)

    score = score_document(_gt(), _ok(pred))

    assert score.exact_match is False
    assert score.items_exact is True
    assert score.header_correct[next(iter(changes))] is False
    assert score.error_tags == frozenset({tag})


@pytest.mark.parametrize(
    ("gt_items", "pred_items", "missing", "extra", "tag"),
    [
        (
            [_item("A", "STEEL", 10), _item("B", "RESIN", 20)],
            [_item("A", "STEEL", 10)],
            1,
            0,
            "missing_item",
        ),
        (
            [_item("A", "STEEL", 10)],
            [_item("A", "STEEL", 10), _item("B", "RESIN", 20)],
            0,
            1,
            "extra_item",
        ),
        (
            [_item("A", "STEEL", 10), _item("A", "STEEL", 10)],
            [_item("A", "STEEL", 10)],
            1,
            0,
            "missing_item",
        ),
    ],
    ids=["missing", "extra", "duplicate-multiplicity"],
)
def test_item_multiset_preserves_missing_extra_and_duplicate_counts(
    gt_items: list[LineItem],
    pred_items: list[LineItem],
    missing: int,
    extra: int,
    tag: str,
) -> None:
    score = score_document(_doc(items=gt_items), _ok(_doc(items=pred_items)))

    assert score.exact_match is False
    assert score.items_exact is False
    assert (score.missing_items, score.extra_items) == (missing, extra)
    assert score.error_tags == frozenset({tag})


@pytest.mark.parametrize(
    ("pred_item", "tags", "counts"),
    [
        (_item("A", "STEEL", 11), {"count_mismatch"}, _counts(1, 1, 0)),
        (_item("X", "STEEL", 10), {"char_substitution"}, _counts(0, 1, 1)),
        (_item("A", None, None), {"null_field"}, _counts(1, 0, 0)),
    ],
    ids=["quantity", "part-number", "nulls"],
)
def test_paired_item_errors_and_field_counts(
    pred_item: LineItem,
    tags: set[str],
    counts: dict[str, tuple[int, int]],
) -> None:
    gt = _doc(items=[_item("A", "STEEL", 10)])

    score = score_document(gt, _ok(_doc(items=[pred_item])))

    assert score.item_field_counts == counts
    assert score.missing_items == score.extra_items == 0
    assert score.error_tags == frozenset(tags)


def test_predicted_null_header_adds_null_and_date_tags() -> None:
    pred = _gt().model_copy(update={"customer_name": None, "delivery_date": None})

    score = score_document(_gt(), _ok(pred))

    assert score.header_correct == {
        "customer_name": False,
        "order_no": True,
        "delivery_date": False,
    }
    assert score.error_tags == frozenset({"null_field", "date_error"})


def test_empty_prediction_marks_all_items_missing_and_all_cells_incorrect() -> None:
    score = score_document(_gt(), _ok(_doc(items=[])))

    assert score.missing_items == 2
    assert score.extra_items == 0
    assert score.item_field_counts == _counts(0, 0, 0, total=2)
    assert score.error_tags == frozenset({"missing_item"})


@pytest.mark.parametrize("status", [ParseStatus.NOT_JSON, ParseStatus.SCHEMA_VIOLATION])
def test_parse_failure_receives_prescribed_zero_score(status: ParseStatus) -> None:
    result = ParseResult(status=status, document=None, error="invalid", raw_output="bad")

    score = score_document(_gt(), result)

    assert score == DocumentScore(
        exact_match=False,
        header_correct=_header_bools(False),
        items_exact=False,
        item_field_counts=_counts(0, 0, 0, total=2),
        missing_items=2,
        extra_items=0,
        error_tags=frozenset({status.value, "missing_item"}),
    )


def test_three_item_greedy_matching_is_deterministic_and_order_independent() -> None:
    gt = _doc(items=[_item("A", "X", 1), _item("A", "Y", 2), _item("B", "Z", 3)])
    pred = _doc(items=[_item("C", "Z", 3), _item("A", "Y", 1), _item("A", "X", 2)])

    first = score_document(gt, _ok(pred))
    second = score_document(gt, _ok(pred))
    reordered = score_document(
        _doc(items=list(reversed(gt.items))),
        _ok(_doc(items=list(reversed(pred.items)))),
    )

    assert first.item_field_counts == _counts(2, 3, 1, total=3)
    assert first.error_tags == frozenset({"char_substitution", "count_mismatch"})
    assert first == second == reordered


def _doc(*, items: list[LineItem]) -> OrderDocument:
    return OrderDocument(
        customer_name="CUSTOMER",
        order_no="ORDER-1",
        delivery_date="2026-07-01",
        items=items,
    )


def _gt() -> OrderDocument:
    return _doc(items=[_item("A", "STEEL", 10), _item("B", "RESIN", 20)])


def _ok(document: OrderDocument) -> ParseResult:
    return ParseResult(ParseStatus.OK, document, None, "")


def _header_bools(value: bool) -> dict[str, bool]:
    return {"customer_name": value, "order_no": value, "delivery_date": value}
