"""Regression tests for T07's tuple-string candidate ordering."""

from ocrbench.parsing import ParseResult, ParseStatus
from ocrbench.schema import LineItem, OrderDocument
from ocrbench.scoring import score_document


def test_part_number_candidates_sort_by_complete_detail_tuple_string() -> None:
    gt = _document(
        [
            LineItem(part_no="P", material=None, num_pieces=1),
            LineItem(part_no="P", material="Z", num_pieces=2),
        ]
    )
    pred = _document(
        [
            LineItem(part_no="P", material="A", num_pieces=2),
            LineItem(part_no="P", material="Z", num_pieces=1),
        ]
    )

    score = score_document(
        gt,
        ParseResult(ParseStatus.OK, pred, None, ""),
    )

    assert score.item_field_counts == {
        "part_no": (2, 2),
        "material": (0, 2),
        "num_pieces": (2, 2),
    }
    assert score.error_tags == frozenset({"char_substitution"})


def _document(items: list[LineItem]) -> OrderDocument:
    return OrderDocument(
        customer_name="CUSTOMER",
        order_no="ORDER-1",
        delivery_date="2026-07-01",
        items=items,
    )
