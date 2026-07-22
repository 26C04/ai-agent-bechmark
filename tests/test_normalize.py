"""Tests for deliberately narrow comparison and GT normalizers."""

from collections import Counter

import pytest

from ocrbench.normalize import (
    UNIT_SUFFIXES,
    comparable_header,
    comparable_items,
    normalize_delivery_date,
    normalize_num_pieces,
    strip_outer_whitespace,
)
from ocrbench.schema import LineItem, OrderDocument


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  A 1  ", "A 1"),
        ("\u3000品番\u3000", "品番"),
        ("\t\n A \r\n", "A"),
        (" A\u3000B ", "A\u3000B"),
    ],
)
def test_strip_outer_whitespace(value: str, expected: str) -> None:
    assert strip_outer_whitespace(value) == expected


def test_comparable_header_only_trims_optional_strings() -> None:
    document = _document(customer_name=" 顧客 ", order_no=None)

    assert comparable_header(document) == ("顧客", None, "2026-07-01")


def test_comparable_items_is_an_order_independent_multiset() -> None:
    first = _document(
        items=[_item(" A ", "鉄", 10), _item("B", " 樹脂 ", 20), _item(" A ", "鉄", 10)]
    )
    reordered = _document(
        items=[_item("B", " 樹脂 ", 20), _item("A", "鉄", 10), _item("A", "鉄", 10)]
    )
    missing_duplicate = _document(items=[_item("A", "鉄", 10), _item("B", "樹脂", 20)])

    assert comparable_items(first) == comparable_items(reordered)
    assert comparable_items(first) != comparable_items(missing_duplicate)
    assert comparable_items(first) == Counter({("A", "鉄", 10): 2, ("B", "樹脂", 20): 1})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-07-01", "2026-07-01"),
        ("2026-7-1", "2026-07-01"),
        ("2026/07/01", "2026-07-01"),
        ("2026/7/1", "2026-07-01"),
        ("2026.07.01", "2026-07-01"),
        ("R8.7.1", None),
        ("7/1", None),
        ("2026-13-01", None),
        ("20260701", None),
        ("2026.7.1", None),
        ("2026-7-01", None),
        ("2026-07-1", None),
        ("2026/7/01", None),
        ("2026/07/1", None),
        (" 2026-07-01 ", None),
    ],
)
def test_normalize_delivery_date(value: str, expected: str | None) -> None:
    assert normalize_delivery_date(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (True, None),
        (0, None),
        (-1, None),
        ("1,000個", 1000),
        ("\uff11\uff10\uff10 枚", 100),
        (" 250 pcs", 250),
        ("0", None),
        ("約100", None),
        ("10個入", 10),
        ("10PCS", 10),
        ("10pc", 10),
        ("\uff11\uff0c\uff10\uff10\uff10 本", 1000),
        ("9" * 5000, None),
    ],
)
def test_normalize_num_pieces(value: int | str, expected: int | None) -> None:
    assert normalize_num_pieces(value) == expected


def test_unit_suffixes_are_public_and_longest_overlapping_unit_comes_first() -> None:
    assert set(UNIT_SUFFIXES) == {"個", "枚", "本", "pcs", "PCS", "pc", "個入"}
    assert UNIT_SUFFIXES.index("個入") < UNIT_SUFFIXES.index("個")


def test_comparison_normalization_preserves_prohibited_variants() -> None:
    assert strip_outer_whitespace("A B") != strip_outer_whitespace("AB")
    assert strip_outer_whitespace("\uff21\uff22\uff23") != strip_outer_whitespace("ABC")
    assert strip_outer_whitespace("abc") != strip_outer_whitespace("ABC")
    assert strip_outer_whitespace("A-1") != strip_outer_whitespace("A\u22121")


def _item(part_no: str | None, material: str | None, num_pieces: int | None) -> LineItem:
    return LineItem(part_no=part_no, material=material, num_pieces=num_pieces)


def _document(
    customer_name: str | None = "customer",
    order_no: str | None = "order",
    delivery_date: str | None = "2026-07-01",
    items: list[LineItem] | None = None,
) -> OrderDocument:
    return OrderDocument(
        customer_name=customer_name,
        order_no=order_no,
        delivery_date=delivery_date,
        items=[] if items is None else items,
    )
