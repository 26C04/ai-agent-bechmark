"""Narrow, deterministic normalizers used by benchmark support tooling.

Comparison normalizers deliberately trim only outer Unicode whitespace. In
particular, they do not perform name matching or character folding. The date
and quantity helpers are for creating or revalidating ground truth only;
model output must instead already satisfy the schema's canonical form.
"""

import re
from collections import Counter
from datetime import date

from ocrbench.schema import OrderDocument

UNIT_SUFFIXES: tuple[str, ...] = ("個入", "pcs", "PCS", "pc", "個", "枚", "本")
"""Recognized trailing quantity units for ground-truth preparation."""

_DATE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})$",
        r"^(?P<year>[0-9]{4})-(?P<month>[0-9])-(?P<day>[0-9])$",
        r"^(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2})$",
        r"^(?P<year>[0-9]{4})/(?P<month>[0-9])/(?P<day>[0-9])$",
        r"^(?P<year>[0-9]{4})\.(?P<month>[0-9]{2})\.(?P<day>[0-9]{2})$",
    )
)
_ASCII_DIGITS_PATTERN = re.compile(r"^[0-9]+$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_FULLWIDTH_DIGITS = str.maketrans({0xFF10 + offset: ord("0") + offset for offset in range(10)})


def strip_outer_whitespace(value: str) -> str:
    """Remove only outer Unicode whitespace, preserving every other character.

    This follows :meth:`str.strip`, including full-width space (U+3000).
    """
    return value.strip()


def comparable_header(doc: OrderDocument) -> tuple[str | None, str | None, str | None]:
    """Return the three header fields with only their outer whitespace trimmed."""
    return (
        _trim_optional(doc.customer_name),
        _trim_optional(doc.order_no),
        _trim_optional(doc.delivery_date),
    )


def comparable_items(doc: OrderDocument) -> Counter[tuple[str | None, str | None, int | None]]:
    """Return trimmed line items as an order-independent, duplicate-aware multiset."""
    return Counter(
        (_trim_optional(item.part_no), _trim_optional(item.material), item.num_pieces)
        for item in doc.items
    )


def normalize_delivery_date(value: str) -> str | None:
    """Normalize an explicitly written Gregorian date to ``YYYY-MM-DD``.

    Accepted separators and widths are deliberately limited to the five forms
    in T05. Dates without a year, Japanese eras, impossible dates, and all
    other forms are rejected rather than inferred. This is a ground-truth
    helper; it must not be applied to model output.
    """
    match = None
    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(value)
        if match is not None:
            break
    if match is None:
        return None

    try:
        parsed = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None
    return parsed.isoformat()


def normalize_num_pieces(value: int | str) -> int | None:
    """Normalize an explicitly formatted positive ground-truth quantity.

    Strings may use commas, full-width commas, whitespace, full-width digits,
    and one listed trailing unit. Invalid or non-positive values return
    ``None``. This is a ground-truth helper; model output is schema-validated
    and must not be normalized with it.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None

    without_separators = _WHITESPACE_PATTERN.sub("", value).replace(",", "").replace("\uff0c", "")
    numeric_text = without_separators.translate(_FULLWIDTH_DIGITS)
    for suffix in UNIT_SUFFIXES:
        if numeric_text.endswith(suffix):
            numeric_text = numeric_text[: -len(suffix)]
            break

    if _ASCII_DIGITS_PATTERN.fullmatch(numeric_text) is None:
        return None
    try:
        normalized = int(numeric_text)
    except ValueError:
        return None
    return normalized if normalized >= 1 else None


def _trim_optional(value: str | None) -> str | None:
    """Trim an optional comparison string without changing missing values."""
    return None if value is None else strip_outer_whitespace(value)
