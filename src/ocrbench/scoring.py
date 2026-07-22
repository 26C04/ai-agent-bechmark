"""Deterministic scoring for one parsed production instruction."""

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from ocrbench.normalize import comparable_header, comparable_items
from ocrbench.parsing import ParseResult, ParseStatus
from ocrbench.schema import OrderDocument

type ItemKey = tuple[str | None, str | None, int | None]
type GroupSortKey = str | tuple[str, int]

_HEADER_FIELDS = ("customer_name", "order_no", "delivery_date")
_ITEM_FIELDS = ("part_no", "material", "num_pieces")


@dataclass(frozen=True)
class DocumentScore:
    """Unit-level score and error classification for one document."""

    exact_match: bool
    header_correct: dict[str, bool]
    items_exact: bool
    item_field_counts: dict[str, tuple[int, int]]
    missing_items: int
    extra_items: int
    error_tags: frozenset[str]


def score_document(gt: OrderDocument, parse_result: ParseResult) -> DocumentScore:
    """Score one prediction using a deterministic greedy item matching heuristic.

    Item order is ignored and duplicate multiplicity is retained. Exact item
    matches are consumed first, followed by matches on part number, matches on
    material and quantity, and finally a sorted positional pairing. The latter
    stages are an ordered heuristic rather than an optimal assignment, but the
    same inputs always produce the same result.
    """
    gt_items = comparable_items(gt)
    if parse_result.status is not ParseStatus.OK:
        return _parse_failure_score(gt_items, parse_result.status)

    pred = parse_result.document
    if pred is None:
        raise ValueError("an OK ParseResult must contain a document")

    gt_header = comparable_header(gt)
    pred_header = comparable_header(pred)
    header_correct = {
        field: gt_value == pred_value
        for field, gt_value, pred_value in zip(_HEADER_FIELDS, gt_header, pred_header, strict=True)
    }

    pred_items = comparable_items(pred)
    items_exact = gt_items == pred_items
    pairs, missing_items, extra_items = _match_items(gt_items, pred_items)

    correct = {field: 0 for field in _ITEM_FIELDS}
    error_tags: set[str] = set()
    _classify_header_errors(gt_header, pred_header, error_tags)
    for gt_item, pred_item in pairs:
        for index, field in enumerate(_ITEM_FIELDS):
            if gt_item[index] == pred_item[index]:
                correct[field] += 1
        _classify_item_errors(gt_item, pred_item, error_tags)

    if missing_items > 0:
        error_tags.add("missing_item")
    if extra_items > 0:
        error_tags.add("extra_item")

    gt_total = sum(gt_items.values())
    item_field_counts = {field: (correct[field], gt_total) for field in _ITEM_FIELDS}
    return DocumentScore(
        exact_match=all(header_correct.values()) and items_exact,
        header_correct=header_correct,
        items_exact=items_exact,
        item_field_counts=item_field_counts,
        missing_items=missing_items,
        extra_items=extra_items,
        error_tags=frozenset(error_tags),
    )


def _parse_failure_score(gt_items: Counter[ItemKey], status: ParseStatus) -> DocumentScore:
    """Return the prescribed zero score for invalid model output."""
    gt_total = sum(gt_items.values())
    error_tags = {status.value}
    if gt_total > 0:
        error_tags.add("missing_item")
    return DocumentScore(
        exact_match=False,
        header_correct={field: False for field in _HEADER_FIELDS},
        items_exact=False,
        item_field_counts={field: (0, gt_total) for field in _ITEM_FIELDS},
        missing_items=gt_total,
        extra_items=0,
        error_tags=frozenset(error_tags),
    )


def _match_items(
    gt_counter: Counter[ItemKey], pred_counter: Counter[ItemKey]
) -> tuple[list[tuple[ItemKey, ItemKey]], int, int]:
    """Pair item multisets in the four deterministic stages required by T07."""
    exact_pairs: list[tuple[ItemKey, ItemKey]] = []
    gt_remainder = gt_counter.copy()
    pred_remainder = pred_counter.copy()
    for item in sorted(gt_counter.keys() & pred_counter.keys(), key=_item_sort_key):
        matched = min(gt_counter[item], pred_counter[item])
        exact_pairs.extend((item, item) for _ in range(matched))
        _consume(gt_remainder, item, matched)
        _consume(pred_remainder, item, matched)

    pairs_by_part, gt_left, pred_left = _pair_by_group(
        _expand(gt_remainder),
        _expand(pred_remainder),
        group_key=lambda item: item[0],
        group_sort_key=_optional_sort_key,
        candidate_sort_key=lambda item: _tuple_sort_key((item[1], item[2])),
    )
    pairs_by_details, gt_left, pred_left = _pair_by_group(
        gt_left,
        pred_left,
        group_key=lambda item: (item[1], item[2]),
        group_sort_key=_tuple_sort_key,
        candidate_sort_key=lambda item: _tuple_sort_key((item[0],)),
    )

    gt_left.sort(key=_item_sort_key)
    pred_left.sort(key=_item_sort_key)
    final_count = min(len(gt_left), len(pred_left))
    final_pairs = list(zip(gt_left[:final_count], pred_left[:final_count], strict=True))
    return (
        exact_pairs + pairs_by_part + pairs_by_details + final_pairs,
        len(gt_left) - final_count,
        len(pred_left) - final_count,
    )


def _pair_by_group[K](
    gt_items: list[ItemKey],
    pred_items: list[ItemKey],
    *,
    group_key: Callable[[ItemKey], K],
    group_sort_key: Callable[[K], GroupSortKey],
    candidate_sort_key: Callable[[ItemKey], str],
) -> tuple[list[tuple[ItemKey, ItemKey]], list[ItemKey], list[ItemKey]]:
    """Greedily pair shared groups and return both sides' unpaired items."""
    gt_groups: dict[K, list[ItemKey]] = defaultdict(list)
    pred_groups: dict[K, list[ItemKey]] = defaultdict(list)
    for item in gt_items:
        gt_groups[group_key(item)].append(item)
    for item in pred_items:
        pred_groups[group_key(item)].append(item)

    pairs: list[tuple[ItemKey, ItemKey]] = []
    gt_left: list[ItemKey] = []
    pred_left: list[ItemKey] = []
    all_groups = sorted(gt_groups.keys() | pred_groups.keys(), key=group_sort_key)
    for group in all_groups:
        gt_group = sorted(gt_groups.get(group, ()), key=candidate_sort_key)
        pred_group = sorted(pred_groups.get(group, ()), key=candidate_sort_key)
        matched = min(len(gt_group), len(pred_group))
        pairs.extend(zip(gt_group[:matched], pred_group[:matched], strict=True))
        gt_left.extend(gt_group[matched:])
        pred_left.extend(pred_group[matched:])
    return pairs, gt_left, pred_left


def _classify_header_errors(
    gt_header: tuple[str | None, str | None, str | None],
    pred_header: tuple[str | None, str | None, str | None],
    error_tags: set[str],
) -> None:
    """Add tags for mismatched header values."""
    for index in (0, 1):
        if gt_header[index] == pred_header[index]:
            continue
        if pred_header[index] is None:
            error_tags.add("null_field")
        elif gt_header[index] is not None:
            error_tags.add("char_substitution")

    if gt_header[2] != pred_header[2]:
        error_tags.add("date_error")
        if pred_header[2] is None:
            error_tags.add("null_field")


def _classify_item_errors(gt_item: ItemKey, pred_item: ItemKey, error_tags: set[str]) -> None:
    """Add tags for mismatched fields in one paired item."""
    for index in (0, 1):
        if gt_item[index] == pred_item[index]:
            continue
        if pred_item[index] is None:
            error_tags.add("null_field")
        elif gt_item[index] is not None:
            error_tags.add("char_substitution")

    if gt_item[2] != pred_item[2]:
        if pred_item[2] is None:
            error_tags.add("null_field")
        elif gt_item[2] is not None:
            error_tags.add("count_mismatch")


def _consume(counter: Counter[ItemKey], item: ItemKey, count: int) -> None:
    """Remove a positive matched count while keeping Counter expansion simple."""
    remaining = counter[item] - count
    if remaining > 0:
        counter[item] = remaining
    else:
        del counter[item]


def _expand(counter: Counter[ItemKey]) -> list[ItemKey]:
    """Expand a multiset into a deterministically sorted duplicate-preserving list."""
    return sorted(counter.elements(), key=_item_sort_key)


def _optional_sort_key(value: str | int | None) -> tuple[str, int]:
    """Sort by string representation with a deterministic type tie-breaker."""
    return (str(value), 0 if value is None else 1)


def _tuple_sort_key(values: tuple[str | int | None, ...]) -> str:
    """Sort by the tuple's complete string representation as required by T07."""
    return str(values)


def _item_sort_key(item: ItemKey) -> str:
    """Return a total, input-order-independent ordering for an item key."""
    return _tuple_sort_key(item)
