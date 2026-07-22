"""Tests for the extraction schema and review decision."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ocrbench.schema import (
    LineItem,
    OrderDocument,
    compute_needs_review,
    ollama_format_schema,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "public" / "order_document.schema.json"


def valid_document_data() -> dict[str, Any]:
    """Return a complete document represented as unvalidated input data."""
    return {
        "customer_name": "匿名顧客A",
        "order_no": "ORDER-001",
        "delivery_date": "2026-07-21",
        "items": [{"part_no": "PART-001", "material": "SYNTHETIC", "num_pieces": 100}],
    }


def test_complete_document_is_accepted() -> None:
    doc = OrderDocument.model_validate(valid_document_data())

    assert doc.customer_name == "匿名顧客A"
    assert doc.items == [LineItem(part_no="PART-001", material="SYNTHETIC", num_pieces=100)]


def test_nullable_fields_are_accepted() -> None:
    doc = OrderDocument.model_validate(
        {
            "customer_name": None,
            "order_no": None,
            "delivery_date": None,
            "items": [{"part_no": None, "material": None, "num_pieces": None}],
        }
    )

    assert doc.customer_name is None
    assert doc.items[0].num_pieces is None


def test_multiple_items_are_accepted() -> None:
    data = valid_document_data()
    data["items"].append({"part_no": "PART-002", "material": "SYNTHETIC-2", "num_pieces": 1})

    assert len(OrderDocument.model_validate(data).items) == 2


def test_empty_items_are_schema_valid_but_need_review() -> None:
    data = valid_document_data()
    data["items"] = []

    doc = OrderDocument.model_validate(data)

    assert compute_needs_review(doc)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {
            **valid_document_data(),
            "unexpected": "forbidden",
        },
        {
            **valid_document_data(),
            "items": [
                {
                    "part_no": "PART-001",
                    "material": "SYNTHETIC",
                    "num_pieces": 100,
                    "bbox": [0, 0, 1, 1],
                }
            ],
        },
        {key: value for key, value in valid_document_data().items() if key != "order_no"},
        {
            **valid_document_data(),
            "items": [{"part_no": "PART-001", "material": "SYNTHETIC"}],
        },
        {**valid_document_data(), "customer_name": 123},
        {
            **valid_document_data(),
            "items": [{"part_no": "PART-001", "material": "SYNTHETIC", "num_pieces": "100"}],
        },
        {
            **valid_document_data(),
            "items": [{"part_no": "PART-001", "material": "SYNTHETIC", "num_pieces": 0}],
        },
        {
            **valid_document_data(),
            "items": [{"part_no": "PART-001", "material": "SYNTHETIC", "num_pieces": -1}],
        },
        {**valid_document_data(), "delivery_date": "2026-02-30"},
        {**valid_document_data(), "delivery_date": "2026/07/01"},
    ],
    ids=[
        "unknown-document-key",
        "unknown-item-key",
        "missing-document-key",
        "missing-item-key",
        "wrong-string-type",
        "numeric-string",
        "zero-pieces",
        "negative-pieces",
        "nonexistent-date",
        "wrong-date-format",
    ],
)
def test_invalid_documents_are_rejected(invalid_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OrderDocument.model_validate(invalid_data)


def test_ollama_schema_matches_golden_file() -> None:
    generated = json.dumps(ollama_format_schema(), sort_keys=True, indent=2) + "\n"

    assert generated == FIXTURE_PATH.read_text(encoding="utf-8")


def test_ollama_schema_forbids_extra_properties_and_requires_every_key() -> None:
    schema = ollama_format_schema()
    item_schema = schema["$defs"]["LineItem"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["customer_name", "order_no", "delivery_date", "items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["required"] == ["part_no", "material", "num_pieces"]


def test_none_document_needs_review() -> None:
    assert compute_needs_review(None)


@pytest.mark.parametrize("header", ["customer_name", "order_no", "delivery_date"])
def test_null_header_needs_review(header: str) -> None:
    data = valid_document_data()
    data[header] = None

    assert compute_needs_review(OrderDocument.model_validate(data))


@pytest.mark.parametrize("field", ["part_no", "material", "num_pieces"])
def test_null_item_field_needs_review(field: str) -> None:
    data = valid_document_data()
    data["items"][0][field] = None

    assert compute_needs_review(OrderDocument.model_validate(data))


def test_complete_document_does_not_need_review() -> None:
    assert not compute_needs_review(OrderDocument.model_validate(valid_document_data()))
