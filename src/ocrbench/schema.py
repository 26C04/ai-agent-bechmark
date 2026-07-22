"""Validated data contract for extracted production instructions."""

import re
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LineItem(BaseModel):
    """One extracted line item, preserving unreadable values as ``None``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    part_no: str | None
    material: str | None
    num_pieces: Annotated[int, Field(ge=1)] | None


class OrderDocument(BaseModel):
    """Structured extraction result for one production instruction."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_name: str | None
    order_no: str | None
    delivery_date: str | None
    items: list[LineItem]

    @field_validator("delivery_date")
    @classmethod
    def validate_delivery_date(cls, value: str | None) -> str | None:
        """Accept only real ISO calendar dates in ``YYYY-MM-DD`` form."""
        if value is None:
            return None
        if _ISO_DATE_PATTERN.fullmatch(value) is None:
            raise ValueError("delivery_date must use YYYY-MM-DD format")
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("delivery_date must be a real calendar date") from error
        return value


def ollama_format_schema() -> dict[str, Any]:
    """Return the JSON Schema supplied as Ollama's ``format`` argument."""
    schema = OrderDocument.model_json_schema()
    line_item_schema = schema["$defs"]["LineItem"]

    assert schema.get("additionalProperties") is False
    assert set(schema.get("required", ())) == {
        "customer_name",
        "order_no",
        "delivery_date",
        "items",
    }
    assert line_item_schema.get("additionalProperties") is False
    assert set(line_item_schema.get("required", ())) == {
        "part_no",
        "material",
        "num_pieces",
    }

    return schema


def compute_needs_review(doc: OrderDocument | None) -> bool:
    """Return whether missing required values or failed validation need review."""
    if doc is None:
        return True
    if doc.customer_name is None or doc.order_no is None or doc.delivery_date is None:
        return True
    if not doc.items:
        return True
    return any(
        item.part_no is None or item.material is None or item.num_pieces is None
        for item in doc.items
    )
