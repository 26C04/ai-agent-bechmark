"""Strict parsing and validation of raw model output."""

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError

from ocrbench.schema import OrderDocument

_MAX_ERROR_LENGTH = 500


class ParseStatus(StrEnum):
    """Outcome of parsing and validating one raw model response."""

    OK = "ok"
    NOT_JSON = "not_json"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass(frozen=True)
class ParseResult:
    """A raw model response and its strict validation outcome."""

    status: ParseStatus
    document: OrderDocument | None
    error: str | None
    raw_output: str


def parse_model_output(raw: str) -> ParseResult:
    """Parse raw model output without extracting or repairing JSON."""
    try:
        decoded = json.loads(raw)
    except Exception as error:
        return _failure(ParseStatus.NOT_JSON, error, raw)

    try:
        document = OrderDocument.model_validate(decoded)
    except ValidationError as error:
        return _failure(ParseStatus.SCHEMA_VIOLATION, error, raw)
    except Exception as error:
        return _failure(ParseStatus.SCHEMA_VIOLATION, error, raw)

    return ParseResult(
        status=ParseStatus.OK,
        document=document,
        error=None,
        raw_output=raw,
    )


def _failure(status: ParseStatus, error: Exception, raw: str) -> ParseResult:
    """Build a deterministic failure result with a bounded error summary."""
    return ParseResult(
        status=status,
        document=None,
        error=str(error)[:_MAX_ERROR_LENGTH],
        raw_output=raw,
    )
