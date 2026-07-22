"""Table-driven tests for strict model-output parsing."""

import json
from typing import Any

import pytest

from ocrbench.parsing import ParseStatus, parse_model_output


def _valid_data() -> dict[str, Any]:
    return {
        "customer_name": "匿名顧客A",
        "order_no": "ORDER-001",
        "delivery_date": "2026-07-01",
        "items": [
            {
                "part_no": "PART-001",
                "material": "SYNTHETIC",
                "num_pieces": 100,
            }
        ],
    }


def _valid_json() -> str:
    return json.dumps(_valid_data(), ensure_ascii=False)


@pytest.mark.parametrize(
    "raw",
    [
        f"```json\n{_valid_json()}\n```",
        f"以下が結果です:\n{_valid_json()}",
        f"{_valid_json()}\n以上です",
        f"\ufeff{_valid_json()}",
        "",
        "{",
    ],
    ids=[
        "markdown-fence",
        "leading-explanation",
        "trailing-explanation",
        "bom",
        "empty",
        "truncated",
    ],
)
def test_non_json_output_is_rejected_without_rescue(raw: str) -> None:
    result = parse_model_output(raw)

    assert result.status is ParseStatus.NOT_JSON
    assert result.document is None
    assert result.error is not None
    assert result.raw_output == raw


@pytest.mark.parametrize(
    "value",
    [
        {**_valid_data(), "unexpected": "forbidden"},
        {key: value for key, value in _valid_data().items() if key != "items"},
        {
            **_valid_data(),
            "items": [
                {
                    "part_no": "PART-001",
                    "material": "SYNTHETIC",
                    "num_pieces": "100",
                }
            ],
        },
        {
            **_valid_data(),
            "items": [
                {
                    "part_no": "PART-001",
                    "material": "SYNTHETIC",
                    "num_pieces": 0,
                }
            ],
        },
        {**_valid_data(), "delivery_date": "2026/07/01"},
        [_valid_data()],
    ],
    ids=[
        "unknown-key",
        "missing-items",
        "numeric-string",
        "zero-pieces",
        "invalid-date",
        "top-level-array",
    ],
)
def test_json_schema_violations_are_rejected(value: object) -> None:
    raw = json.dumps(value, ensure_ascii=False)

    result = parse_model_output(raw)

    assert result.status is ParseStatus.SCHEMA_VIOLATION
    assert result.document is None
    assert result.error is not None
    assert result.raw_output == raw


@pytest.mark.parametrize(
    "raw",
    [
        _valid_json(),
        f"\n  {_valid_json()}\n",
        json.dumps(
            {
                "customer_name": None,
                "order_no": None,
                "delivery_date": None,
                "items": [{"part_no": None, "material": None, "num_pieces": None}],
            }
        ),
    ],
    ids=["complete", "json-whitespace", "nullable-fields"],
)
def test_valid_output_is_accepted(raw: str) -> None:
    result = parse_model_output(raw)

    assert result.status is ParseStatus.OK
    assert result.document is not None
    assert result.error is None
    assert result.raw_output == raw


def test_error_summary_is_truncated_to_500_characters() -> None:
    data = _valid_data()
    for index in range(100):
        data[f"unexpected_{index}"] = index

    result = parse_model_output(json.dumps(data))

    assert result.status is ParseStatus.SCHEMA_VIOLATION
    assert result.error is not None
    assert len(result.error) == 500
