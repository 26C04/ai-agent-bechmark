"""Strict-type regression tests for the extraction schema."""

import pytest
from pydantic import ValidationError

from ocrbench.schema import LineItem


@pytest.mark.parametrize("num_pieces", [True, 100.0], ids=["boolean", "float"])
def test_num_pieces_rejects_non_integer_numeric_values(num_pieces: object) -> None:
    with pytest.raises(ValidationError):
        LineItem.model_validate(
            {"part_no": "PART-001", "material": "SYNTHETIC", "num_pieces": num_pieces}
        )
