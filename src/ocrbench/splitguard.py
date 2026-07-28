"""Software guardrails for benchmark split isolation and blind evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.metrics import RunSummary

Split = Literal["dev", "selection", "final"]
_SPLITS = frozenset(("dev", "selection", "final"))


class SplitGuardError(OcrBenchError):
    """Raised when a split isolation or disclosure policy rejects an operation."""


class FinalAccessError(SplitGuardError):
    """Raised when both software confirmations for Final test are not present."""


class DetailReportAccessError(SplitGuardError):
    """Raised when a detail report is requested outside Development."""


def _validated_split(split: str) -> Split:
    if split not in _SPLITS:
        raise SplitGuardError(f"Unsupported benchmark split: {split!r}")
    return split  # type: ignore[return-value]


def resolve_split_dir(split: Split) -> Path:
    """Resolve Development/Selection separately from the ACL-isolated Final test."""
    validated = _validated_split(split)
    if validated == "final":
        return config.final_dir()
    return config.data_dir()


def ensure_final_allowed(*, confirm_final_flag: bool) -> None:
    """Require both explicit software confirmations before a Final test run.

    This check only prevents accidental use. Windows account separation and
    NTFS ACLs are the primary security boundary.
    """
    flag_confirmed = confirm_final_flag is True
    environment_confirmed = os.environ.get("OCRBENCH_ALLOW_FINAL") == "1"
    if not (flag_confirmed and environment_confirmed):
        raise FinalAccessError(
            "Final test access requires both --confirm-final and "
            "OCRBENCH_ALLOW_FINAL=1. This software check only prevents accidental "
            "operation; a separate Windows account and NTFS ACLs are the primary "
            "security boundary."
        )


def console_summary_for(split: str, summary: RunSummary) -> str:
    """Render only console-safe aggregate information for the requested split."""
    validated = _validated_split(split)
    if validated != "dev":
        return (
            f"Completed {summary.n_docs} documents. Run artifacts were written "
            "to the requested output directory."
        )
    return json.dumps(
        asdict(summary), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )


def ensure_detail_report_allowed(split: str) -> None:
    """Allow value-level detail reports only for the Development split."""
    validated = _validated_split(split)
    if validated != "dev":
        raise DetailReportAccessError(
            "Detailed reports are restricted to the Development split to preserve "
            "Selection and Final test blindness."
        )


__all__ = [
    "DetailReportAccessError",
    "FinalAccessError",
    "SplitGuardError",
    "console_summary_for",
    "ensure_detail_report_allowed",
    "ensure_final_allowed",
    "resolve_split_dir",
]
