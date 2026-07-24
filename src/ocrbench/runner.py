"""Sequential, deterministic OCR benchmark execution and safe publication."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import re
import stat
import subprocess
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Final, Literal, cast

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.dataset import (
    DatasetError,
    DocEntry,
    Manifest,
    dataset_fingerprint,
    load_manifest,
)
from ocrbench.ollama_adapter import ChatResult, OllamaAdapter, RealOllamaAdapter
from ocrbench.parsing import ParseResult, ParseStatus, parse_model_output
from ocrbench.preprocess import PREPROCESS_VERSION, PreprocessError, preprocess
from ocrbench.prompts import PromptVersion
from ocrbench.results import DocumentResult, RunManifest, RunResults, serialize_score
from ocrbench.schema import OrderDocument, compute_needs_review, ollama_format_schema
from ocrbench.scoring import score_document

WARM_THRESHOLD_NS: Final[int] = 1_000_000_000
_TEMPERATURE: Final[float] = 0.0
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PROMPT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_LOGGER = logging.getLogger(__name__)


class RunnerError(OcrBenchError):
    """Raised when a benchmark run cannot be executed or safely published."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or path.is_junction():
        return True
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and getattr(details, "st_file_attributes", 0) & reparse_flag)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_ground_truth(data_root: Path, entry: DocEntry) -> OrderDocument:
    """Read one anonymous GT file without following an alias or exposing its values."""
    gt_directory = data_root / "gt"
    path = gt_directory / f"{entry.doc_id}.json"
    try:
        if _is_link_or_reparse(gt_directory) or _is_link_or_reparse(path):
            raise RunnerError(f"Ground truth is unsafe for {entry.doc_id}")
        root = gt_directory.resolve(strict=True)
        resolved = path.resolve(strict=True)
        details = path.stat(follow_symlinks=False)
        if resolved.parent != root or not stat.S_ISREG(details.st_mode):
            raise RunnerError(f"Ground truth is unsafe for {entry.doc_id}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
        return OrderDocument.model_validate(value)
    except RunnerError:
        raise
    except OSError, UnicodeError, json.JSONDecodeError, ValueError:
        raise RunnerError(f"Ground truth is invalid for {entry.doc_id}") from None


def _validate_prompt(prompt: PromptVersion) -> PromptVersion:
    if (
        not isinstance(prompt.name, str)
        or not isinstance(prompt.hash, str)
        or not isinstance(prompt.text, str)
    ):
        raise RunnerError("Prompt version fields must be strings")
    try:
        expected_hash = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()[:12]
    except UnicodeEncodeError:
        raise RunnerError("Prompt text must be valid UTF-8") from None
    if prompt.hash != expected_hash:
        raise RunnerError("Prompt content does not match its content hash")
    if _PROMPT_NAME_PATTERN.fullmatch(prompt.name) is None:
        raise RunnerError("Prompt name does not match the T13 registry contract")
    return PromptVersion(prompt.name, prompt.hash, prompt.text)


def _validated_run_id(value: str) -> str:
    if not isinstance(value, str) or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise RunnerError("run_id is unsafe or invalid")
    return value


def _validate_inputs(
    *,
    data_root: Path,
    supplied_manifest: Manifest,
    split: str,
    prompt: PromptVersion,
) -> tuple[Manifest, PromptVersion, str]:
    if split not in {"dev", "selection", "final"}:
        raise RunnerError(f"Unsupported split: {split!r}")
    validated_prompt = _validate_prompt(prompt)
    try:
        manifest = Manifest.model_validate(supplied_manifest.model_dump(mode="python"))
    except AttributeError, ValueError:
        raise RunnerError("Supplied dataset manifest is invalid") from None
    try:
        on_disk = load_manifest(data_root)
    except DatasetError:
        raise RunnerError("On-disk dataset validation failed") from None
    if on_disk != manifest:
        raise RunnerError("Supplied manifest does not match the validated on-disk manifest")
    if not any(entry.split == split for entry in manifest.docs):
        raise RunnerError(f"Dataset has no documents in split {split!r}")
    try:
        fingerprint = dataset_fingerprint(data_root, manifest)
    except DatasetError:
        raise RunnerError("Dataset fingerprinting failed") from None
    return manifest, validated_prompt, fingerprint


def _check_output_path(runs_root: Path, directory_name: str) -> None:
    """Reject existing or aliased output paths before metadata or inference calls."""
    root = Path(runs_root)
    try:
        if root.exists() or root.is_symlink():
            if _is_link_or_reparse(root):
                raise RunnerError("Runs root must not be a symlink or reparse alias")
            if not stat.S_ISDIR(root.stat(follow_symlinks=False).st_mode):
                raise RunnerError("Runs root is not a directory")
            target = root / directory_name
            if target.exists() or target.is_symlink():
                raise RunnerError("Run output already exists; no inference was started")
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("Cannot safely inspect the run output path") from None


def _gpu_info() -> str | None:
    """Return best-effort NVIDIA name/driver metadata without surfacing probe errors."""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creation_flags,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(lines)[:500] if lines else None


def _adapter_kind(adapter: OllamaAdapter) -> Literal["real", "fake"]:
    return "real" if isinstance(adapter, RealOllamaAdapter) else "fake"


def _metadata(
    adapter: OllamaAdapter,
    model_tag: str,
) -> tuple[str, str, bool | None]:
    try:
        digest = adapter.resolve_digest(model_tag)
    except Exception:
        raise RunnerError("Adapter failed to resolve the allowed model digest") from None
    try:
        version = adapter.server_version()
    except Exception:
        raise RunnerError("Adapter failed to report its server version") from None
    try:
        placement = adapter.gpu_placement(model_tag)
    except Exception:
        raise RunnerError("Adapter failed to report model GPU placement") from None
    fully_loaded = placement.fully_on_gpu if placement is not None else None
    if fully_loaded is not True:
        _LOGGER.warning(
            "Model %s is not explicitly confirmed as fully GPU-loaded; "
            "the run will be marked reference",
            model_tag,
        )
    return digest, version, fully_loaded


def _safe_adapter_error() -> str:
    return "Adapter inference failed; exception details were intentionally suppressed"


def _synthetic_adapter_failure() -> ParseResult:
    return ParseResult(
        status=ParseStatus.NOT_JSON,
        document=None,
        error="adapter inference failed",
        raw_output="",
    )


def _sum_optional(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(cast(int, value) for value in values)


def _tokens_per_second(results: list[ChatResult]) -> float | None:
    if not results:
        return None
    if any(result.output_tokens is None or result.eval_duration_ns is None for result in results):
        return None
    output_tokens = sum(cast(int, result.output_tokens) for result in results)
    eval_duration_ns = sum(cast(int, result.eval_duration_ns) for result in results)
    if eval_duration_ns <= 0:
        return None
    return output_tokens / (eval_duration_ns / 1_000_000_000)


def _duration_sum_ms(values: list[int | None]) -> float:
    return sum(value for value in values if value is not None) / 1_000_000


def _document_result(
    *,
    adapter: OllamaAdapter,
    data_root: Path,
    entry: DocEntry,
    model_tag: str,
    prompt: PromptVersion,
) -> DocumentResult:
    ground_truth = _load_ground_truth(data_root, entry)
    total_started = perf_counter_ns()
    try:
        prepared = preprocess(data_root / "raw" / entry.file)
    except PreprocessError:
        raise RunnerError(f"Preprocessing failed for {entry.doc_id}") from None

    response_schema = ollama_format_schema()
    chat_results: list[ChatResult] = []
    raw_outputs: list[str] = []
    attempt_count = 1
    parse_started: int | None = None
    adapter_error: str | None = None

    def call_adapter() -> ChatResult:
        return adapter.generate_structured(
            model=model_tag,
            prompt=prompt.text,
            image_png=prepared.png_bytes,
            format_schema=copy.deepcopy(response_schema),
            seed=config.SEED,
            temperature=_TEMPERATURE,
        )

    try:
        first_chat = call_adapter()
    except Exception:
        first_parse = _synthetic_adapter_failure()
        final_parse = first_parse
        adapter_error = _safe_adapter_error()
    else:
        chat_results.append(first_chat)
        raw_outputs.append(first_chat.content)
        parse_started = perf_counter_ns()
        first_parse = parse_model_output(first_chat.content)
        final_parse = first_parse
        if first_parse.status in {
            ParseStatus.NOT_JSON,
            ParseStatus.SCHEMA_VIOLATION,
        }:
            attempt_count = 2
            try:
                retry_chat = call_adapter()
            except Exception:
                final_parse = _synthetic_adapter_failure()
                adapter_error = _safe_adapter_error()
            else:
                chat_results.append(retry_chat)
                raw_outputs.append(retry_chat.content)
                final_parse = parse_model_output(retry_chat.content)

    parse_finished = perf_counter_ns()
    total_finished = parse_finished
    score = score_document(ground_truth, final_parse)
    load_values = [result.load_duration_ns for result in chat_results]
    warm = (
        len(chat_results) == attempt_count
        and all(value is not None for value in load_values)
        and bool(load_values)
        and max(cast(int, value) for value in load_values) < WARM_THRESHOLD_NS
    )
    prediction = (
        final_parse.document.model_dump(mode="json") if final_parse.document is not None else None
    )
    prompt_tokens = (
        None
        if adapter_error is not None
        else _sum_optional([result.prompt_tokens for result in chat_results])
    )
    output_tokens = (
        None
        if adapter_error is not None
        else _sum_optional([result.output_tokens for result in chat_results])
    )
    infer_ns = [
        (result.prompt_eval_duration_ns or 0) + (result.eval_duration_ns or 0)
        for result in chat_results
    ]
    parse_validate_ms = (
        (parse_finished - parse_started) / 1_000_000 if parse_started is not None else 0.0
    )
    return DocumentResult(
        doc_id=entry.doc_id,
        source_kind=entry.source_kind,
        first_attempt_status=first_parse.status,
        final_status=final_parse.status,
        attempt_count=attempt_count,
        prediction=prediction,
        raw_outputs=raw_outputs,
        score=serialize_score(score),
        needs_review=compute_needs_review(final_parse.document),
        warm=warm,
        timings_ms={
            "preprocess": prepared.duration_ms,
            "load": _duration_sum_ms(load_values),
            "infer": sum(infer_ns) / 1_000_000,
            "parse_validate": parse_validate_ms,
            "total": (total_finished - total_started) / 1_000_000,
        },
        tokens={"prompt": prompt_tokens, "output": output_tokens},
        tokens_per_sec=(None if adapter_error is not None else _tokens_per_second(chat_results)),
        error=adapter_error,
    )


def _revalidate_dataset(data_root: Path, manifest: Manifest, fingerprint: str) -> None:
    try:
        on_disk = load_manifest(data_root)
        current_fingerprint = dataset_fingerprint(data_root, manifest)
    except DatasetError:
        raise RunnerError("Dataset changed or became invalid during benchmark execution") from None
    if on_disk != manifest or current_fingerprint != fingerprint:
        raise RunnerError("Dataset changed during benchmark execution; results were not published")


def _prepare_runs_root(runs_root: Path) -> Path:
    root = Path(runs_root)
    try:
        if root.exists() or root.is_symlink():
            if _is_link_or_reparse(root):
                raise RunnerError("Runs root must not be a symlink or reparse alias")
            if not stat.S_ISDIR(root.stat(follow_symlinks=False).st_mode):
                raise RunnerError("Runs root is not a directory")
        else:
            root.mkdir(parents=True)
        if _is_link_or_reparse(root):
            raise RunnerError("Runs root must not be a symlink or reparse alias")
        return root.resolve(strict=True)
    except RunnerError:
        raise
    except OSError:
        raise RunnerError("Cannot create or inspect the runs root") from None


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise


def _file_identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode):
        raise OSError(f"{path.name} is not a regular file")
    return details.st_dev, details.st_ino


def _publish_run(
    *,
    runs_root: Path,
    directory_name: str,
    manifest_payload: bytes,
    results_payload: bytes,
) -> Path:
    root = _prepare_runs_root(runs_root)
    staging = root / f".ocrbench-run-{uuid.uuid4().hex}.tmp"
    target = root / directory_name
    target_created = False
    linked: list[tuple[Path, tuple[int, int]]] = []
    try:
        staging.mkdir(mode=0o700)
        staged_manifest = staging / "manifest.json"
        staged_results = staging / "results.json"
        _write_exclusive(staged_manifest, manifest_payload)
        _write_exclusive(staged_results, results_payload)

        target.mkdir(mode=0o700)
        target_created = True
        for staged_path in (staged_manifest, staged_results):
            published_path = target / staged_path.name
            expected_identity = _file_identity(staged_path)
            os.link(staged_path, published_path, follow_symlinks=False)
            linked.append((published_path, expected_identity))

        staged_manifest.unlink()
        staged_results.unlink()
        staging.rmdir()
        return target
    except FileExistsError:
        reason = "Run output already exists; no files were overwritten"
    except OSError:
        reason = "Run artifacts could not be published safely"

    for published_path, expected_identity in reversed(linked):
        try:
            owned = _file_identity(published_path) == expected_identity
        except OSError:
            owned = False
        if owned:
            with suppress(OSError):
                published_path.unlink()
    if target_created:
        with suppress(OSError):
            target.rmdir()
    if staging.exists() and not _is_link_or_reparse(staging):
        for name in ("manifest.json", "results.json"):
            with suppress(OSError):
                (staging / name).unlink()
        with suppress(OSError):
            staging.rmdir()
    raise RunnerError(reason)


def run_benchmark(
    *,
    adapter: OllamaAdapter,
    data_root: Path,
    manifest: Manifest,
    split: Literal["dev", "selection", "final"],
    model_tag: str,
    prompt: PromptVersion,
    runs_root: Path,
    run_id: str | None = None,
) -> Path:
    """Execute one deterministic run and return its safely published directory."""
    if model_tag not in config.ALLOWED_MODELS:
        raise RunnerError(f"Model tag is not allowed: {model_tag!r}")

    selected_run_id = _validated_run_id(run_id) if run_id is not None else uuid.uuid4().hex[:8]
    validated_manifest, validated_prompt, fingerprint = _validate_inputs(
        data_root=data_root,
        supplied_manifest=manifest,
        split=split,
        prompt=prompt,
    )
    started_at = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    directory_name = f"{started_at}_{selected_run_id}"
    _check_output_path(runs_root, directory_name)

    digest, ollama_version, gpu_fully_loaded = _metadata(adapter, model_tag)
    try:
        run_manifest = RunManifest(
            run_id=selected_run_id,
            started_at_utc=started_at,
            model_tag=model_tag,
            model_digest=digest,
            prompt_name=validated_prompt.name,
            prompt_hash=validated_prompt.hash,
            split=split,
            dataset_fingerprint=fingerprint,
            preprocess_version=PREPROCESS_VERSION,
            ollama_version=ollama_version,
            seed=config.SEED,
            temperature=_TEMPERATURE,
            gpu_fully_loaded=gpu_fully_loaded,
            participation="primary" if gpu_fully_loaded is True else "reference",
            os_info=platform.platform(),
            gpu_info=_gpu_info(),
            adapter_kind=_adapter_kind(adapter),
        )
    except ValueError:
        raise RunnerError("Adapter metadata does not satisfy the run manifest") from None

    entries = sorted(
        (entry for entry in validated_manifest.docs if entry.split == split),
        key=lambda entry: entry.doc_id,
    )
    documents = [
        _document_result(
            adapter=adapter,
            data_root=data_root,
            entry=entry,
            model_tag=model_tag,
            prompt=validated_prompt,
        )
        for entry in entries
    ]
    try:
        results = RunResults(schema_version=1, manifest=run_manifest, documents=documents)
        manifest_payload = (run_manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
        results_payload = (results.model_dump_json(indent=2) + "\n").encode("utf-8")
    except UnicodeEncodeError, ValueError:
        raise RunnerError("Run results could not be serialized safely") from None

    _revalidate_dataset(data_root, validated_manifest, fingerprint)
    return _publish_run(
        runs_root=runs_root,
        directory_name=directory_name,
        manifest_payload=manifest_payload,
        results_payload=results_payload,
    )


__all__ = ["WARM_THRESHOLD_NS", "RunnerError", "run_benchmark"]
