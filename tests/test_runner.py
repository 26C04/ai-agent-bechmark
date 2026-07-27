"""Integration and failure-injection tests for the sequential benchmark runner."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import ocrbench.runner as runner_module
from ocrbench import config
from ocrbench.dataset import Manifest
from ocrbench.metrics import aggregate
from ocrbench.ollama_adapter import ChatResult, GpuPlacement
from ocrbench.parsing import ParseStatus
from ocrbench.preprocess import preprocess
from ocrbench.prompts import PromptVersion
from ocrbench.results import load_run
from ocrbench.runner import WARM_THRESHOLD_NS, RunnerError, run_benchmark
from tests.helpers import make_mini_dataset


@dataclass(frozen=True)
class RecordedCall:
    model: str
    prompt: str
    image_png: bytes
    format_schema: dict[str, Any]
    seed: int
    temperature: float


Event = str | ChatResult | Exception


class RecordingAdapter:
    """T12-local fake that records complete immutable inference inputs."""

    def __init__(
        self,
        events: Sequence[Event],
        *,
        gpu: bool | None = True,
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self.events = list(events)
        self.gpu = gpu
        self.on_call = on_call
        self.calls: list[RecordedCall] = []
        self.metadata_calls: list[str] = []

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        image_png: bytes,
        format_schema: dict[str, Any],
        seed: int,
        temperature: float = 0.0,
    ) -> ChatResult:
        index = len(self.calls)
        self.calls.append(
            RecordedCall(
                model,
                prompt,
                bytes(image_png),
                copy.deepcopy(format_schema),
                seed,
                temperature,
            )
        )
        if self.on_call is not None:
            self.on_call(index)
        if index >= len(self.events):
            raise AssertionError("recording adapter events are exhausted")
        event = self.events[index]
        if isinstance(event, Exception):
            raise event
        if isinstance(event, ChatResult):
            return event
        return _chat(event)

    def resolve_digest(self, model: str) -> str:
        self.metadata_calls.append(f"digest:{model}")
        return "sha256:recording"

    def gpu_placement(self, model: str) -> GpuPlacement | None:
        self.metadata_calls.append(f"gpu:{model}")
        if self.gpu is None:
            return None
        return GpuPlacement(self.gpu, "synthetic placement")

    def server_version(self) -> str:
        self.metadata_calls.append("version")
        return "fake-1.0"


def _chat(
    content: str,
    *,
    prompt_tokens: int | None = 10,
    output_tokens: int | None = 5,
    load_ns: int | None = 1_000_000,
    prompt_eval_ns: int | None = 2_000_000,
    eval_ns: int | None = 1_000_000,
) -> ChatResult:
    return ChatResult(
        content=content,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_ns=4_000_000,
        load_duration_ns=load_ns,
        prompt_eval_duration_ns=prompt_eval_ns,
        eval_duration_ns=eval_ns,
    )


def _prompt() -> PromptVersion:
    text = "Extract the required fields and return only the requested JSON object."
    hash12 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return PromptVersion("base", hash12, text)


def _dataset(tmp_path: Path) -> tuple[Path, Manifest]:
    root = tmp_path / "dataset"
    return root, make_mini_dataset(root)


def _entries(manifest: Manifest, split: str) -> list[Any]:
    return sorted(
        (entry for entry in manifest.docs if entry.split == split),
        key=lambda entry: entry.doc_id,
    )


def _valid_outputs(root: Path, manifest: Manifest, split: str) -> list[str]:
    return [
        (root / "gt" / f"{entry.doc_id}.json").read_text(encoding="utf-8")
        for entry in _entries(manifest, split)
    ]


@pytest.fixture(autouse=True)
def _stable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_utc_now",
        lambda: datetime(2026, 7, 24, 1, 2, 3, tzinfo=UTC),
    )
    monkeypatch.setattr(runner_module, "_gpu_info", lambda: "Synthetic GPU, 1.0")


def test_run_round_trip_has_expected_manifest_schema_and_sorted_order(
    tmp_path: Path,
) -> None:
    root, manifest = _dataset(tmp_path)
    adapter = RecordingAdapter(_valid_outputs(root, manifest, "dev"))
    runs_root = tmp_path / "runs"

    run_dir = run_benchmark(
        adapter=adapter,
        data_root=root,
        manifest=manifest,
        split="dev",
        model_tag=config.ALLOWED_MODELS[0],
        prompt=_prompt(),
        runs_root=runs_root,
        run_id="round-trip",
    )
    results = load_run(run_dir)

    assert run_dir == runs_root / "20260724T010203Z_round-trip"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert results.schema_version == 1
    assert results.manifest.model_digest == "sha256:recording"
    assert results.manifest.prompt_hash == _prompt().hash
    assert results.manifest.gpu_fully_loaded is True
    assert results.manifest.participation == "primary"
    assert results.manifest.adapter_kind == "fake"
    assert results.manifest.gpu_info == "Synthetic GPU, 1.0"
    assert [document.doc_id for document in results.documents] == sorted(
        document.doc_id for document in results.documents
    )
    assert all(document.final_status is ParseStatus.OK for document in results.documents)

    expected_image_digests = [
        hashlib.sha256(preprocess(root / "raw" / entry.file).png_bytes).hexdigest()
        for entry in _entries(manifest, "dev")
    ]
    assert [
        hashlib.sha256(call.image_png).hexdigest() for call in adapter.calls
    ] == expected_image_digests


def test_retry_uses_byte_for_byte_identical_inputs_and_succeeds(tmp_path: Path) -> None:
    root, manifest = _dataset(tmp_path)
    valid = _valid_outputs(root, manifest, "selection")
    adapter = RecordingAdapter(["not json", valid[0], valid[1]])

    run_dir = run_benchmark(
        adapter=adapter,
        data_root=root,
        manifest=manifest,
        split="selection",
        model_tag=config.ALLOWED_MODELS[1],
        prompt=_prompt(),
        runs_root=tmp_path / "runs",
        run_id="retry",
    )
    first = load_run(run_dir).documents[0]

    assert first.attempt_count == 2
    assert first.first_attempt_status is ParseStatus.NOT_JSON
    assert first.final_status is ParseStatus.OK
    assert adapter.calls[0] == adapter.calls[1]
    assert adapter.calls[0].seed == config.SEED
    assert adapter.calls[0].temperature == 0.0


def test_two_parse_failures_are_persisted_and_next_document_runs(tmp_path: Path) -> None:
    root, manifest = _dataset(tmp_path)
    valid = _valid_outputs(root, manifest, "selection")
    adapter = RecordingAdapter(["not json", '{"items": []}', valid[1]])

    results = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id="two-failures",
        )
    )

    assert len(results.documents) == 2
    assert results.documents[0].attempt_count == 2
    assert results.documents[0].final_status is ParseStatus.SCHEMA_VIOLATION
    assert results.documents[0].prediction is None
    assert results.documents[1].final_status is ParseStatus.OK


@pytest.mark.parametrize("failure_at", [0, 1], ids=["first-attempt", "retry"])
def test_adapter_exceptions_are_sanitized_and_run_continues(
    tmp_path: Path,
    failure_at: int,
) -> None:
    root, manifest = _dataset(tmp_path)
    valid = _valid_outputs(root, manifest, "selection")
    secret = "SECRET-CUSTOMER-AND-PATH-C:/private/order.png"
    if failure_at == 0:
        events: list[Event] = [RuntimeError(secret), valid[1]]
    else:
        events = ["not json", RuntimeError(secret), valid[1]]
    adapter = RecordingAdapter(events)

    results = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id=f"adapter-failure-{failure_at}",
        )
    )
    failed = results.documents[0]

    assert failed.final_status is ParseStatus.NOT_JSON
    assert failed.prediction is None
    assert failed.error is not None
    assert secret not in failed.error
    assert failed.tokens == {"prompt": None, "output": None}
    assert failed.tokens_per_sec is None
    assert failed.warm is False
    assert results.documents[1].final_status is ParseStatus.OK


def test_timing_token_and_warm_aggregation_uses_all_retry_attempts(
    tmp_path: Path,
) -> None:
    root, manifest = _dataset(tmp_path)
    valid = _valid_outputs(root, manifest, "selection")
    first = _chat(
        "not json",
        prompt_tokens=10,
        output_tokens=4,
        load_ns=2_000_000,
        prompt_eval_ns=3_000_000,
        eval_ns=1_000_000,
    )
    retry = _chat(
        valid[0],
        prompt_tokens=20,
        output_tokens=6,
        load_ns=4_000_000,
        prompt_eval_ns=5_000_000,
        eval_ns=2_000_000,
    )
    adapter = RecordingAdapter([first, retry, valid[1]])

    results = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id="timings",
        )
    )
    document = results.documents[0]

    assert document.timings_ms["load"] == pytest.approx(6.0)
    assert document.timings_ms["infer"] == pytest.approx(11.0)
    assert document.timings_ms["total"] >= document.timings_ms["preprocess"]
    assert document.timings_ms["parse_validate"] >= 0.0
    assert document.tokens == {"prompt": 30, "output": 10}
    assert document.tokens_per_sec == pytest.approx(10 / 0.003)
    assert document.warm is True


def test_mixed_missing_metadata_propagates_unknown_tokens_and_cold_state(
    tmp_path: Path,
) -> None:
    root, manifest = _dataset(tmp_path)
    valid = _valid_outputs(root, manifest, "selection")
    first = _chat("not json", prompt_tokens=10, output_tokens=4)
    retry = _chat(
        valid[0],
        prompt_tokens=20,
        output_tokens=None,
        load_ns=WARM_THRESHOLD_NS,
    )
    adapter = RecordingAdapter([first, retry, valid[1]])

    document = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id="missing-metadata",
        )
    ).documents[0]

    assert document.tokens == {"prompt": 30, "output": None}
    assert document.tokens_per_sec is None
    assert document.warm is False


@pytest.mark.parametrize("gpu", [False, None])
def test_gpu_is_primary_only_when_explicitly_true(
    tmp_path: Path,
    gpu: bool | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, manifest = _dataset(tmp_path)
    adapter = RecordingAdapter(_valid_outputs(root, manifest, "selection"), gpu=gpu)

    results = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id=f"gpu-{gpu}",
        )
    )

    assert results.manifest.gpu_fully_loaded is gpu
    assert results.manifest.participation == "reference"
    assert "reference" in caplog.text


def test_disallowed_model_fails_before_dataset_or_adapter_side_effects(
    tmp_path: Path,
) -> None:
    adapter = RecordingAdapter([])
    runs_root = tmp_path / "runs"

    with pytest.raises(RunnerError, match="not allowed"):
        run_benchmark(
            adapter=adapter,
            data_root=tmp_path / "does-not-exist",
            manifest=Manifest(schema_version=1, docs=[]),
            split="dev",
            model_tag="unapproved:latest",
            prompt=_prompt(),
            runs_root=runs_root,
        )

    assert adapter.calls == []
    assert adapter.metadata_calls == []
    assert not runs_root.exists()


def test_supplied_manifest_must_match_disk_before_metadata(tmp_path: Path) -> None:
    root, manifest = _dataset(tmp_path)
    adapter = RecordingAdapter([])
    different = Manifest(schema_version=1, docs=list(reversed(manifest.docs)))

    with pytest.raises(RunnerError, match="does not match"):
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=different,
            split="dev",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
        )

    assert adapter.metadata_calls == []


def test_invalid_direct_prompt_fails_before_metadata(tmp_path: Path) -> None:
    root, manifest = _dataset(tmp_path)
    text = "Synthetic prompt"
    invalid = PromptVersion(
        "../unsafe",
        hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        text,
    )
    adapter = RecordingAdapter([])

    with pytest.raises(RunnerError, match="T13 registry"):
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="dev",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=invalid,
            runs_root=tmp_path / "runs",
        )

    assert adapter.metadata_calls == []
    assert adapter.calls == []


def test_output_collision_fails_before_metadata_and_does_not_overwrite(
    tmp_path: Path,
) -> None:
    root, manifest = _dataset(tmp_path)
    runs_root = tmp_path / "runs"
    target = runs_root / "20260724T010203Z_collision"
    target.mkdir(parents=True)
    marker = target / "owned.txt"
    marker.write_text("existing", encoding="utf-8")
    adapter = RecordingAdapter(_valid_outputs(root, manifest, "dev"))

    with pytest.raises(RunnerError, match="already exists"):
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="dev",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=runs_root,
            run_id="collision",
        )

    assert marker.read_text(encoding="utf-8") == "existing"
    assert adapter.calls == []
    assert adapter.metadata_calls == []


def test_dataset_mutation_aborts_before_any_run_artifact_is_published(
    tmp_path: Path,
) -> None:
    root, manifest = _dataset(tmp_path)
    entry = _entries(manifest, "selection")[0]

    def mutate(index: int) -> None:
        if index == 0:
            path = root / "raw" / entry.file
            path.write_bytes(path.read_bytes() + b"mutated")

    adapter = RecordingAdapter(
        _valid_outputs(root, manifest, "selection"),
        on_call=mutate,
    )
    runs_root = tmp_path / "runs"

    with pytest.raises(RunnerError, match="Dataset changed"):
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=runs_root,
            run_id="mutated",
        )

    assert not runs_root.exists()


def test_document_results_feed_directly_into_aggregate(tmp_path: Path) -> None:
    root, manifest = _dataset(tmp_path)
    adapter = RecordingAdapter(_valid_outputs(root, manifest, "selection"))
    results = load_run(
        run_benchmark(
            adapter=adapter,
            data_root=root,
            manifest=manifest,
            split="selection",
            model_tag=config.ALLOWED_MODELS[0],
            prompt=_prompt(),
            runs_root=tmp_path / "runs",
            run_id="aggregate",
        )
    )

    summary = aggregate(results.documents)
    assert summary.n_docs == 2
    assert summary.exact_match_count == 2
    assert summary.schema_valid_rate == 1.0


def test_publish_rollback_removes_target_after_staged_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_unlink = Path.unlink
    staged_unlinks = 0

    def fail_second_staged_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal staged_unlinks
        if path.parent.name.startswith(".ocrbench-run-") and path.name in {
            "manifest.json",
            "results.json",
        }:
            staged_unlinks += 1
            if staged_unlinks == 2:
                raise OSError("injected staged unlink failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_second_staged_unlink)

    with pytest.raises(RunnerError, match="published safely"):
        runner_module._publish_run(
            runs_root=tmp_path / "runs",
            directory_name="20260724T010203Z_injected",
            manifest_payload=b"manifest",
            results_payload=b"results",
        )

    assert not (tmp_path / "runs" / "20260724T010203Z_injected").exists()
    assert not list((tmp_path / "runs").glob(".ocrbench-run-*.tmp"))


def test_publish_rollback_removes_target_when_final_staging_rmdir_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_rmdir = Path.rmdir
    injected = False

    def fail_first_staging_rmdir(path: Path) -> None:
        nonlocal injected
        if path.name.startswith(".ocrbench-run-") and not injected:
            injected = True
            raise OSError("injected staging rmdir failure")
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_first_staging_rmdir)

    with pytest.raises(RunnerError, match="published safely"):
        runner_module._publish_run(
            runs_root=tmp_path / "runs",
            directory_name="20260724T010203Z_rmdir",
            manifest_payload=b"manifest",
            results_payload=b"results",
        )

    assert not (tmp_path / "runs" / "20260724T010203Z_rmdir").exists()
    assert not list((tmp_path / "runs").glob(".ocrbench-run-*.tmp"))


def test_publish_never_overwrites_a_racing_target_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_link = os.link

    def inject_racer(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if destination.name == "results.json":
            destination.write_bytes(b"racer")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", inject_racer)

    with pytest.raises(RunnerError):
        runner_module._publish_run(
            runs_root=tmp_path / "runs",
            directory_name="20260724T010203Z_race",
            manifest_payload=b"manifest",
            results_payload=b"results",
        )

    target = tmp_path / "runs" / "20260724T010203Z_race"
    assert (target / "results.json").read_bytes() == b"racer"
    assert not (target / "manifest.json").exists()
