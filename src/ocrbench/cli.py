"""Command-line interface for the local OCR benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from pydantic import ValidationError

from ocrbench import __version__, config
from ocrbench.config import ConfigError, OcrBenchError
from ocrbench.dataset import (
    DatasetError,
    build_manifest,
    dataset_fingerprint,
    load_manifest,
    validate_split_counts,
)
from ocrbench.metrics import aggregate, paired_comparison, reproducibility
from ocrbench.ollama_adapter import ModelNotFoundError, create_adapter
from ocrbench.prompts import (
    PromptRegistryError,
    PromptVersion,
    activate,
    add_prompt,
    get_active,
    lint_prompt,
    list_versions,
    rollback,
)
from ocrbench.reporting import write_anonymous_summary, write_detail_report, write_report
from ocrbench.results import ResultsError, RunResults, load_run
from ocrbench.runner import RunnerError, run_benchmark
from ocrbench.selection import (
    AdoptionDecision,
    CandidateEval,
    LoopState,
    apply_adoption,
    is_adoptable,
    load_loop_state,
    pick_best,
    save_loop_state,
    should_stop,
    update_after_candidate,
    update_after_error,
)
from ocrbench.splitguard import (
    Split,
    SplitGuardError,
    console_summary_for,
    ensure_detail_report_allowed,
    ensure_final_allowed,
    resolve_split_dir,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_Handler = Callable[[argparse.Namespace], None]
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EXECUTION_COMPATIBILITY_FIELDS = (
    "preprocess_version",
    "ollama_version",
    "seed",
    "temperature",
    "adapter_kind",
    "gpu_fully_loaded",
    "participation",
)
_ADOPTION_COMPATIBILITY_FIELDS = (
    "model_tag",
    "model_digest",
    "prompt_name",
)
_REPRODUCIBILITY_COMPATIBILITY_FIELDS = (
    *_ADOPTION_COMPATIBILITY_FIELDS,
    "prompt_hash",
)


class _ParserExit(Exception):
    """Carry an argparse help/version exit through the pure ``main`` function."""

    def __init__(self, status: int) -> None:
        self.status = status


class _CliInputError(OcrBenchError):
    """Raised for command-line values or cross-artifact validation failures."""


class _ArgumentParser(argparse.ArgumentParser):
    """Keep argparse testable without allowing it to terminate the process."""

    def error(self, message: str) -> NoReturn:
        raise _CliInputError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        if message:
            self._print_message(message, sys.stdout if status == 0 else sys.stderr)
        raise _ParserExit(status)


def _shared_parser() -> _ArgumentParser:
    parser = _ArgumentParser(add_help=False)
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit deterministic strict JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="re-raise errors with a traceback",
    )
    return parser


def _leaf(
    subparsers: argparse._SubParsersAction[_ArgumentParser],
    name: str,
    *,
    shared: _ArgumentParser,
    help_text: str,
) -> _ArgumentParser:
    return subparsers.add_parser(
        name,
        parents=[shared],
        help=help_text,
        description=help_text,
    )


def _run_id(value: str) -> str:
    """Validate a run ID during parsing, before any handler side effect."""
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line parser."""
    shared = _shared_parser()
    parser = _ArgumentParser(
        prog="ocrbench",
        parents=[shared],
        description="Run deterministic local OCR benchmarks.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    dataset = commands.add_parser(
        "dataset",
        parents=[shared],
        help="build and validate secure dataset manifests",
        description="Build and validate secure dataset manifests.",
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", metavar="COMMAND")
    dataset_build = _leaf(
        dataset_commands,
        "build-manifest",
        shared=shared,
        help_text="build an anonymous manifest from an assignments CSV",
    )
    dataset_build.add_argument("--root", type=Path, required=True, help="dataset root directory")
    dataset_build.add_argument(
        "--assignments", type=Path, required=True, help="split assignments CSV"
    )
    dataset_build.set_defaults(handler=_handle_dataset_build_manifest)

    dataset_validate = _leaf(
        dataset_commands,
        "validate",
        shared=shared,
        help_text="validate manifest contents and optional exact split counts",
    )
    dataset_validate.add_argument("--root", type=Path, required=True, help="dataset root directory")
    dataset_validate.add_argument(
        "--strict-counts",
        action="store_true",
        help="reject counts that differ from the ADR split table",
    )
    dataset_validate.set_defaults(handler=_handle_dataset_validate)

    fingerprint = _leaf(
        dataset_commands,
        "fingerprint",
        shared=shared,
        help_text="calculate the complete dataset fingerprint",
    )
    fingerprint.add_argument("--root", type=Path, required=True, help="dataset root directory")
    fingerprint.set_defaults(handler=_handle_dataset_fingerprint)

    prompt = commands.add_parser(
        "prompt",
        parents=[shared],
        help="manage the append-only prompt registry",
        description="Manage the append-only prompt registry.",
    )
    prompt_commands = prompt.add_subparsers(dest="prompt_command", metavar="COMMAND")
    prompt_add = _leaf(
        prompt_commands,
        "add",
        shared=shared,
        help_text="add an immutable prompt version and show lint warnings",
    )
    prompt_add.add_argument("--name", required=True, help="prompt registry name")
    prompt_add.add_argument("--file", type=Path, required=True, help="UTF-8 prompt text file")
    prompt_add.add_argument("--note", default="", help="audit-history note")
    prompt_add.set_defaults(handler=_handle_prompt_add)

    prompt_list = _leaf(
        prompt_commands,
        "list",
        shared=shared,
        help_text="list all immutable versions in hash order",
    )
    prompt_list.add_argument("--name", required=True, help="prompt registry name")
    prompt_list.set_defaults(handler=_handle_prompt_list)

    prompt_show = _leaf(
        prompt_commands,
        "show",
        shared=shared,
        help_text="show a prompt version, or the active version by default",
    )
    prompt_show.add_argument("--name", required=True, help="prompt registry name")
    prompt_show.add_argument("--hash", dest="prompt_hash", help="12-character prompt hash")
    prompt_show.set_defaults(handler=_handle_prompt_show)

    prompt_activate = _leaf(
        prompt_commands,
        "activate",
        shared=shared,
        help_text="activate an existing immutable prompt version",
    )
    prompt_activate.add_argument("--name", required=True, help="prompt registry name")
    prompt_activate.add_argument("--hash", dest="prompt_hash", required=True)
    prompt_activate.add_argument("--note", default="", help="audit-history note")
    prompt_activate.set_defaults(handler=_handle_prompt_activate)

    prompt_rollback = _leaf(
        prompt_commands,
        "rollback",
        shared=shared,
        help_text="restore the previously active prompt version",
    )
    prompt_rollback.add_argument("--name", required=True, help="prompt registry name")
    prompt_rollback.set_defaults(handler=_handle_prompt_rollback)

    run = _leaf(
        commands,
        "run",
        shared=shared,
        help_text="execute one deterministic benchmark run",
    )
    run.add_argument("--model", required=True, help="allowlisted exact model tag")
    run.add_argument("--split", choices=("dev", "selection", "final"), required=True)
    run.add_argument("--prompt-name", required=True, help="active prompt registry name")
    run.add_argument("--run-id", type=_run_id, help="stable run identifier")
    run.add_argument(
        "--confirm-final",
        action="store_true",
        help="confirm Final access; OCRBENCH_ALLOW_FINAL=1 is also required",
    )
    run.set_defaults(handler=_handle_run)

    report = _leaf(
        commands,
        "report",
        shared=shared,
        help_text="write deterministic run reports",
    )
    report.add_argument("--run-dir", type=Path, required=True, help="completed run directory")
    report.add_argument(
        "--detail",
        action="store_true",
        help="write confidential value-level details (Development only)",
    )
    report.set_defaults(handler=_handle_report)

    export = _leaf(
        commands,
        "export-summary",
        shared=shared,
        help_text="write an anonymous Development summary for an agent",
    )
    export.add_argument("--run-dir", type=Path, required=True, help="completed run directory")
    export.add_argument("--out", type=Path, required=True, help="summary JSON destination")
    export.set_defaults(handler=_handle_export_summary)

    select = commands.add_parser(
        "select",
        parents=[shared],
        help="adopt Development prompts or blindly select a Selection winner",
        description="Adopt Development prompts or blindly select a Selection winner.",
    )
    select_commands = select.add_subparsers(dest="select_command", metavar="COMMAND")
    select_adopt = _leaf(
        select_commands,
        "adopt",
        shared=shared,
        help_text="apply the deterministic Development adoption policy",
    )
    select_adopt.add_argument("--baseline-run", type=Path, required=True)
    select_adopt.add_argument("--candidate-run", type=Path, required=True)
    select_adopt.add_argument("--prompt-name", required=True)
    select_adopt.set_defaults(handler=_handle_select_adopt)

    select_best = _leaf(
        select_commands,
        "pick-best",
        shared=shared,
        help_text="blindly select the strongest matching Selection run",
    )
    select_best.add_argument("--candidate-runs", type=Path, nargs="+", required=True)
    select_best.set_defaults(handler=_handle_select_pick_best)

    loop = commands.add_parser(
        "loop",
        parents=[shared],
        help="persist and evaluate prompt-improvement loop state",
        description="Persist and evaluate prompt-improvement loop state.",
    )
    loop_commands = loop.add_subparsers(dest="loop_command", metavar="COMMAND")
    loop_init = _leaf(
        loop_commands,
        "init",
        shared=shared,
        help_text="initialize a prompt-improvement loop state file",
    )
    loop_init.add_argument("--state", type=Path, required=True)
    loop_init.add_argument("--baseline-prompt-hash", required=True)
    loop_init.add_argument("--best-exact-match-rate", type=float, default=0.0)
    loop_init.set_defaults(handler=_handle_loop_init)

    loop_check = _leaf(
        loop_commands,
        "check",
        shared=shared,
        help_text="evaluate all deterministic loop stopping conditions",
    )
    loop_check.add_argument("--state", type=Path, required=True)
    loop_check.set_defaults(handler=_handle_loop_check)

    loop_record = _leaf(
        loop_commands,
        "record",
        shared=shared,
        help_text="record a candidate decision or an execution error",
    )
    loop_record.add_argument("--state", type=Path, required=True)
    loop_record.add_argument("--baseline-run", type=Path)
    loop_record.add_argument("--candidate-run", type=Path)
    loop_record.add_argument(
        "--error",
        action="store_true",
        help="record a failed candidate execution instead of two Development runs",
    )
    loop_record.set_defaults(handler=_handle_loop_record)

    compare = _leaf(
        commands,
        "compare-runs",
        shared=shared,
        help_text="compare two paired runs or three reproducibility runs",
    )
    compare.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    compare.add_argument(
        "--confirm-final",
        action="store_true",
        help="confirm Final access; OCRBENCH_ALLOW_FINAL=1 is also required",
    )
    compare.set_defaults(handler=_handle_compare_runs)
    return parser


def _handle_dataset_build_manifest(arguments: argparse.Namespace) -> None:
    manifest = build_manifest(arguments.root, arguments.assignments)
    payload = {
        "manifest": str(arguments.root / "manifest.json"),
        "n_docs": len(manifest.docs),
    }
    _emit(arguments, payload, f"Built manifest for {len(manifest.docs)} documents.")


def _handle_dataset_validate(arguments: argparse.Namespace) -> None:
    manifest = load_manifest(arguments.root)
    violations = validate_split_counts(manifest, strict=arguments.strict_counts)
    payload = {
        "n_docs": len(manifest.docs),
        "split_count_warnings": violations,
        "valid": True,
    }
    message = f"Dataset is valid ({len(manifest.docs)} documents)."
    if violations:
        message = "Dataset is structurally valid.\n" + "\n".join(
            f"Warning: {item}" for item in violations
        )
    _emit(arguments, payload, message)


def _handle_dataset_fingerprint(arguments: argparse.Namespace) -> None:
    manifest = load_manifest(arguments.root)
    fingerprint = dataset_fingerprint(arguments.root, manifest)
    _emit(arguments, {"fingerprint": fingerprint}, fingerprint)


def _handle_prompt_add(arguments: argparse.Namespace) -> None:
    try:
        text = arguments.file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _CliInputError("cannot read the prompt text file as UTF-8") from error
    warnings = lint_prompt(text)
    version = add_prompt(
        _prompt_registry_root(),
        arguments.name,
        text,
        note=arguments.note,
    )
    payload = _prompt_payload(version)
    payload["warnings"] = warnings
    human = f"Added prompt {version.name}/{version.hash}."
    if warnings:
        human += "\n" + "\n".join(f"Warning: {warning}" for warning in warnings)
    _emit(arguments, payload, human)


def _handle_prompt_list(arguments: argparse.Namespace) -> None:
    versions = list_versions(_prompt_registry_root(), arguments.name)
    payload = {
        "name": arguments.name,
        "versions": [_prompt_payload(version) for version in versions],
    }
    _emit(arguments, payload, "\n".join(version.hash for version in versions))


def _handle_prompt_show(arguments: argparse.Namespace) -> None:
    if arguments.prompt_hash is None:
        version = get_active(_prompt_registry_root(), arguments.name)
    else:
        version = _registered_prompt(arguments.name, arguments.prompt_hash)
    _emit(arguments, _prompt_payload(version), version.text)


def _handle_prompt_activate(arguments: argparse.Namespace) -> None:
    activate(
        _prompt_registry_root(),
        arguments.name,
        arguments.prompt_hash,
        note=arguments.note,
    )
    payload = {"active_hash": arguments.prompt_hash, "name": arguments.name}
    _emit(arguments, payload, f"Activated prompt {arguments.name}/{arguments.prompt_hash}.")


def _handle_prompt_rollback(arguments: argparse.Namespace) -> None:
    version = rollback(_prompt_registry_root(), arguments.name)
    payload = {"active_hash": version.hash, "name": version.name}
    _emit(arguments, payload, f"Rolled back to prompt {version.name}/{version.hash}.")


def _handle_run(arguments: argparse.Namespace) -> None:
    model_tag = cast(str, arguments.model)
    if model_tag not in config.ALLOWED_MODELS:
        raise _CliInputError(f"model tag is not allowlisted: {model_tag!r}")
    split = cast(Split, arguments.split)
    if split == "final":
        ensure_final_allowed(confirm_final_flag=bool(arguments.confirm_final))
    data_root = resolve_split_dir(split)
    manifest = load_manifest(data_root)
    prompt = get_active(_prompt_registry_root(), arguments.prompt_name)
    adapter = create_adapter()
    run_dir = run_benchmark(
        adapter=adapter,
        data_root=data_root,
        manifest=manifest,
        split=split,
        model_tag=model_tag,
        prompt=prompt,
        runs_root=config.runs_dir(),
        run_id=arguments.run_id,
        confirm_final_flag=bool(arguments.confirm_final),
    )
    results = load_run(run_dir)
    summary = aggregate(results.documents)
    if split == "dev":
        payload: dict[str, Any] = {
            "n_docs": summary.n_docs,
            "run_dir": str(run_dir),
            "split": split,
            "summary": asdict(summary),
        }
        human = f"Run directory: {run_dir}\n{console_summary_for(split, summary)}"
    else:
        payload = {"n_docs": summary.n_docs, "run_dir": str(run_dir), "split": split}
        human = f"{console_summary_for(split, summary)}\nRun directory: {run_dir}"
    _emit(arguments, payload, human)


def _handle_report(arguments: argparse.Namespace) -> None:
    if arguments.detail:
        results = load_run(arguments.run_dir)
        ensure_detail_report_allowed(results.manifest.split)
        data_root = resolve_split_dir(results.manifest.split)
        output = write_detail_report(arguments.run_dir, data_root)
        kind = "detail"
    else:
        output = write_report(arguments.run_dir)
        kind = "aggregate"
    _emit(arguments, {"kind": kind, "report": str(output)}, f"Wrote {kind} report to {output}.")


def _handle_export_summary(arguments: argparse.Namespace) -> None:
    results = load_run(arguments.run_dir)
    if results.manifest.split != "dev":
        raise _CliInputError("agent-facing summary export is restricted to Development runs")
    output = write_anonymous_summary(arguments.run_dir, arguments.out)
    payload = {
        "n_docs": len(results.documents),
        "out": str(output),
        "split": results.manifest.split,
    }
    _emit(arguments, payload, f"Wrote anonymous summary to {output}.")


def _handle_select_adopt(arguments: argparse.Namespace) -> None:
    baseline_results = load_run(arguments.baseline_run)
    candidate_results = load_run(arguments.candidate_run)
    _require_matching_runs((baseline_results, candidate_results), required_split="dev")
    _require_same_manifest_fields(
        (baseline_results, candidate_results),
        _ADOPTION_COMPATIBILITY_FIELDS,
    )
    _require_prompt_name(baseline_results, arguments.prompt_name)
    _require_prompt_name(candidate_results, arguments.prompt_name)
    baseline = CandidateEval(
        _registered_prompt(arguments.prompt_name, baseline_results.manifest.prompt_hash),
        aggregate(baseline_results.documents),
    )
    candidate = CandidateEval(
        _registered_prompt(arguments.prompt_name, candidate_results.manifest.prompt_hash),
        aggregate(candidate_results.documents),
    )
    decision = is_adoptable(baseline, candidate)
    active = apply_adoption(
        _prompt_registry_root(),
        arguments.prompt_name,
        baseline,
        (candidate,),
    )
    payload = {
        "active_prompt_hash": active.hash,
        "adopted": decision.adopted,
        "candidate_prompt_hash": candidate.prompt.hash,
        "reasons": list(decision.reasons),
    }
    outcome = "adopted" if decision.adopted else "not adopted"
    _emit(arguments, payload, f"Candidate {candidate.prompt.hash} was {outcome}.")


def _handle_select_pick_best(arguments: argparse.Namespace) -> None:
    run_dirs = cast(list[Path], arguments.candidate_runs)
    results = [load_run(path) for path in run_dirs]
    _require_matching_runs(results, required_split="selection")
    ordered = sorted(
        zip(run_dirs, results, strict=True),
        key=lambda item: _stable_run_key(item[0], item[1]),
    )
    candidates = [_candidate_from_results(result) for _, result in ordered]
    best = pick_best(candidates)
    selected_index = next(index for index, candidate in enumerate(candidates) if candidate is best)
    selected_run_input, selected_results = ordered[selected_index]
    selected_run = selected_run_input.resolve(strict=True)
    selected_manifest = selected_results.manifest
    payload = {
        "model_digest": selected_manifest.model_digest,
        "model_tag": selected_manifest.model_tag,
        "prompt_hash": selected_manifest.prompt_hash,
        "prompt_name": selected_manifest.prompt_name,
        "run_dir": str(selected_run),
    }
    _emit(
        arguments,
        payload,
        f"Selected {selected_manifest.model_tag} with prompt "
        f"{best.prompt.name}/{best.prompt.hash} from {selected_run}.",
    )


def _handle_loop_init(arguments: argparse.Namespace) -> None:
    state = LoopState(
        started_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidates_tried=0,
        consecutive_no_improve=0,
        consecutive_errors=0,
        best_exact_match_rate=arguments.best_exact_match_rate,
        baseline_prompt_hash=arguments.baseline_prompt_hash,
    )
    save_loop_state(arguments.state, state)
    _emit(arguments, _state_payload(state), f"Initialized loop state at {arguments.state}.")


def _handle_loop_check(arguments: argparse.Namespace) -> None:
    state = load_loop_state(arguments.state)
    reason = should_stop(state, now_utc=datetime.now(UTC))
    payload = {
        "reason": reason.value if reason is not None else None,
        "state": _state_payload(state),
        "stop": reason is not None,
    }
    human = f"Stop: {reason.value}" if reason is not None else "Continue."
    _emit(arguments, payload, human)


def _handle_loop_record(arguments: argparse.Namespace) -> None:
    state = load_loop_state(arguments.state)
    if arguments.error:
        if arguments.baseline_run is not None or arguments.candidate_run is not None:
            raise _CliInputError("--error cannot be combined with run paths")
        updated = update_after_error(state)
        decision: AdoptionDecision | None = None
    else:
        if arguments.baseline_run is None or arguments.candidate_run is None:
            raise _CliInputError("candidate recording requires --baseline-run and --candidate-run")
        baseline_results = load_run(arguments.baseline_run)
        candidate_results = load_run(arguments.candidate_run)
        _require_matching_runs((baseline_results, candidate_results), required_split="dev")
        _require_same_manifest_fields(
            (baseline_results, candidate_results),
            _ADOPTION_COMPATIBILITY_FIELDS,
        )
        if baseline_results.manifest.prompt_hash != state.baseline_prompt_hash:
            raise _CliInputError("baseline run prompt hash does not match the loop state")
        baseline = _candidate_from_results(baseline_results)
        candidate = _candidate_from_results(candidate_results)
        decision = is_adoptable(baseline, candidate)
        updated = update_after_candidate(
            state,
            decision,
            exact_match_rate=candidate.summary.exact_match_rate,
        )
    save_loop_state(arguments.state, updated)
    payload: dict[str, Any] = {"state": _state_payload(updated)}
    if decision is None:
        payload["recorded"] = "error"
        human = "Recorded a candidate execution error."
    else:
        payload["decision"] = {
            "adopted": decision.adopted,
            "reasons": list(decision.reasons),
        }
        payload["recorded"] = "candidate"
        human = "Recorded the candidate adoption decision."
    _emit(arguments, payload, human)


def _handle_compare_runs(arguments: argparse.Namespace) -> None:
    run_dirs = cast(list[Path], arguments.run_dirs)
    if len(run_dirs) not in (2, 3):
        raise _CliInputError("compare-runs requires exactly two or three run directories")
    results = [load_run(path) for path in run_dirs]
    _require_matching_runs(results)
    split = results[0].manifest.split
    if split == "selection":
        raise _CliInputError("Selection run comparisons are not exposed")
    if split == "final":
        ensure_final_allowed(confirm_final_flag=bool(arguments.confirm_final))
    if len(results) == 3:
        _require_same_manifest_fields(
            results,
            _REPRODUCIBILITY_COMPATIBILITY_FIELDS,
        )
    if len(results) == 2:
        comparison = paired_comparison(results[0].documents, results[1].documents)
        payload = {"kind": "paired_comparison", **asdict(comparison)}
        human = (
            f"Paired comparison: n={comparison.n_docs}, a_only={comparison.a_only}, "
            f"b_only={comparison.b_only}, both={comparison.both}, "
            f"neither={comparison.neither}, p={comparison.mcnemar_p_value:.6g}"
        )
    else:
        summary = reproducibility([result.documents for result in results])
        payload = {"kind": "reproducibility", **asdict(summary)}
        human = (
            f"Reproducibility: n={summary.n_docs}, "
            f"identical prediction rate={summary.identical_prediction_rate:.6f}"
        )
    _emit(arguments, payload, human)


def _prompt_registry_root() -> Path:
    configured = os.environ.get("OCRBENCH_PROMPTS_DIR")
    if configured is None:
        return _REPOSITORY_ROOT / "prompts"
    if not configured.strip():
        raise ConfigError("OCRBENCH_PROMPTS_DIR must not be empty")
    return Path(configured)


def _registered_prompt(name: str, prompt_hash: str) -> PromptVersion:
    for version in list_versions(_prompt_registry_root(), name):
        if version.hash == prompt_hash:
            return version
    raise _CliInputError(f"prompt version is not registered: {name}/{prompt_hash}")


def _prompt_payload(version: PromptVersion) -> dict[str, Any]:
    return {"hash": version.hash, "name": version.name, "text": version.text}


def _candidate_from_results(results: RunResults) -> CandidateEval:
    """Resolve the run's immutable prompt before constructing a candidate."""
    manifest = results.manifest
    prompt = _registered_prompt(manifest.prompt_name, manifest.prompt_hash)
    return CandidateEval(prompt=prompt, summary=aggregate(results.documents))


def _require_prompt_name(results: RunResults, expected: str) -> None:
    if results.manifest.prompt_name != expected:
        raise _CliInputError("run prompt name does not match --prompt-name")


def _require_same_manifest_fields(
    results: Sequence[RunResults],
    fields: Sequence[str],
) -> None:
    reference = results[0].manifest
    for field in fields:
        expected = getattr(reference, field)
        if any(getattr(result.manifest, field) != expected for result in results[1:]):
            raise _CliInputError(f"runs must share manifest field {field!r}")


def _require_matching_runs(
    results: Sequence[RunResults],
    *,
    required_split: str | None = None,
) -> None:
    if not results:
        raise _CliInputError("at least one run is required")
    splits = {result.manifest.split for result in results}
    if len(splits) != 1:
        raise _CliInputError("runs must use the same benchmark split")
    split = next(iter(splits))
    if required_split is not None and split != required_split:
        raise _CliInputError(f"operation requires {required_split} runs")
    fingerprints = {result.manifest.dataset_fingerprint for result in results}
    if len(fingerprints) != 1:
        raise _CliInputError("runs must use the same dataset fingerprint")
    expected_doc_ids = {document.doc_id for document in results[0].documents}
    if any(
        {document.doc_id for document in result.documents} != expected_doc_ids
        for result in results[1:]
    ):
        raise _CliInputError("runs must contain the same document IDs")
    _require_same_manifest_fields(results, _EXECUTION_COMPATIBILITY_FIELDS)


def _stable_run_key(run_dir: Path, results: RunResults) -> tuple[str, ...]:
    """Order exact metric ties independently of caller input order."""
    manifest = results.manifest
    return (
        manifest.model_tag,
        manifest.model_digest,
        manifest.prompt_name,
        manifest.prompt_hash,
        manifest.started_at_utc,
        manifest.run_id,
        str(run_dir.resolve(strict=True)),
    )


def _state_payload(state: LoopState) -> dict[str, Any]:
    return state.model_dump(mode="json")


def _emit(arguments: argparse.Namespace, payload: dict[str, Any], human: str) -> None:
    if bool(getattr(arguments, "json", False)):
        output = {"status": "ok", **payload}
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    elif human:
        print(human)


def _sanitized_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return message if message else error.__class__.__name__


def _emit_error(error: BaseException, *, exit_code: int, json_output: bool) -> None:
    """Emit one sanitized error record without a traceback."""
    message = _sanitized_error(error)
    if json_output:
        print(
            json.dumps(
                {"error": message, "exit_code": exit_code, "status": "error"},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    label = "error" if exit_code == 2 else "runtime error"
    print(f"{label}: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return 0 on success, 1 at runtime, or 2 for rejected input."""
    parser = build_parser()
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    if not raw_arguments:
        parser.print_help()
        return 0
    arguments: argparse.Namespace | None = None
    requested_debug = "--debug" in raw_arguments
    requested_json = "--json" in raw_arguments
    try:
        arguments = parser.parse_args(raw_arguments)
        handler = cast(_Handler | None, getattr(arguments, "handler", None))
        if handler is None:
            raise _CliInputError("a command and subcommand are required")
        handler(arguments)
        return 0
    except _ParserExit as error:
        return error.status
    except (
        _CliInputError,
        ConfigError,
        DatasetError,
        PromptRegistryError,
        ResultsError,
        SplitGuardError,
        ValidationError,
        ValueError,
    ) as error:
        if requested_debug or (arguments is not None and bool(getattr(arguments, "debug", False))):
            raise
        _emit_error(error, exit_code=2, json_output=requested_json)
        return 2
    except (ModelNotFoundError, RunnerError) as error:
        if requested_debug or (arguments is not None and bool(getattr(arguments, "debug", False))):
            raise
        _emit_error(error, exit_code=1, json_output=requested_json)
        return 1
    except Exception as error:
        if requested_debug or (arguments is not None and bool(getattr(arguments, "debug", False))):
            raise
        _emit_error(error, exit_code=1, json_output=requested_json)
        return 1
