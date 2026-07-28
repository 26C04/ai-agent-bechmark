# Windows Setup and Smoke-Test Runbook

This runbook prepares the ADR-0001 reference workstation: Windows 11 Home,
Intel Core Ultra 7 265K, 64 GB RAM, NVIDIA GeForce RTX 5070 Ti, and a 2 TB NVMe
SSD. Run the commands as the ordinary laboratory user; keep benchmark data and
run outputs outside the repository.

## 1. Verify the NVIDIA driver and GPU

Open PowerShell and record the GPU and driver version before installing models:

```powershell
nvidia-smi
```

Ollama's current Windows documentation requires Windows 10 22H2 or later and
an NVIDIA driver version 551.61 or later. Record the command output with the
benchmark notes. Do not treat a successful command as proof that a model is
fully resident on the GPU; that is checked after the smoke test.

## 2. Install uv and Python 3.14

Install `uv` from its official Windows distribution channel:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell session, then clone the approved repository revision and
synchronize its locked environment:

```powershell
uv sync
uv run ocrbench --version
```

Use the repository's `uv.lock`; do not replace, loosen, or silently regenerate
the pinned dependencies during a benchmark. The project requires Python 3.14.

## 3. Install and maintain Ollama

Install Ollama for Windows from the official installer, then verify the version:

```powershell
ollama --version
```

Ollama for Windows runs in the background and exposes its local API at
`http://localhost:11434`. Its official documentation says that Windows updates
download automatically and are applied by choosing **Restart to update** from
the taskbar item; a current installer can also be downloaded manually. There is
no documented update-disable switch in this runbook. For a controlled benchmark
window, a human maintainer must defer applying an offered update, record the
installed version, and schedule the update and re-validation outside the window.
After any approved update, record the new version, rerun smoke, and compare the
recorded model digests before accepting results.

Never disable, bypass, rename, copy around, or elevate to evade Windows App
Control, WDAC, AppLocker, or Smart App Control. Use signed, IT-approved
installations or human-managed policy exceptions.

## 4. Pull and record the approved models

Pull exactly the five ADR-approved tags:

```powershell
ollama pull gemma4:12b
ollama pull qwen3.5:9b
ollama pull minicpm-v4.5:8b
ollama pull glm-ocr:bf16
ollama pull ministral-3:14b
ollama list
```

After each pull, record the tag and immutable digest shown by `ollama list` in
the laboratory's benchmark record. Tags alone are not reproducibility evidence.
Do not substitute a newer tag or a similarly named model without an approved
reference-configuration change.

## 5. Configure external data and run locations

Choose approved external NTFS locations for Development and Selection data and
for run outputs. Set the user-level values only after the directories have been
created and reviewed:

```powershell
New-Item -ItemType Directory -Force "D:\OCRBench\data"
New-Item -ItemType Directory -Force "D:\OCRBench\runs"
setx OCRBENCH_DATA_DIR "D:\OCRBench\data"
setx OCRBENCH_RUNS_DIR "D:\OCRBench\runs"
```

Open a new terminal after `setx`. Do not set `OCRBENCH_FINAL_DIR` here. Final
data requires the separate-account and ACL procedure in
[the Windows Final-test isolation runbook](isolation-windows.md).

## 6. Run one anonymous-sample smoke test

Use one locally approved anonymous sample image or PDF. It must contain no
production customer data, credentials, personal data, or unreviewed ground
truth. Do not place a real production document in the repository, test fixture,
issue, pull request, or agent prompt.

```powershell
uv run ocrbench smoke --model gemma4:12b --image "D:\OCRBench\samples\anonymous.png"
```

The command loads the active `base` prompt by default. To use another registered
active prompt, supply `--prompt-name <name>`. Its output records the server
version, resolved model digest, image source and preprocessing metadata,
timings, token data when available, parse status, and `needs_review`.

The smoke result passes only when all four checks are satisfied:

- The displayed model digest matches the digest recorded after `ollama pull`.
- GPU placement is explicitly reported as `100% GPU`.
- A warm run meets the 30-second target.
- The final `ParseStatus` is `OK`.

## 7. Confirm full GPU residence

Immediately after smoke, inspect the loaded model:

```powershell
ollama ps
```

The `PROCESSOR` value must be `100% GPU` for the reference configuration.
Any CPU/RAM offload, partial GPU placement, missing placement, or other warning
is a diagnostic failure: record it, do not call the configuration reference
compliant, and have a human investigate VRAM capacity, driver state, and model
selection. The smoke command itself reports the same warning; it does not alter
Ollama configuration.

## 8. Run the real Ollama integration test

After smoke passes, run the opt-in real-server test from the approved machine:

```powershell
uv run pytest -m ollama --no-cov
```

This test needs a reachable local Ollama server and an approved pulled model.
It is deliberately excluded from ordinary CI and from the default test suite.

## 9. Troubleshoot without changing the reference silently

If smoke reports a model-not-found error, pull the exact approved tag and record
its digest. If it reports a parse failure, it prints the diagnostics and exits
with status 1; preserve the output for human review. A valid parse exits 0.
Input errors, including an unapproved model without an explicit smoke-only
override, exit 2.

If Ollama or a selected model reports that disabling thinking is unsupported,
preserve the exact diagnostic and stop the run. A human maintainer must decide
whether to select another approved model/version or formally revise the
reference configuration and rerun validation. Do not change hidden model
settings, patch Ollama, or improvise a bypass.

`--allow-any-model` is diagnostic-only and is available only to `ocrbench
smoke`. It prints an emphatic warning and must not be used for a benchmark run,
reference acceptance, candidate comparison, or result publication. Never use it
as a substitute for approving a model or recording its digest.

## 10. Prepare the next benchmark step

For a passing reference smoke, retain the command output and record the model
tag, digest, active prompt content hash, Ollama version, preprocessing version,
GPU placement, and date. Then proceed only to the next approved task in
`tasks/README.md`. Before any Final-test operation, complete the isolation
runbook linked in section 5; the CLI confirmation flag is not a security
boundary.

Official references: [uv installation](https://docs.astral.sh/uv/getting-started/installation/),
[Ollama on Windows](https://docs.ollama.com/windows), and
[Ollama FAQ](https://docs.ollama.com/faq).
