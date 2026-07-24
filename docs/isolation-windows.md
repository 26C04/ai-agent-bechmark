# Isolating the Final Test on Windows

This runbook establishes the operating-system boundary required by ADR-0001.
The `--confirm-final` option and `OCRBENCH_ALLOW_FINAL=1` are only safeguards
against accidental operation. They are not a security boundary. A separate
Windows local account and NTFS access control lists (ACLs) are the primary
boundary.

The commands below must be reviewed and run by a human administrator. Replace
the example account names and paths before running them. Do not run an agent,
prompt-improvement process, editor, or development shell under the Final-test
account.

## 1. Create a dedicated local account

Open an elevated Command Prompt and create a standard local user. Using `*`
prompts for the password instead of storing it in shell history:

```bat
net user finaltest * /add
net user finaltest
```

Do not add `finaltest` to `Administrators`. Do not reuse the development user's
password or credentials. Record account ownership and recovery information in
the laboratory's approved secret-management system. The development and agent
execution accounts must also be standard users, not members of
`Administrators`. They must not be able to elevate without approval from a
separate human administrator because an elevated administrator token can access
the protected directories.

## 2. Prepare and protect the Final-test directories

Create dedicated input and output directories on NTFS. The output directory
also needs protection because `results.json` contains document-level
predictions.

```bat
mkdir D:\OCRBench\final
mkdir D:\OCRBench\final-runs
icacls "D:\OCRBench\final" /inheritance:r /grant "finaltest:(OI)(CI)F" /grant "Administrators:(OI)(CI)F"
icacls "D:\OCRBench\final-runs" /inheritance:r /grant "finaltest:(OI)(CI)F" /grant "Administrators:(OI)(CI)F"
```

Inspect the effective ACLs before copying any data:

```bat
icacls "D:\OCRBench\final"
icacls "D:\OCRBench\final-runs"
```

Remove any explicit grants for the development user, agent execution user, and
other interactive users. For example, after substituting the real account:

```bat
icacls "D:\OCRBench\final" /remove:g "LAB-PC\developer" "LAB-PC\agentuser"
icacls "D:\OCRBench\final-runs" /remove:g "LAB-PC\developer" "LAB-PC\agentuser"
```

Do not remove administrator access until the laboratory has an independently
tested recovery procedure. Windows 11 Home does not provide the Local Security
Policy console, but `net user`, `icacls`, and separate local accounts still
provide the required boundary on NTFS.

## 3. Stage and verify the dataset

Sign in interactively as `finaltest`, set the process environment, and validate
the fixed dataset. Keep the directories outside the repository.

```powershell
$env:OCRBENCH_FINAL_DIR = "D:\OCRBench\final"
$env:OCRBENCH_RUNS_DIR = "D:\OCRBench\final-runs"
ocrbench dataset validate --root $env:OCRBENCH_FINAL_DIR --strict-counts
ocrbench dataset fingerprint --root $env:OCRBENCH_FINAL_DIR
```

Before the first scored run, confirm all of the following:

- The Final manifest contains exactly the pre-registered 40 documents: 20 scans
  and 20 photos.
- Every source file and ground-truth file matches the fixed manifest and
  fingerprint.
- A person other than the ground-truth author has compared the Final
  ground-truth JSON with the source documents, as required by ADR-0001 section
  4.
- The approved model digest, prompt content hash, Ollama version, preprocessing
  version, seed, and inference settings are recorded.

## 4. Prove that the boundary denies the development account

Sign out of `finaltest` and sign in as the normal development or agent execution
user. Both commands below must fail with `Access is denied`:

```bat
dir "D:\OCRBench\final"
dir "D:\OCRBench\final-runs"
```

From an elevated administrator shell, also verify that the ACL structures are
valid:

```bat
icacls "D:\OCRBench\final" /verify
icacls "D:\OCRBench\final-runs" /verify
```

`icacls /verify` checks ACL consistency; the failed `dir` commands under the
actual development and agent accounts are the effective-access tests. Do not
proceed if either account can list, read, write, change permissions on, or take
ownership of either directory through its normal execution token.
Also confirm that neither account can elevate without approval from a separate
human administrator.

## 5. Verify the software confirmation gate

The software gate deliberately requires two independent confirmations. From a
test environment that contains only anonymous fixtures, verify that each
incomplete combination raises `FinalAccessError`:

```powershell
Remove-Item Env:OCRBENCH_ALLOW_FINAL -ErrorAction SilentlyContinue
ocrbench run --model gemma4:12b --split final --prompt-name base
ocrbench run --model gemma4:12b --split final --prompt-name base --confirm-final

$env:OCRBENCH_ALLOW_FINAL = "1"
ocrbench run --model gemma4:12b --split final --prompt-name base
```

Only an authorized Final-test session may supply both confirmations:

```powershell
$env:OCRBENCH_ALLOW_FINAL = "1"
ocrbench run --model gemma4:12b --split final --prompt-name base --confirm-final
Remove-Item Env:OCRBENCH_ALLOW_FINAL
```

Set `OCRBENCH_ALLOW_FINAL` only for the authorized process; do not persist it
with `setx`, a profile, a task definition, or a repository file. The runner
accepts the exact value `1` only.

## 6. Handle outputs without breaking blindness

`report.md` may be generated in the ACL-protected Final run directory for human
review. Never provide Final or Selection `results.json`, detailed reports,
images, ground truth, individual predictions, or per-document differences to a
prompt-improvement agent.

The detail-report command is restricted to the Development split. When an agent
needs evaluation feedback, provide only the allowlisted anonymous summary
export produced by the reporting command. Review that export through the
laboratory's disclosure process before transferring it outside the isolated
account.

## Final authorization checklist

- [ ] `finaltest` is a dedicated standard local account.
- [ ] Final input and output directories are outside the repository on NTFS.
- [ ] Inherited ACLs are removed and only `finaltest` and administrators have
      access.
- [ ] The development and agent accounts receive `Access is denied`.
- [ ] The development and agent accounts are non-administrators and cannot
      elevate without separate human administrator approval.
- [ ] The Final manifest and fingerprint are fixed.
- [ ] Ground truth has been independently checked against the source documents.
- [ ] The model digest and prompt hash match the pre-registered configuration.
- [ ] Missing either software confirmation produces `FinalAccessError`.
- [ ] No agent process runs as `finaltest` or holds a token that can read the
      Final directories.
- [ ] Only approved aggregate exports leave the isolated environment.
