# ocrbench

`ocrbench` is a local benchmark foundation for comparing OCR and structured
extraction from production instructions with Vision Language Models served by
Ollama. The benchmark is designed for reproducible experiments without placing
real documents or ground truth in the Git repository.

## Setup

Install Python 3.14 and [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run ocrbench --version
uv run pytest
```

Tests that require a real Ollama server are excluded from the normal test suite
and CI. After starting the local server and installing the target model, run
them explicitly:

```powershell
uv run pytest -m ollama --no-cov
```

## Git hooks

Install [Lefthook](https://github.com/evilmartians/lefthook) in the repository:

```powershell
lefthook install
```

The pre-commit hook checks staged paths for confidential or bulky benchmark
data, then applies Ruff fixes and formatting. The pre-push hook runs Ruff,
Mypy in strict mode, and the normal pytest suite.

The architectural decisions are in
[ADR-0001](adr/0001-local-ocr-benchmark-architecture.md). The implementation
sequence and cross-module contracts are in the
[task list](tasks/README.md).

## External dataset layout

Real datasets must be outside the repository. `OCRBENCH_DATA_DIR` stores
development and selection data. `OCRBENCH_FINAL_DIR` has the same layout but is
reserved for the final split and must use the separate Windows account and NTFS
ACL isolation described by ADR-0001.

```text
$OCRBENCH_DATA_DIR/
├── raw/<doc-id>.<pdf|jpg|jpeg|png>
├── gt/<doc-id>.json
└── manifest.json

$OCRBENCH_FINAL_DIR/
├── raw/<doc-id>.<pdf|jpg|jpeg|png>
├── gt/<doc-id>.json
└── manifest.json
```

Each manifest uses schema version 1. Its `file` values are canonical,
single-component names relative to `raw/`. A document ID is always
`doc-<first 12 lowercase hex characters of the raw-file SHA-256>`, and the file
stem must equal that ID. Ground-truth files must satisfy the strict
`OrderDocument` schema.

Example:

```json
{
  "docs": [
    {
      "doc_id": "doc-0123456789ab",
      "file": "doc-0123456789ab.png",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source_kind": "photo",
      "split": "dev"
    }
  ],
  "schema_version": 1
}
```

## Building a manifest

1. Create `<dataset-root>/raw/` and place each source PDF or image directly in
   that directory. Subdirectories, absolute paths, backslashes, symlinks, and
   unsupported extensions are rejected.
2. Create a UTF-8 CSV with exactly the columns `file`, `source_kind`, and
   `split`. `file` is the current direct-child filename in `raw/`;
   `source_kind` is `scan` or `photo`; `split` is `dev`, `selection`, or
   `final`.
3. Call `build_manifest`. It preflights every row before changing anything,
   renames sources to content-derived anonymous names, and atomically writes
   `manifest.json`.
4. Create `gt/<doc-id>.json` for every generated entry. Final-test ground truth
   must be checked against the original by someone other than its author.
5. Load the completed dataset, check the split counts, and record its
   fingerprint.

```csv
file,source_kind,split
incoming-scan.pdf,scan,dev
phone-photo.jpg,photo,selection
```

```python
from pathlib import Path

from ocrbench.dataset import (
    build_manifest,
    dataset_fingerprint,
    load_manifest,
    validate_split_counts,
)

root = Path(r"D:\ocrbench-data")
manifest = build_manifest(root, Path(r"D:\assignments.csv"))

# Add and independently verify gt/<doc-id>.json files before loading.
manifest = load_manifest(root)
warnings = validate_split_counts(manifest, strict=False)
fingerprint = dataset_fingerprint(root, manifest)
```

The builder rejects malformed CSV data, duplicate assignments, duplicate file
content, truncated-hash collisions, missing files, and target collisions before
the first rename. It never overwrites a raw target. Renames use anonymous
staging names and are rolled back after ordinary I/O failures; the manifest is
prepared separately and installed with an atomic replacement. The persisted
manifest therefore contains no original source filename.

Fingerprinting is independent of manifest document order. It hashes canonical
sorted manifest data plus freshly streamed hashes of every raw and ground-truth
file, using explicit component framing so concatenation cannot be ambiguous.

## Confidentiality rules

- Never commit real production instructions, ground-truth JSON, individual
  predictions, or detailed error diffs.
- Store real datasets and run outputs only in repository-external directories
  referenced by `OCRBENCH_DATA_DIR`, `OCRBENCH_FINAL_DIR`, and
  `OCRBENCH_RUNS_DIR`.
- Only explicitly synthetic, anonymous fixtures under
  `tests/fixtures/public/` may be tracked. Generate images and PDFs in test code
  whenever practical.
- Do not provide cloud agents with images, ground-truth values, or misread
  values. Only anonymous aggregate metrics and error categories may leave the
  isolated environment.
- Do not rely on a CLI flag as the final-test security boundary. Use a separate
  Windows user and NTFS ACLs.
- Keep the GitHub repository private until the research results are published,
  as required by ADR-0001.
