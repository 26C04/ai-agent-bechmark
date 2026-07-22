# T01: プロジェクト初期化と品質ツールチェーン

| 項目 | 内容 |
|---|---|
| Phase | 0 基盤 |
| ADR 参照 | §11 実装と品質管理 |
| 依存 | なし |
| 後続 | 全タスク |
| 主な成果物 | `pyproject.toml`, `uv.lock`, `.python-version`, `src/ocrbench/`, `tests/` |

## 目的

Python 3.14 + `uv` の src layout プロジェクトを作成し、Ruff / Mypy strict / pytest + coverage の
品質ゲートを最初の PR から機能させる。以降の全タスクはこの上に積む。

## スコープ

### 含む
- パッケージ雛形、CLI エントリポイントのスタブ、`config.py`（定数と環境変数読み取り）
- Ruff / Mypy / pytest / coverage の設定
### 含まない
- Lefthook（T02）、CI（T03）、業務ロジック一切

## 実装指示

### 1. プロジェクト定義

`pyproject.toml` を作成する（`uv init --package` 相当から調整）。

- `name = "ocrbench"`、`requires-python = ">=3.14"`
- ランタイム依存: `pydantic`（v2 系）, `pillow`, `pypdfium2`, `ollama`
  — バージョンは実装時点の最新安定版を `uv add` で解決し `uv.lock` に固定する
- dev 依存（`[dependency-groups]` の `dev`）: `ruff`, `mypy`, `pytest`, `pytest-cov`
- `[project.scripts]` に `ocrbench = "ocrbench.cli:main"`
- `.python-version` に `3.14`

### 2. ソース雛形

```text
src/ocrbench/__init__.py    # __version__ を定義
src/ocrbench/py.typed       # 空ファイル（PEP 561）
src/ocrbench/config.py
src/ocrbench/cli.py
```

`config.py` に以下を定義する。

```python
SEED: Final[int] = 20260721          # 推論 seed・統計 seed の唯一の既定値
ALLOWED_MODELS: Final[tuple[str, ...]] = (
    "gemma4:12b",
    "qwen3.5:9b",
    "minicpm-v4.5:8b",
    "glm-ocr:bf16",
    "ministral-3:14b",
)  # ADR §6 の初期許可モデル

def data_dir() -> Path: ...      # env OCRBENCH_DATA_DIR。未設定なら ConfigError
def final_dir() -> Path: ...     # env OCRBENCH_FINAL_DIR。未設定なら ConfigError
def runs_dir() -> Path: ...      # env OCRBENCH_RUNS_DIR。未設定なら ConfigError
```

- `ConfigError` は `ocrbench` 共通の例外基底 `OcrBenchError(Exception)` の派生とし、`config.py` に置く
- 環境変数は呼び出し時に読む（import 時に読まない。テストで monkeypatch しやすくするため）

`cli.py` はスタブとする: `argparse` で `--version` のみ対応し、`main(argv: list[str] | None = None) -> int`
を定義。`[project.scripts]` 経由で終了コードが返るよう `def entrypoint() -> None: sys.exit(main())` 形式でも、
`main` を直接使う形式でもよいが、**テストから `main(["--version"])` を呼べる**構造にする。

### 3. ツール設定（`pyproject.toml` 内）

```toml
[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
strict = true
files = ["src", "tests", "scripts"]   # scripts/ は T02 で追加されるが先に宣言してよい

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["ollama: requires a real local Ollama server (excluded by default)"]
addopts = "-m 'not ollama' --cov=ocrbench --cov-report=term-missing --cov-fail-under=80"
```

- `-m 'not ollama'` が既定。実 Ollama テストは `uv run pytest -m ollama --no-cov` で明示実行する
  運用（ADR §11）。この旨をルート `README.md` に記載する

### 4. README 更新

ルート `README.md` に最小のセットアップ手順を記載する:
`uv sync` → `uv run ocrbench --version` → `uv run pytest`。
ADR と `tasks/README.md` へのリンクを張る。

## 提供インターフェース（後続タスクとの契約）

- `ocrbench.config`: `SEED`, `ALLOWED_MODELS`, `data_dir()`, `final_dir()`, `runs_dir()`,
  `OcrBenchError`, `ConfigError`
- `ocrbench.cli.main(argv: list[str] | None = None) -> int`

## 受け入れ基準（DoD）

- [ ] `uv sync` が成功し、`uv.lock` がコミットされている
- [ ] `uv run ocrbench --version` がバージョンを表示して終了コード 0
- [ ] `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy` / `uv run pytest` が全て成功
- [ ] coverage が 80% 以上（スタブ段階なので `cli` / `config` のテストで満たす）
- [ ] `pytest -m ollama` を打っても収集エラーにならない（該当テスト 0 件で正常終了、または skip）

## テスト要件

- `tests/test_config.py`: 環境変数設定時に `Path` が返る / 未設定時に `ConfigError`（monkeypatch 使用）、
  `ALLOWED_MODELS` が ADR §6 の 5 タグと一致
- `tests/test_cli.py`: `main(["--version"])` が 0 を返し版数文字列を出力する（capsys）
