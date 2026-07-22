# T02: .gitignore・Lefthook・機密データ commit ガード

| 項目 | 内容 |
|---|---|
| Phase | 0 基盤 |
| ADR 参照 | §5 機密情報と評価汚染の防止、§11 実装と品質管理 |
| 依存 | T01 |
| 後続 | 全タスク（コミット時に常時作動） |
| 主な成果物 | `.gitignore`, `lefthook.yml`, `scripts/guard_staged_files.py` |

## 目的

実帳票・正解 JSON・個別予測・大容量ファイルの誤 commit を、`.gitignore` と Lefthook の二重で拒否する。
`.gitignore` は「うっかり」を防ぎ、ガードスクリプトは `git add -f` のような明示的ミスも止める。

## スコープ

### 含む
- `.gitignore` 整備、Lefthook 設定（pre-commit / pre-push）、ガードスクリプトとそのテスト
### 含まない
- CI（T03）、Final test の OS レベル隔離（T16 / H3）

## 実装指示

### 1. `.gitignore`

以下の方針で作成する。順序に注意（否定パターンは後）。

```gitignore
# Python / uv
__pycache__/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/

# ADR-0001 §5: 実データ・実行結果はリポジトリ外。リポジトリ内に置かれても無視する
data/
runs/
*.env

# 実帳票になり得る形式は既定で全面禁止
*.pdf
*.jpg
*.jpeg
*.png
*.tif
*.tiff
*.bmp
*.webp
*.heic

# 正解 JSON の命名規約（gt_*.json / *_gt.json）も禁止
gt_*.json
*_gt.json

# 匿名 fixture のみ明示的に許可（ADR §5）
!tests/fixtures/public/
!tests/fixtures/public/**
```

### 2. `scripts/guard_staged_files.py`

ステージ済みファイルを検査し、違反があれば一覧を表示して終了コード 1 で commit を止める。
Windows でも動くよう純 Python（標準ライブラリのみ）で書く。

- 判定ロジックは `check_staged_paths(paths: list[str], size_of: Callable[[str], int]) -> list[Violation]`
  のような**純粋関数へ分離**し、ユニットテスト可能にする
- `main()` は `git diff --cached --name-only -z --diff-filter=ACMR` の結果を渡すだけの薄い層にする
- 違反条件（`tests/fixtures/public/` 配下はすべて免除）:
  1. パスが `data/`, `runs/` で始まる
  2. 拡張子が画像・PDF 系（`.gitignore` と同じリスト）
  3. ファイル名が `gt_*.json` / `*_gt.json` に一致
  4. ファイルサイズ > 5,000,000 bytes（帳票スキャンの誤混入と大容量ファイル対策）
  5. 拡張子が `.env`
- 免除にも上限を設ける: `tests/fixtures/public/` 配下でもサイズ > 5MB は拒否

### 3. `lefthook.yml`（ADR §11 の構成そのまま）

```yaml
pre-commit:
  parallel: false
  commands:
    guard-secrets:
      run: uv run python scripts/guard_staged_files.py
      priority: 1
    ruff-fix:
      glob: "*.py"
      run: uv run ruff check --fix {staged_files}
      stage_fixed: true
      priority: 2
    ruff-format:
      glob: "*.py"
      run: uv run ruff format {staged_files}
      stage_fixed: true
      priority: 3

pre-push:
  parallel: true
  commands:
    ruff-check:
      run: uv run ruff check .
    ruff-format-check:
      run: uv run ruff format --check .
    mypy:
      run: uv run mypy
    pytest:
      run: uv run pytest
```

- Lefthook 本体はコミット対象外のローカルツールとする。導入手順（`lefthook install`）を
  ルート `README.md` に追記する。`uv` の dev 依存へ `lefthook` を追加してもよい（wheel 配布あり）。
  追加した場合は `uv run lefthook install` を手順とする

## 提供インターフェース

- `scripts/guard_staged_files.py` の `check_staged_paths()`（T03 の CI からも将来呼べる形）

## 受け入れ基準（DoD）

- [ ] `data/x.json`, `foo.pdf`, `tests/gt_sample.json`, 6MB のファイル, `.env` をステージすると commit が拒否される
- [ ] `tests/fixtures/public/sample.png`（小容量）はステージ・commit できる
- [ ] `git check-ignore` で `.gitignore` の許可 / 禁止が上記と一致する
- [ ] `lefthook install` 後、Python ファイルの commit 時に ruff の自動修正が働く
- [ ] pre-push で 4 コマンド（ruff check / format --check / mypy / pytest）が走る
- [ ] 品質ゲート（ruff / mypy / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_staged_files_guard.py`: 判定関数のテーブル駆動テスト
  - 拒否: `data/` 配下、`runs/` 配下、各画像拡張子、`gt_*.json`、`*_gt.json`、5MB 超、`.env`
  - 許可: `src/ocrbench/foo.py`、`tests/fixtures/public/mini.png`、`prompts/base/abc123.txt`
  - 免除の上限: `tests/fixtures/public/big.png`（6MB）は拒否
- `.gitignore` の検証は `git check-ignore` を subprocess で呼ぶ統合テスト 1 本（tmp リポジトリ不要、
  本リポジトリのワークツリーに対しパス文字列で判定）
