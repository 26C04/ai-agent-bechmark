# T03: GitHub Actions CI

| 項目 | 内容 |
|---|---|
| Phase | 0 基盤 |
| ADR 参照 | §11 実装と品質管理 |
| 依存 | T01 |
| 後続 | 全タスク（PR ごとに常時作動） |
| 主な成果物 | `.github/workflows/ci.yml` |

## 目的

`ubuntu-latest` 上で lockfile 検証・Ruff・Mypy・pytest を実行する CI を用意する。
実データ・GPU・Ollama モデルは CI で一切使わない（ADR §11）。

## スコープ

### 含む
- CI ワークフロー 1 本
### 含まない
- CD・PyPI 公開・自動デプロイ（ADR §11 で初期版は行わないと明記）
- Windows ランナーでの実行（実機評価はローカルの H タスク）

## 実装指示

`.github/workflows/ci.yml` を作成する。

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.14"
      # lockfile が pyproject.toml と一致するか検証（ADR §11「lockfile 検証」）
      - run: uv lock --check
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy
      # pytest の既定 addopts が -m "not ollama" のため、実 Ollama は CI で決して要求されない
      - run: uv run pytest
```

注意点:

- 実行コマンド列は Lefthook の pre-push（T02）と**同一**に保つ。将来ドリフトしないよう、
  ci.yml とlefthook.yml の双方に相互参照コメントを入れる
- アクションのバージョン（`@v4` / `@v5`）は実装時点の最新メジャーを使う
- リポジトリは研究発表まで非公開（ADR §5）。Secrets は使わない。fork PR 対応の設定は不要

## 受け入れ基準（DoD）

- [ ] `ci.yml` が有効な YAML であり、`actionlint` 相当のチェック（手元実行または目視）で文法エラーがない
- [ ] 実行ステップが Lefthook pre-push と同じ 4 種 + lockfile 検証で構成されている
- [ ] GPU・Ollama・実データへの参照が一切ない（`-m ollama` を実行しない）
- [ ] push 後、GitHub 上で CI が green になる（このタスクの PR 自体で確認）

## テスト要件

- 追加の pytest は不要（ワークフロー自体が検証物）。既存テストが CI 上で通ることを確認する
