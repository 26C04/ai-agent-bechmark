# ocrbench

Ollama 上の Vision Language Model を使い、生産指示書の OCR・構造化抽出を比較するための
ローカルベンチマーク基盤です。

## セットアップ

Python 3.14 と [uv](https://docs.astral.sh/uv/) を用意し、次を実行します。

```powershell
uv sync
uv run ocrbench --version
uv run pytest
```

実 Ollama を使うテストは通常のテスト・CIから除外されています。ローカルサーバーと対象モデルを
用意したうえで、明示的に次を実行してください。

```powershell
uv run pytest -m ollama --no-cov
```

## Git hooks

[Lefthook](https://github.com/evilmartians/lefthook) をローカルへ導入した後、リポジトリ直下で
フックを有効化します。

```powershell
lefthook install
```

pre-commitでは機密データ混入ガードとRuffの自動修正・整形、pre-pushではRuff、Mypy、pytestが
実行されます。

設計判断は [ADR-0001](adr/0001-local-ocr-benchmark-architecture.md)、実装順と各タスクの契約は
[タスクリスト](tasks/README.md)を参照してください。
