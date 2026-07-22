# T17: CLI 統合と fake E2E

| 項目 | 内容 |
|---|---|
| Phase | 3 統合 |
| ADR 参照 | §11（インターフェースは CLI のみ）、§8（エージェントは共通 Python CLI を実行）、§10 |
| 依存 | T14, T15, T16（= T04〜T16 全部） |
| 後続 | T18、H2〜H5（全運用がこの CLI を使う） |
| 主な成果物 | `src/ocrbench/cli.py`（本実装）|

## 目的

これまでの全モジュールを `ocrbench` コマンドの argparse サブコマンドとして束ね、
fake adapter による**エンドツーエンドの通し**を成立させる。エージェント（H4）と人間（H2/H5）の
唯一のインターフェースになる。

## スコープ

### 含む
- サブコマンド実装、終了コード規約、fake E2E テスト
### 含まない
- smoke コマンド（T18）、新しい業務ロジック（既存モジュールへの委譲のみ。
  ハンドラ内にロジックを書かない）

## 実装指示

### 1. サブコマンド構成

```text
ocrbench --version
ocrbench dataset build-manifest --root <dir> --assignments <csv>
ocrbench dataset validate --root <dir> [--strict-counts]
ocrbench dataset fingerprint --root <dir>
ocrbench prompt add --name <n> --file <txt> [--note ...]        # lint 警告を表示
ocrbench prompt list --name <n>
ocrbench prompt show --name <n> [--hash <h>]                    # 省略時 active
ocrbench prompt activate --name <n> --hash <h> [--note ...]
ocrbench prompt rollback --name <n>
ocrbench run --model <tag> --split <dev|selection|final> --prompt-name <n>
             [--run-id <id>] [--confirm-final]
ocrbench report --run-dir <dir>                                  # report.md + analysis.md 雛形
ocrbench report --run-dir <dir> --detail                         # 詳細差分（dev のみ。T16 ガード）
ocrbench export-summary --run-dir <dir> --out <json>             # 匿名集計（エージェント向け）
ocrbench select adopt --baseline-run <dir> --candidate-run <dir> --prompt-name <n>
                                                                 # is_adoptable → apply_adoption
ocrbench select pick-best --candidate-runs <dir>... 
ocrbench loop init|check|record --state <json> ...               # LoopState の操作（should_stop 等）
ocrbench compare-runs --run-dirs <dir> <dir> [<dir>]             # 再現一致率 / 対応あり比較
```

- `run` は T16 の `resolve_split_dir` / `ensure_final_allowed` を通し、adapter は
  `create_adapter()`（env `OCRBENCH_ADAPTER`）で得る。モデルタグは `config.ALLOWED_MODELS` を検証
- `run` 完了時の標準出力は T16 `console_summary_for` に従う（selection / final で集計を出さない）
- `select` / `loop` / `export-summary` はエージェントが叩く前提。出力は機械可読（JSON を stdout）と
  人間可読の両立のため `--json` フラグを共通で用意する
- `compare-runs`: 2 run なら `paired_comparison`、3 run なら `reproducibility` を表示（T08）

### 2. 実装規約

- `cli.py` は「引数解析 → 対応モジュール呼び出し → 出力整形」のみ。1 サブコマンド 1 関数
- 終了コード規約: `0` 成功 / `1` 実行時エラー（adapter 障害等）/ `2` 入力・検証・ガード拒否
  （DatasetError, FinalAccessError, ConfigError, PromptRegistryError など）
- すべての例外はハンドラ境界で捕捉し、stderr に 1 行サマリ + 終了コード。トレースバックを
  ユーザーへ見せない（`--debug` で再送出）
- `main(argv)` は引き続きテスト可能な純関数スタイルを維持

### 3. fake E2E（このタスクの本丸）

`tests/test_cli_e2e.py` に、実 Ollama なしの通しテストを書く:

1. tmp_path にミニデータセット構築（T11 fixture）+ 環境変数設定（monkeypatch）
2. `dataset build-manifest` → `dataset validate` → `dataset fingerprint`
3. `prompt add` → `prompt activate`
4. `OCRBENCH_ADAPTER=fake` で `run --model gemma4:12b --split dev`
   （fake の既定応答を「妥当な JSON」にし、1 件は不正応答にして再試行パスも通す）
5. `report` → report.md 生成を確認
6. `export-summary` → 匿名集計 JSON を確認
7. `select pick-best`（run 2 つで）と `compare-runs`
8. final ゲート: `run --split final` がフラグなしで終了コード 2

## 提供インターフェース

- `ocrbench` CLI 全サブコマンド（H2〜H5 の運用手順が依存する安定インターフェース）

## 受け入れ基準（DoD）

- [ ] 上記サブコマンドがすべて動作し `--help` に説明がある
- [ ] fake E2E（1〜8）が pytest で通る
- [ ] 終了コード規約（0/1/2）がテストで検証されている
- [ ] ハンドラにビジネスロジックが無い（既存モジュール呼び出しのみ。レビュー観点）
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_cli_e2e.py`: 上記の通しシナリオ
- `tests/test_cli.py`（拡張): サブコマンド単位の異常系
  - 未知モデルタグ → 2 / データディレクトリ未設定（ConfigError）→ 2 /
    存在しない run-dir → 2 / fake adapter 障害注入 → 1
  - `--json` 出力が `json.loads` 可能
  - `--help` のスナップショット（サブコマンド一覧が揃っていること）
