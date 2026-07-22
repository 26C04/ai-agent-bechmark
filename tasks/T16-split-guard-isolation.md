# T16: split ガードと Final test 隔離補助

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §4（分割固定）、§5（Final の OS 隔離・CLI フラグを信用しない）、§8-5（Selection の盲検） |
| 依存 | T11, T12 |
| 後続 | T17、H3（隔離手順の実施） |
| 主な成果物 | `src/ocrbench/splitguard.py`, `docs/isolation-windows.md` |

## 目的

Final test / Selection への誤アクセスと評価汚染を防ぐ**ソフトウェア側の防壁**を実装する。
主防壁はあくまで OS レベル分離（別 Windows ユーザー + NTFS ACL。H3）であり、CLI 内のフラグは
補助に過ぎない（ADR §5「CLI 内のフラグだけをテストロックとして信用しない」）。その前提を
ドキュメントにも明記する。

## スコープ

### 含む
- split → データディレクトリ解決、final 実行の二重ゲート、selection の個別結果秘匿、隔離手順書
### 含まない
- NTFS ACL の設定作業そのもの（H3）、runner 本体（T12。ここのフックを呼ぶ側）

## 実装指示

### 1. split ごとのディレクトリ解決（`splitguard.py`）

```python
def resolve_split_dir(split: Literal["dev", "selection", "final"]) -> Path:
    """dev / selection → config.data_dir()、final → config.final_dir()。
    final のデータが誤って data_dir 側に置かれても参照されない構造にする。"""

class FinalAccessError(OcrBenchError): ...

def ensure_final_allowed(*, confirm_final_flag: bool) -> None:
    """final split の実行前チェック。両方を要求:
    1. CLI フラグ --confirm-final が指定されている
    2. 環境変数 OCRBENCH_ALLOW_FINAL == "1"
    どちらか欠けたら FinalAccessError（メッセージに「主防壁は OS 分離であり、
    本チェックは誤操作防止の補助である」旨を含める）。"""
```

- T12 の `run_benchmark` 呼び出し前に CLI（T17）がこれを通す。T12 側にも `split=="final"` で
  `ensure_final_allowed` を呼ぶ防御を入れる（このタスクで T12 に 3–5 行の追修正を行ってよい）

### 2. Selection の個別結果秘匿（ADR §8-5）

Selection はエージェントに個別結果を返さず、評価プログラムだけが最良候補を選ぶ。CLI 出力での
漏洩を防ぐため:

```python
def console_summary_for(split: str, summary: RunSummary) -> str:
    """run 完了時にコンソールへ出してよい要約を split 別に構成する。
    dev: 集計値をすべて表示してよい。
    selection / final: n 件・完走・出力先パスのみ表示（率・件数・帳票別情報を出さない）。"""
```

- `write_detail_report`（T14）は selection / final の run ディレクトリでは**呼び出しをエラー**にする
  ガード関数 `ensure_detail_report_allowed(split)` を提供し、T14 の書き出し口で呼ぶ
  （T14 への 2–3 行の追修正を含む）
- report.md 自体は生成してよい（人間の確認用）。エージェントへ渡すのは export（T14 の匿名集計）だけ、
  という運用を docs に明記

### 3. `docs/isolation-windows.md`（H3 の作業手順書）

以下を含む手順書を書く（コマンド例付き。実行は人間）:

1. Final test 専用の Windows ローカルユーザー作成（`net user finaltest ...`）
2. `OCRBENCH_FINAL_DIR` の NTFS ACL 設定例:
   `icacls <dir> /inheritance:r /grant finaltest:(OI)(CI)F /grant Administrators:(OI)(CI)F`
   — 開発ユーザー・プロンプト改善エージェント実行ユーザーからは読み取り不可にする
3. 検証チェックリスト: 開発ユーザーで `dir` が拒否されること、`ocrbench run --split final` が
   フラグ・環境変数なしで FinalAccessError になること、Final GT が第三者照合済みであること（ADR §4）
4. 「CLI フラグは補助。ACL が主防壁」の明記と、エージェントに与える権限の最小化
   （エージェントプロセスは final ディレクトリへの ACL を持つユーザーで動かさない）

## 提供インターフェース

- `resolve_split_dir()`, `ensure_final_allowed()`, `FinalAccessError`,
  `console_summary_for()`, `ensure_detail_report_allowed()`

## 受け入れ基準（DoD）

- [ ] フラグ・環境変数の 4 組合せ（両方 / フラグのみ / env のみ / 両方なし）で final 許可が
      両方あり時のみ通る
- [ ] `resolve_split_dir("final")` が `OCRBENCH_FINAL_DIR` を返し、data_dir を参照しない
- [ ] selection の `console_summary_for` に率・件数（n 以外）・doc_id が含まれない
- [ ] selection / final での詳細レポート生成がエラーになる
- [ ] `docs/isolation-windows.md` に手順 1–4 が揃っている
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_splitguard.py`:
  - 4 組合せの final ゲート（monkeypatch で env 制御）
  - ディレクトリ解決の split 対応
  - `console_summary_for`: dev では率が出る / selection では出ない（文字列検査。
    canary として特徴的な数値を summary に仕込み不在を確認）
  - `ensure_detail_report_allowed` の split 別挙動
  - T12 側防御の結合テスト: fake で `--split final` 相当を env なしで呼び FinalAccessError
