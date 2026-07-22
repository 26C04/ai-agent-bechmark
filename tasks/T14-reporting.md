# T14: report.md 生成と匿名集計エクスポート

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §10 レポートと実験履歴、§5（クラウドへ渡せるのは匿名集計のみ）、§9（指標） |
| 依存 | T08, T12 |
| 後続 | T17、H4（エージェントへの入力を生成）、H5 |
| 主な成果物 | `src/ocrbench/reporting.py` |

## 目的

数値と表を **CLI が決定的に生成**する（エージェントは考察 `analysis.md` だけを書く。ADR §10）。
クラウド型エージェントへ渡してよい「匿名集計」を allowlist 方式で機械的に保証する（ADR §5）。

## スコープ

### 含む
- `report.md` 生成、匿名集計 JSON エクスポート、詳細差分レポート（ローカル専用）、`analysis.md` 雛形
### 含まない
- 考察文の生成（エージェント / 人間）、複数 run の比較 UI（compare-runs の表は T17 で薄く実装）

## 実装指示

`reporting.py` に以下を実装する。入力は T12 の `RunResults` と T08 の `RunSummary`。

### 1. `report.md`（run ディレクトリ内に生成）

```python
def render_report(results: RunResults, summary: RunSummary) -> str: ...
def write_report(run_dir: Path) -> Path:      # load_run → aggregate → render → run_dir/report.md
```

内容（すべて表・数値。decision に必要な情報を過不足なく）:

- 実験条件表: model_tag / model_digest / prompt_name / prompt_hash / split / dataset_fingerprint /
  preprocess_version / ollama_version / seed / temperature / participation（reference なら目立つ注記）/
  os_info / gpu_info / adapter_kind（fake なら「テスト実行」と明記）
- 主指標: 帳票完全一致率 + Wilson 95% CI、n
- 補助指標表: schema 妥当率 / 初回 JSON 成功率 / 再試行後成功率 / needs_review 件数
- 項目別正解率表（ヘッダー 3 項目 + 明細 3 項目。明細側は bootstrap CI 付き）
- スキャン・写真別の表
- エラー分類の件数表
- レイテンシ表: warm p50 / p95、平均内訳（preprocess / load / infer / parse_validate）、
  **30 秒制約（ADR §7）超過の帳票数**
- token 表: prompt / output 合計と平均、tokens/sec 平均

決定性の規則:

- 表の行は必ずキーの辞書順・doc_id 昇順などの固定順で出す
- 浮動小数は固定フォーマット（率は `%.1f%%`、CI は `%.3f`、時間は `%.0f ms`）
- 実行時刻など入力に無い値を勝手に入れない（`started_at_utc` は manifest から転記）
- 同一入力 → バイト単位で同一出力（ゴールデンテストで固定）

### 2. 匿名集計エクスポート（ADR §5 の核心）

```python
def export_anonymous_summary(results: RunResults, summary: RunSummary) -> dict[str, Any]:
    """クラウド型エージェントへ渡してよい集計だけを allowlist 方式で構築する。

    含む: RunManifest の全項目（機微なし）、RunSummary の全集計値、エラー分類件数、
          帳票ごとの (doc_id, source_kind, exact_match, final_status, error_tags, needs_review,
          total_ms, warm) — doc_id は匿名 ID なので可。
    含まない（禁止）: prediction、raw_outputs、GT 由来の値、差分文字列、ファイルパス。
    実装は「RunResults から必要フィールドだけを手で写す」構築とし、dict の一括コピーを使わない。"""

def write_anonymous_summary(run_dir: Path, out_path: Path) -> Path: ...
```

- 検証関数 `assert_no_forbidden_content(payload, results)` を実装し、エクスポート時に必ず通す:
  ペイロードの JSON 文字列に、prediction / raw_outputs / GT の**いずれの文字列値も出現しない**ことを
  走査して確認する（防御的な二重チェック）

### 3. 詳細差分レポート（ローカル専用）

```python
def render_detail_report(results: RunResults, gt_by_doc: Mapping[str, OrderDocument]) -> str: ...
def write_detail_report(run_dir: Path, data_root: Path) -> Path:   # run_dir/report_detail.md
```

- 帳票ごとに GT と予測の全項目差分・エラータグ・raw 出力の有無を表示（エラー分析用）
- **runs ディレクトリ（リポジトリ外）にのみ書き、エクスポート対象にしない**。ファイル冒頭に
  「機密: リポジトリ・クラウドへ持ち出し禁止（ADR §5）」の注意書きを埋め込む

### 4. `analysis.md` 雛形

`write_report` 実行時、`analysis.md` が無ければ空テンプレート（見出しのみ: 失敗傾向 / 次の改善案 /
入力に使った匿名集計の hash）を生成する。既存なら触らない（エージェント / 人間の編集領域）。

## 提供インターフェース

- `render_report()`, `write_report()`, `export_anonymous_summary()`, `write_anonymous_summary()`,
  `render_detail_report()`, `write_detail_report()`

## 受け入れ基準（DoD）

- [ ] ゴールデンテスト: 固定入力 → `report.md` がバイト一致（`tests/fixtures/public/golden_report.md`）
- [ ] 匿名集計に「金糸雀（canary）」が漏れない: GT・予測・raw 出力へ埋めた一意文字列が
      エクスポート JSON に出現しないことを機械検証
- [ ] 詳細レポートには canary が出る（≒ 情報が落ちていない）
- [ ] `analysis.md` の新規生成と既存保護
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_reporting.py`:
  - fake run（T12 の成果物を tmp_path に生成）→ write_report → ゴールデン比較
  - 決定性: 2 回レンダーで同一、documents の順序入替で同一
  - canary テスト（DoD のとおり）: 匿名集計に不在 / 詳細レポートに存在
  - participation="reference" と adapter_kind="fake" の注記が report.md に出ること
  - 30 秒超過帳票数の集計（fake の duration を操作して検証）
