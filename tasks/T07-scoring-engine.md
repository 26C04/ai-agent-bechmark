# T07: 採点エンジン

| 項目 | 内容 |
|---|---|
| Phase | 1 コア |
| ADR 参照 | §2 抽出スキーマ（評価規則）、§9 評価指標（エラー分類） |
| 依存 | T05, T06 |
| 後続 | T08, T12 |
| 主な成果物 | `src/ocrbench/scoring.py` |

## 目的

正解 (`gt`) と予測 (`pred`) の帳票 1 件分を決定的に採点する。主指標「帳票単位の完全一致」と、
補助指標（項目別正解・エラー分類）の**単位データ**をここで確定させる。集計は T08 が行う。

## スコープ

### 含む
- 帳票 1 件の採点、明細の多重集合比較と貪欲マッチング、エラータグ付け
### 含まない
- 率・信頼区間などの集計（T08）、needs_review（T04 で定義済み、runner が記録）

## 実装指示

`scoring.py` に以下を実装する。比較の前処理は **T05 の `comparable_header` / `comparable_items` のみ**を
使う（trim 以外の変換をしない）。

### 1. 結果型

```python
@dataclass(frozen=True)
class DocumentScore:
    exact_match: bool                            # 主指標: ヘッダー全項目 + 明細多重集合が完全一致
    header_correct: dict[str, bool]              # {"customer_name": .., "order_no": .., "delivery_date": ..}
    items_exact: bool                            # 明細多重集合の完全一致
    item_field_counts: dict[str, tuple[int, int]]  # field -> (correct, gt_total)。field は part_no/material/num_pieces
    missing_items: int                           # GT にあり予測に無い明細数
    extra_items: int                             # 予測にあり GT に無い明細数
    error_tags: frozenset[str]                   # 下記 3. の分類
```

```python
def score_document(gt: OrderDocument, parse_result: ParseResult) -> DocumentScore: ...
```

- `parse_result.status != OK` の場合: `exact_match=False`、`header_correct` 全 False、
  `items_exact=False`、`item_field_counts` は (0, GT セル数)、`missing_items=len(gt.items)`、
  `extra_items=0`、`error_tags` に `"not_json"` または `"schema_violation"`
- ヘッダー比較: trim 後の厳密一致。`None == None` は一致とみなす（GT はデータセット規約上
  原則非 null だが、スキーマ上あり得るため定義しておく）

### 2. 明細のマッチング（決定的な貪欲法）

`items` の順序は評価しない。多重集合（Counter）で以下の手順によりペアを作る。

1. `(part_no, material, num_pieces)` の完全一致キーを `min(gt_count, pred_count)` 件ずつペア化して除去
2. 残りから `part_no` が一致する組をペア化（複数候補は (material, num_pieces) の文字列表現の
   辞書順ソートで先頭から。決定性のため）
3. さらに残りは (material, num_pieces) 一致 → それも無ければ辞書順ソート同士の位置合わせでペア化
4. ペアにならなかった GT 側の残り = `missing_items`、予測側の残り = `extra_items`

- 各ペアについて field ごとの正誤を数え、`item_field_counts` に加算する
  （`gt_total` は GT の明細数 × 1 field なので `len(gt.items)`）
- `items_exact` は手順 1 で全件ペア化され残りが無いこと（= Counter が等しいこと）と同値
- このマッチングは順序性のあるヒューリスティックであること、ただし入力が同じなら常に同じ結果を
  返すことを docstring に明記する

### 3. エラータグ（ADR §9「欠落明細、余分な明細、文字置換などのエラー分類」）

以下の規則で `error_tags` を付ける（複数可）:

| タグ | 条件 |
|---|---|
| `not_json` / `schema_violation` | parse 失敗（ParseStatus に対応） |
| `missing_item` | `missing_items > 0` |
| `extra_item` | `extra_items > 0` |
| `null_field` | 不一致箇所のうち予測側が None のものがある |
| `char_substitution` | 文字列 field（customer_name / order_no / part_no / material）で双方非 None かつ不一致 |
| `date_error` | delivery_date が不一致 |
| `count_mismatch` | ペア化された明細の num_pieces が双方非 None かつ不一致 |

### 4. 完全一致の定義（ADR §9）

`exact_match = (parse OK) and (ヘッダー 3 項目すべて正解) and items_exact`

## 提供インターフェース

- `DocumentScore`, `score_document()`

## 受け入れ基準（DoD）

- [ ] 順序を入れ替えただけの明細が完全一致になる
- [ ] 重複明細（同一明細 2 件 vs 1 件）が不一致になり `missing_item` が付く
- [ ] 同一入力で常に同一の `DocumentScore` が返る（決定性）
- [ ] parse 失敗時の規定値（上記 1.）どおりに返る
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_scoring.py`（テーブル駆動）:
  - 完全一致 / 順序入替一致 / trim 差のみ一致
  - ヘッダー 1 項目違い（`char_substitution`）、日付違い（`date_error`）
  - 明細: 欠落・余分・重複件数違い・num_pieces 違い（`count_mismatch`）・part_no 置換
  - 予測 null（`null_field`）、items 空予測（`missing_item` × N）
  - `not_json` / `schema_violation` の規定値
  - `item_field_counts` の数え上げ検証（例: 明細 3 件中 part_no 2 件正解 → ("part_no", (2, 3))）
  - 決定性: 同一入力 2 回で dataclass が等価
