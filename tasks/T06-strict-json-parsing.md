# T06: 厳密 JSON パースとスキーマ検証

| 項目 | 内容 |
|---|---|
| Phase | 1 コア |
| ADR 参照 | §2 抽出スキーマ（形式エラー規則） |
| 依存 | T04 |
| 後続 | T07, T12 |
| 主な成果物 | `src/ocrbench/parsing.py` |

## 目的

モデルの生出力文字列を**救済なし**で判定する層を作る。
「JSON の前後にある説明文、Markdown、コードフェンスは救済せず形式エラーとする」（ADR §2）を
実装とテストで固定する。

## スコープ

### 含む
- 生出力 → `ParseResult` の変換、失敗分類
### 含まない
- 再試行制御（T12 の責務）、採点（T07）

## 実装指示

`parsing.py` に以下を実装する。

```python
class ParseStatus(StrEnum):
    OK = "ok"                            # 妥当な JSON かつ OrderDocument に適合
    NOT_JSON = "not_json"                # JSON として読めない（フェンス・前後の説明文を含む）
    SCHEMA_VIOLATION = "schema_violation"  # JSON だが OrderDocument に適合しない

@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    document: OrderDocument | None       # OK のときのみ非 None
    error: str | None                    # 失敗時の要約（先頭 500 文字に切り詰め）
    raw_output: str                      # モデル生出力（results.json 保存用）

def parse_model_output(raw: str) -> ParseResult: ...
```

判定規則:

1. `json.loads(raw)` を**そのまま**呼ぶ。JSON 仕様上の前後空白（space / tab / LF / CR）だけが許容され、
   それ以外の前置き・後置き・コードフェンスは `NOT_JSON` になる。
   **事前の strip・フェンス除去・正規表現抽出などの救済コードを書いてはならない**
2. `json.loads` が成功したら `OrderDocument.model_validate` にかける。
   `ValidationError` は `SCHEMA_VIOLATION`（エラー要約を `error` へ）
3. トップレベルが dict 以外（配列・文字列・数値）も 2. に流し、`SCHEMA_VIOLATION` とする
4. どの分岐でも例外を外へ漏らさない（`ParseResult` で返す）

補足:

- `raw_output` を保持するのは、results.json（T12）でエラー分類・再現分析に使うため。
  クラウドへ出す匿名集計（T14）には含まれない
- BOM 付き出力（`﻿{...}`）は Python の `json.loads` では失敗する。これも救済せず `NOT_JSON` とする
  （テストで固定）

## 提供インターフェース

- `ParseStatus`, `ParseResult`, `parse_model_output()`

## 受け入れ基準（DoD）

- [ ] コードフェンス付き・前後説明文付き・BOM 付き出力が `NOT_JSON` になる
- [ ] 未知キー・キー欠落・型違反・不正日付が `SCHEMA_VIOLATION` になる
- [ ] 正常出力（前後に JSON 仕様内の空白・改行があるもの含む）が `OK` になる
- [ ] 実装に「救済」コード（strip / フェンス除去 / 正規表現による JSON 抽出）が存在しない
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_parsing.py`（テーブル駆動）:
  - `NOT_JSON`: ` ```json {...} ``` `、`"以下が結果です:\n{...}"`、`"{...}\n以上です"`、
    `"﻿{...}"`、空文字列、`"{"`（途切れ）
  - `SCHEMA_VIOLATION`: 未知キー追加、`items` 欠落、`num_pieces: "100"`、`num_pieces: 0`、
    `delivery_date: "2026/07/01"`、トップレベル配列 `[{...}]`
  - `OK`: 完全な出力、`"\n  {...}\n"`（前後空白のみ）、null を含む出力
  - `OK` 時に `document` が非 None、失敗時に None であること
