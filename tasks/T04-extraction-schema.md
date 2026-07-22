# T04: 抽出スキーマとデータ契約

| 項目 | 内容 |
|---|---|
| Phase | 1 コア |
| ADR 参照 | §2 抽出スキーマ |
| 依存 | T01 |
| 後続 | T05, T06, T07, T10, T11, T12 |
| 主な成果物 | `src/ocrbench/schema.py`, `tests/fixtures/public/order_document.schema.json` |

## 目的

モデル出力と正解 JSON の**唯一のデータ契約**を Pydantic v2 で定義し、Ollama の structured outputs へ
渡す JSON Schema と、`needs_review` のシステム算出を提供する。

## スコープ

### 含む
- `LineItem` / `OrderDocument` モデル、JSON Schema 生成、`needs_review` 算出
### 含まない
- 文字列正規化（T05）、生出力のパース（T06）、採点（T07）

## 実装指示

### 1. データモデル（`schema.py`）

```python
class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    part_no: str | None
    material: str | None
    num_pieces: Annotated[int, Field(ge=1)] | None

class OrderDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_name: str | None
    order_no: str | None
    delivery_date: str | None      # 下記 validator 参照
    items: list[LineItem]
```

ADR §2 の評価規則をスキーマとして厳密に写す:

- **全キー必須**（default を与えない）。キー欠落はスキーマ違反。読み取れない値は `null`（ADR: 推測させず null）
- `strict=True` により `"100"` → `100` のような型強制を禁止する（数値は int で出力させる）
- `delivery_date` は `field_validator` で検証: `^\d{4}-\d{2}-\d{2}$` に一致し、かつ
  `datetime.date.fromisoformat` が成功する実在日のみ許可。`null` は許可
- `num_pieces` は 1 以上の int または `null`（桁区切り・単位の除去はモデル側の責務。文字列は受けない）
- `extra="forbid"` → 生成 JSON Schema に `additionalProperties: false` が入る（両モデルとも）
- バウンディングボックス・根拠座標・`needs_review` のフィールドは**定義しない**（ADR: モデルに生成させない）

### 2. Ollama 用 JSON Schema

```python
def ollama_format_schema() -> dict[str, Any]:
    """Ollama chat の format= に渡す JSON Schema を返す。"""
```

- `OrderDocument.model_json_schema()` をベースにする
- 返り値に対する自己検証を入れる: ルートと `$defs` 内 `LineItem` の双方に
  `additionalProperties: false` が含まれること、`required` に全キーが含まれることを assert する
  （Pydantic のバージョン更新で契約が静かに壊れるのを防ぐ）

### 3. スキーマのゴールデンファイル

`tests/fixtures/public/order_document.schema.json` に生成スキーマをコミットし、テストで
`ollama_format_schema()` と完全一致を検証する（`json.dumps(..., sort_keys=True, indent=2)` で正規化）。
意図的にスキーマを変えるときはゴールデンも更新し、PR で差分レビューできるようにする。

### 4. `needs_review` の算出（ADR: モデルでなくシステムが算出）

```python
def compute_needs_review(doc: OrderDocument | None) -> bool:
    """必須値の欠落・検証失敗から needs_review を算出する。

    doc が None（パース・検証失敗）→ True
    customer_name / order_no / delivery_date のいずれかが None → True
    items が空 → True
    いずれかの item の part_no / material / num_pieces が None → True
    それ以外 → False
    """
```

## 提供インターフェース

- `OrderDocument`, `LineItem`, `ollama_format_schema()`, `compute_needs_review()`

## 受け入れ基準（DoD）

- [ ] 正常 JSON（null 含む）が `OrderDocument.model_validate` を通る
- [ ] 未知キー・キー欠落・型違反（`"100"`、`0`、負数、不正日付 `2026-02-30`、形式違反 `2026/07/01`）が
      `ValidationError` になる
- [ ] ゴールデンスキーマがコミットされ、一致テストが通る
- [ ] `compute_needs_review` が docstring の全分岐を実装している
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_schema.py`:
  - 受理: 全項目あり / 各フィールド null / items 複数 / items 空配列（validate は通る。needs_review 側で拾う）
  - 拒否: 上記 DoD の各違反ケース（`pytest.raises(ValidationError)` のテーブル駆動）
  - `ollama_format_schema()` のゴールデン一致、`additionalProperties: false` の存在
  - `compute_needs_review` の全分岐（None / ヘッダー null / items 空 / item 内 null / 完全）
