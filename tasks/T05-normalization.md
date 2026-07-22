# T05: 正規化ユーティリティ

| 項目 | 内容 |
|---|---|
| Phase | 1 コア |
| ADR 参照 | §2 抽出スキーマ（評価規則） |
| 依存 | T04 |
| 後続 | T07（採点）、H2（正解 JSON 作成支援） |
| 主な成果物 | `src/ocrbench/normalize.py` |

## 目的

ADR §2 の評価規則のうち「比較前の正規化」を一箇所に実装する。原則は
**「前後の空白だけを除去し、それ以外は一切変換しない」**。過剰な正規化は厳密比較の趣旨に反するため、
やらないことをテストで固定する。

## スコープ

### 含む
- 比較用正規化（trim のみ）、GT 作成支援用の日付・数量正規化
### 含まない
- 名寄せ・表記ゆれ吸収（ADR で明示的に禁止）、採点そのもの（T07）

## 実装指示

`normalize.py` に以下を実装する。

### 1. 比較用正規化（採点で使用）

```python
def strip_outer_whitespace(value: str) -> str:
    """前後の空白のみ除去する。内部空白・全角/半角・大小文字・記号は変更しない。

    Python の str.strip()（Unicode 空白。全角空白 U+3000 を含む）に準拠する。
    """

def comparable_header(doc: OrderDocument) -> tuple[str | None, str | None, str | None]:
    """(customer_name, order_no, delivery_date) を trim 済みで返す。None はそのまま。"""

def comparable_items(doc: OrderDocument) -> Counter[tuple[str | None, str | None, int | None]]:
    """items を (part_no, material, num_pieces) の多重集合（Counter）として返す。

    文字列は trim 済み。順序は評価せず、同一明細の重複件数は保持する（ADR §2）。
    """
```

### 2. GT 作成・再検証支援（CLI `dataset validate` や H2 で使用）

```python
def normalize_delivery_date(value: str) -> str | None:
    """人が転記した日付表記を YYYY-MM-DD へ正規化する。

    受理: "2026-07-01"、"2026-7-1"、"2026/07/01"、"2026/7/1"、"2026.07.01"（ゼロ埋めして返す）
    拒否（None）: 年なし（"7/1"）、和暦、実在しない日付、それ以外の形式。
    推測はしない（ADR: 年は帳票上で一意に読めることが前提）。
    """

def normalize_num_pieces(value: int | str) -> int | None:
    """数量表記を 1 以上の int へ正規化する。

    int: 1 以上ならそのまま、0 以下は None。
    str: 桁区切り（"," と "，" と空白）と末尾の単位語（"個", "枚", "本", "pcs", "PCS", "pc", "個入"）を
    除去してから int 化。全角数字は半角化する。失敗・0 以下は None。
    """
```

- 単位語リストはモジュール定数 `UNIT_SUFFIXES` として公開し、H2 で追補できるようにする
- これらは **GT 作成側の道具**であり、モデル出力へは適用しない（モデル出力はスキーマ検証（T04）で
  既に正規形に強制されている）。docstring にその旨を明記する

### 3. やらないことの固定

以下を**変換しないこと**をテストで保証する（ADR の厳密比較規則の裏面）:

- 内部空白（`"A B"` ≠ `"AB"` のまま）
- 全角/半角（`"ＡＢＣ"` ≠ `"ABC"` のまま）※数量の全角数字のみ例外（上記 2.）
- 大小文字（`"abc"` ≠ `"ABC"` のまま）
- 記号（`"A-1"` ≠ `"A−1"` のまま）

## 提供インターフェース

- `strip_outer_whitespace()`, `comparable_header()`, `comparable_items()`,
  `normalize_delivery_date()`, `normalize_num_pieces()`, `UNIT_SUFFIXES`

## 受け入れ基準（DoD）

- [ ] trim が前後の半角・全角空白（U+3000）・タブ・改行を除去し、内部空白を保持する
- [ ] `comparable_items` が順序非依存・重複保持（Counter）である
- [ ] 日付正規化が docstring の受理 / 拒否例をすべて満たす
- [ ] 数量正規化が docstring の受理 / 拒否例をすべて満たす
- [ ] 「やらないこと」のテストが存在する
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_normalize.py`（テーブル駆動）:
  - trim: `"  A 1  "` → `"A 1"`、`"　品番　"` → `"品番"`、内部の `　` は保持
  - items: 順序入替で等価、重複 2 件と 1 件は不等価
  - 日付: 受理 5 形式 + 拒否（`"R8.7.1"`, `"7/1"`, `"2026-13-01"`, `"20260701"`）
  - 数量: `"1,000個"` → 1000、`"１００ 枚"` → 100、`" 250 pcs"` → 250、`"0"` → None、`"約100"` → None
  - 不変性: 全角/半角・大小文字・記号・内部空白が変換されないこと
