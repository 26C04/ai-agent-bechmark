# T13: プロンプトレジストリ（版管理）

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §8 プロンプト自動改善（版管理・禁止事項）、§7（位置ヒント規則） |
| 依存 | T01 |
| 後続 | T12, T15, T17、H4（エージェントが CLI 経由で使用） |
| 主な成果物 | `src/ocrbench/prompts.py`, `prompts/`（初期プロンプト含む） |

## 目的

プロンプトを**内容ハッシュ付き・追記専用**で版管理する。削除・破壊的上書きを不可能にし、
`active` の参照だけを自動更新してロールバック可能にする（ADR §8）。

## スコープ

### 含む
- レジストリ実装、共通初期プロンプト、禁止パターンの静的チェック
### 含まない
- 候補生成そのもの（エージェント = H4 の仕事）、採用判定（T15）

## 実装指示

### 1. ディレクトリ形式（リポジトリ内 `prompts/`。Git 管理する）

プロンプト本文は実在値を含まない規約（ADR §7）なので Git 管理してよい。

```text
prompts/
└── <name>/                # 例: base, gemma4-tuned
    ├── <hash12>.txt       # 本文。ファイル名 = 内容の sha256 先頭 12 桁
    ├── active             # 現在の hash12 を書いた 1 行テキスト
    └── history.jsonl      # 追記専用: {"ts": "...", "action": "add|activate|rollback", "hash": "...", "note": "..."}
```

### 2. API（`prompts.py`）

```python
@dataclass(frozen=True)
class PromptVersion:
    name: str
    hash: str          # sha256(text) 先頭 12 桁
    text: str

class PromptRegistryError(OcrBenchError): ...

def add_prompt(registry_root: Path, name: str, text: str, note: str = "") -> PromptVersion:
    """本文を保存し history に追記する。同一ハッシュが既にあれば何もせず既存を返す（冪等）。
    既存ファイルと同名で内容が異なる事態はハッシュ衝突以外あり得ないため、その場合は即エラー。
    既存 <hash>.txt の内容変更・削除に相当する操作は API として提供しない。"""

def get_active(registry_root: Path, name: str) -> PromptVersion:
    """active が指す版を返す。active 未設定・参照先欠落はエラー。"""

def activate(registry_root: Path, name: str, hash12: str, note: str = "") -> None:
    """active を更新し history に追記する。存在しない hash はエラー。"""

def rollback(registry_root: Path, name: str) -> PromptVersion:
    """history 上で activate/rollback を遡り、一つ前に active だった版へ戻す。無ければエラー。"""

def list_versions(registry_root: Path, name: str) -> list[PromptVersion]: ...

def lint_prompt(text: str) -> list[str]:
    """禁止事項（ADR §7）の機械検出可能な近似チェック。警告文字列を返す（エラーにはしない）:
    - 絶対座標らしき表現: 正規表現で `x=`, `y=`, `px`, `bbox`, `座標` を検出
    - few-shot の実在値混入は機械判定できないため、"example" / "例:" を検出したら
      「実在値を含む few-shot は禁止（ADR §7）。合成値であることを確認せよ」と注意を返す
    判定は保守的な警告に留め、採否は人間がレビューする。"""
```

- `registry_root` は既定でリポジトリの `prompts/` を指すが、テスト容易性のため引数で受ける
- history.jsonl の `ts` は UTC。追記のみで書き換えない

### 3. 共通初期プロンプト（`prompts/base/`）

一次評価（ADR §8-1）で 5 モデル共通に使う初期プロンプト v1 を日本語で作成し、
`add_prompt` の形式どおりに配置（`<hash12>.txt` + `active` + `history.jsonl`）する。内容要件:

- 生産指示書の画像から JSON のみを出力させる（説明文・コードフェンス禁止を明記）
- スキーマの各キーと型、`null` 規則（読み取れない値は推測せず null。ADR §2）
- `delivery_date` は `YYYY-MM-DD`、`num_pieces` は桁区切り・単位を除いた整数
- 位置ヒントは「表の材質列」のような**一般的な意味ヒントのみ**（ADR §7 で許可された範囲）
- 実在の顧客名・品番・材質の例を**含めない**（ADR §4・§7）。値の例が必要なら明らかな架空値を使う
- ハッシュがファイル名と一致することをテストで検証する

## 提供インターフェース

- `PromptVersion`, `add_prompt()`, `get_active()`, `activate()`, `rollback()`, `list_versions()`,
  `lint_prompt()`, `PromptRegistryError`

## 受け入れ基準（DoD）

- [ ] add → activate → add(v2) → activate(v2) → rollback で v1 に戻る一連が動く
- [ ] 同一本文の add が冪等（ファイル増殖なし・history には追記される）
- [ ] 既存版の内容を変更・削除する公開 API が存在しない
- [ ] `prompts/base/` に初期プロンプトが配置され、ハッシュ整合・lint 警告ゼロ
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_prompts.py`（tmp_path 上のレジストリ）:
  - DoD の一連操作、冪等性、未知 hash の activate エラー、active 未設定エラー、rollback 履歴なしエラー
  - history.jsonl が追記専用で全操作を記録していること
  - `lint_prompt`: 座標表現・example 検出の警告、クリーンな本文で警告なし
  - リポジトリ実体 `prompts/base/` の整合テスト（ファイル名 = sha256 先頭 12 桁、active の解決成功）
