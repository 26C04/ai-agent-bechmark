# T11: データセット manifest・分割・fingerprint

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §4 データセットと分割、§5（データはリポジトリ外） |
| 依存 | T04 |
| 後続 | T12, T16, T17、H2（データ準備で CLI を使用） |
| 主な成果物 | `src/ocrbench/dataset.py` |

## 目的

リポジトリ外データディレクトリの規約を定め、manifest の生成・検証・fingerprint 計算を実装する。
分割（dev 40 / selection 20 / final 40）は固定し、データと manifest のハッシュから
データセット fingerprint を生成する（ADR §4）。

## スコープ

### 含む
- 外部ディレクトリレイアウト規約、manifest モデル、検証、fingerprint、manifest 生成補助
### 含まない
- 実データの収集・GT 作成（H2）、final のアクセスガード（T16）

## 実装指示

### 1. 外部ディレクトリレイアウト（docstring とルート README に記載）

```text
$OCRBENCH_DATA_DIR/            # dev + selection
├── raw/<doc_id>.<pdf|jpg|jpeg|png>
├── gt/<doc_id>.json           # OrderDocument 準拠の正解
└── manifest.json

$OCRBENCH_FINAL_DIR/           # final 専用。同一レイアウト。別 ACL（T16 / H3）
```

### 2. manifest モデル（`dataset.py`、Pydantic）

```python
class DocEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str                      # 匿名 ID: "doc-" + ファイル内容 sha256 先頭 12 桁
    source_kind: Literal["scan", "photo"]
    file: str                        # raw/ からの相対パス
    sha256: str                      # 原本ファイルのハッシュ
    split: Literal["dev", "selection", "final"]

class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int              # 1
    docs: list[DocEntry]
```

- doc_id を内容ハッシュ由来にすることで、元ファイル名（顧客名等を含み得る）が匿名 ID に漏れない
  （ADR §5「匿名 ID」の実装）

### 3. 関数

```python
def load_manifest(root: Path) -> Manifest:
    """manifest.json を読み検証する。検証項目:
    - doc_id 重複なし / raw ファイル存在 / sha256 一致
    - gt/<doc_id>.json が存在し OrderDocument として妥当
    - split がこの root に許される値か（final ディレクトリに dev が混在等は T16 で扱うため、ここでは不問）
    違反は DatasetError（違反の全リストをメッセージへ）。"""

def validate_split_counts(manifest: Manifest, *, strict: bool) -> list[str]:
    """ADR §4 の表と照合し逸脱を文字列リストで返す。
    dev=40(scan20/photo20), selection=20(10/10), final=40(20/20)。
    strict=False では警告リスト（データ収集途中でも使えるように）、strict=True は逸脱があれば DatasetError。"""

def dataset_fingerprint(root: Path, manifest: Manifest) -> str:
    """sha256。入力順に依存しない決定的な計算:
    manifest の正規化 JSON（sort_keys）+ doc_id 昇順に各 raw ファイル sha256 + 各 gt ファイル sha256
    を連結してハッシュ。"""

def build_manifest(root: Path, assignments_csv: Path) -> Manifest:
    """H2 のデータ準備支援。CSV（列: file, source_kind, split）から manifest を生成して返す。
    - file は root/raw/ 配下の実在ファイル
    - doc_id を sha256 から採番し、raw ファイルを <doc_id>.<ext> へリネームする（元名の匿名化）
    - 同一内容（同一 sha256）の重複ファイルは DatasetError（ADR §4: 同一原本の派生は別件と数えない）
    - 書き込みは manifest.json のみ。gt/ の存在チェックはしない（GT 作成前に実行できるように）"""
```

### 4. テスト用ミニデータセット

`tests/helpers.py`（T09 で作成済みのヘルパー）を拡張し、`tmp_path` に合成ミニデータセットを組み立てる
fixture を提供する: 生成画像 + 妥当な GT JSON + manifest（例: dev 4 件 scan2/photo2、selection 2、final 2）。
T12 / T16 / T17 のテストで再利用する。

## 提供インターフェース

- `Manifest`, `DocEntry`, `DatasetError`, `load_manifest()`, `validate_split_counts()`,
  `dataset_fingerprint()`, `build_manifest()`
- `tests/helpers.py` のミニデータセット fixture

## 受け入れ基準（DoD）

- [ ] `load_manifest` が正常セットを通し、壊れたセット（ファイル欠落 / ハッシュ不一致 / GT 不正 /
      doc_id 重複）で全違反を列挙した `DatasetError` を出す
- [ ] `dataset_fingerprint` が docs の並び順を変えても同一値、ファイル 1 バイト変更で別値
- [ ] `build_manifest` が匿名 doc_id 採番・リネーム・重複検出を行う
- [ ] `validate_split_counts` が ADR §4 の表どおりに判定する
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_dataset.py`: 上記 DoD 各ケース（すべて `tmp_path` 上の合成データで実施）
  - fingerprint の決定性（2 回計算・順序入替）と感度（内容変更）
  - build_manifest: CSV から生成 → load_manifest が通る往復テスト、重複ファイルで DatasetError
  - strict / 非 strict の分割数検証
