# T09: 画像前処理パイプライン

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §3 共通画像前処理 |
| 依存 | T01 |
| 後続 | T12, T18 |
| 主な成果物 | `src/ocrbench/preprocess.py` |

## 目的

全モデル共通の**最小前処理**を実装する。PDF は固定 300 DPI で画像化、写真は EXIF の向きを反映、
RGB へ統一、縦横比維持。内容依存の切り抜き・台形補正・傾き補正・鮮明化は行わない（ADR §3）。

## スコープ

### 含む
- PDF / JPEG / PNG → PNG バイト列への変換、前処理バージョンと所要時間の記録
### 含まない
- リサイズ・圧縮・画質改善の類（初期版では一切禁止）、複数ページ対応（対象外）

## 実装指示

`preprocess.py` に以下を実装する。

```python
PREPROCESS_VERSION: Final[str] = "v1"   # 挙動を変えたら必ず上げる（results.json に記録される）

@dataclass(frozen=True)
class PreprocessResult:
    png_bytes: bytes
    width: int
    height: int
    source_kind: Literal["pdf", "image"]
    duration_ms: float
    version: str                         # PREPROCESS_VERSION

class PreprocessError(OcrBenchError): ...

def preprocess(path: Path) -> PreprocessResult: ...
```

処理規則:

1. 拡張子で分岐: `.pdf` → PDF 処理、`.jpg` / `.jpeg` / `.png` → 画像処理、それ以外は `PreprocessError`
2. **PDF**（`pypdfium2`）:
   - 1 ページのみ許可。2 ページ以上は `PreprocessError`（ADR: 複数ページは対象外）
   - `scale = 300 / 72` でレンダリング（固定 300 DPI）
   - レンダリング結果を Pillow Image にして RGB 化
3. **画像**（Pillow）:
   - `ImageOps.exif_transpose()` で EXIF Orientation を反映
   - `convert("RGB")` で RGB 統一（RGBA / グレースケール / CMYK も RGB へ）
4. リサイズ・切り抜きをしない（寸法は入力のまま。縦横比は自動的に維持される）
5. PNG（可逆）へエンコードして `png_bytes` にする
6. `duration_ms` は関数先頭〜エンコード完了を `time.perf_counter()` で計測
7. 例外（破損ファイル・読めない形式）はすべて `PreprocessError` に包んで送出

## テスト用 fixture の方針（重要）

バイナリのコミットを避けるため、**テストコード内で生成**する。

- PDF: 最小の 1 ページ空白 PDF をバイト列リテラルとして `tests/` 内ヘルパーに定数で持つ
  （`%PDF-1.4` から始まる手書きの数百バイト。MediaBox `[0 0 612 792]` = Letter。
  2 ページ版も同様に定数で用意）
- JPEG / PNG: Pillow で生成（例: 200×100 の塗り潰し画像）。EXIF Orientation=6（90° 回転）付き JPEG は
  `Image.Exif` に `0x0112 = 6` を設定して `save(exif=...)` で生成

## 提供インターフェース

- `PREPROCESS_VERSION`, `PreprocessResult`, `PreprocessError`, `preprocess()`

## 受け入れ基準（DoD）

- [ ] Letter サイズ PDF（612×792 pt）→ 300 DPI で約 2550×3300 px（±2 px 許容）
- [ ] Orientation=6 の 200×100 JPEG → 100×200 に回転され、EXIF なし PNG になる
- [ ] RGBA PNG・グレースケール JPEG が RGB PNG になる
- [ ] 2 ページ PDF・`.txt`・破損 JPEG が `PreprocessError`
- [ ] `duration_ms > 0` かつ `version == PREPROCESS_VERSION`
- [ ] リポジトリに画像・PDF バイナリがコミットされていない（T02 のガードとも整合）
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_preprocess.py`: 上記 DoD の各ケース。生成 fixture ヘルパーは
  `tests/helpers.py`（または conftest.py の fixture）に置き、T11 / T12 のテストからも再利用できるようにする
