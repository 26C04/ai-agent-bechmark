# T10: Ollama adapter と fake 実装

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §6 モデルと実行環境、§7 推論プロトコル、§11（adapter / fake / ollama marker） |
| 依存 | T04 |
| 後続 | T12, T17, T18 |
| 主な成果物 | `src/ocrbench/ollama_adapter.py` |

## 目的

公式 Ollama Python client を**薄い adapter で包み**、接続先を localhost に固定する（ADR §11)。
通常テストは fake に差し替え、実 Ollama を使うテストは `ollama` marker の明示実行に限定する。

## スコープ

### 含む
- adapter Protocol、real 実装、fake 実装、digest 解決、GPU 配置確認
### 含まない
- 再試行・時間集計のロジック（T12）、モデルの pull（手順書 T18 / H1）

## 実装指示

`ollama_adapter.py` に以下を実装する。

### 1. 型と Protocol

```python
OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"   # 変更手段を提供しない（ADR §11: localhost 固定）

@dataclass(frozen=True)
class ChatResult:
    content: str                      # モデル生出力（そのまま T06 へ渡す）
    prompt_tokens: int | None         # prompt_eval_count
    output_tokens: int | None         # eval_count
    total_duration_ns: int | None
    load_duration_ns: int | None      # モデル読込時間の分離記録に使う（ADR §7）
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None

@dataclass(frozen=True)
class GpuPlacement:
    fully_on_gpu: bool                # 100% GPU なら True（ADR §6: 主ベンチマーク参加条件）
    detail: str                       # "100% GPU" / "55%/45% CPU/GPU" など ps の生情報

class OllamaAdapter(Protocol):
    def generate_structured(
        self, *, model: str, prompt: str, image_png: bytes,
        format_schema: dict[str, Any], seed: int, temperature: float = 0.0,
    ) -> ChatResult: ...
    def resolve_digest(self, model: str) -> str: ...
    def gpu_placement(self, model: str) -> GpuPlacement | None: ...
    def server_version(self) -> str: ...
```

### 2. `RealOllamaAdapter`

- `ollama.Client(host=OLLAMA_HOST)` を内部生成。host を外から注入する引数は**作らない**
- `generate_structured` は `client.chat()` を 1 回呼ぶ:
  - `messages=[{"role": "user", "content": prompt, "images": [image_png]}]`
  - `format=format_schema`（structured outputs）
  - `options={"temperature": temperature, "seed": seed}`
  - `think=False`（thinking 無効。ADR §7。モデルが think 非対応でもエラーにならないよう、
    非対応エラー時は think 指定なしで 1 回だけ呼び直す実装は不可 — **例外はそのまま上げる**。
    対応可否は smoke（T18）で判明させ、設定変更は人間が行う）
  - `stream=False`、`keep_alive="10m"`（ウォーム維持。定数として公開）
- 応答から `ChatResult` の各フィールドを詰める（欠けているフィールドは None）
- `resolve_digest`: `client.list()` の結果からモデルタグ完全一致で digest を返す。
  見つからなければ `OcrBenchError` 派生の `ModelNotFoundError`
- `gpu_placement`: `client.ps()` の結果から該当モデルの VRAM 配置を判定。
  `size_vram >= size` なら `fully_on_gpu=True`。モデルが ps に居なければ None
- `server_version`: `client.version()`（無ければ `/api/version` 相当のフィールド）から取得
- ollama ライブラリの応答は属性アクセス（typed）を基本にし、mypy strict を通す

### 3. `FakeOllamaAdapter`（通常テストの主役）

```python
class FakeOllamaAdapter:
    """スクリプト化された応答を順に返す fake。OllamaAdapter Protocol を満たす。"""
    def __init__(self, responses: Sequence[str] | None = None, *,
                 digest: str = "sha256:fake", fully_on_gpu: bool = True) -> None: ...
    calls: list[FakeCall]              # 呼び出し記録（model, prompt, schema, seed, image のサイズ）
```

- `responses` を先頭から返す。尽きたら `IndexError` でなく明示的な `AssertionError`（テスト作法）
- 応答ごとに擬似メトリクス（tokens = len(content) // 4、duration は固定値）を付ける
- 便宜コンストラクタ `FakeOllamaAdapter.always(json_text)` （常に同じ応答）を用意
- fake は**プロダクションコードと同じモジュール**に置いてよい（E2E で `OCRBENCH_ADAPTER=fake` として
  使うため。ADR の「通常テストでは adapter を fake に差し替える」の実装先）

### 4. adapter 選択ヘルパー

```python
def create_adapter(kind: str | None = None) -> OllamaAdapter:
    """kind または env OCRBENCH_ADAPTER（既定 "real"）から adapter を生成する。
    fake の場合は「常に有効な空 JSON を返す」既定応答にする。"""
```

## 提供インターフェース

- `OllamaAdapter`, `RealOllamaAdapter`, `FakeOllamaAdapter`, `ChatResult`, `GpuPlacement`,
  `create_adapter()`, `OLLAMA_HOST`, `ModelNotFoundError`

## 受け入れ基準（DoD）

- [ ] `RealOllamaAdapter` に接続先を変える公開手段が無い（localhost 固定）
- [ ] fake が Protocol を満たす（`x: OllamaAdapter = FakeOllamaAdapter()` が mypy を通る）
- [ ] 呼び出し記録に seed / temperature / schema が残り、テストから検証できる
- [ ] 実 Ollama テストが `@pytest.mark.ollama` 付きで、通常 pytest では実行されない
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_ollama_adapter.py`（fake 中心・通常実行）:
  - スクリプト応答が順に返る / 尽きたら AssertionError / `always` の動作
  - `calls` の記録内容（temperature=0、seed=config.SEED が渡ること）
  - `create_adapter("fake")` / env 切替 / 不明値で ConfigError
- `tests/test_ollama_real.py`（`@pytest.mark.ollama`）:
  - `server_version()` が文字列を返す
  - `resolve_digest` が存在モデルで sha256 文字列、不在モデルで ModelNotFoundError
  - ※ モデル pull 済みが前提。docstring に前提を書く。CI では実行されない
