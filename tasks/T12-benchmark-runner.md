# T12: ベンチマーク実行エンジンと results.json

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §7 推論プロトコル、§10 レポートと実験履歴、§2（再試行規則） |
| 依存 | T06, T07, T09, T10, T11, T13 |
| 後続 | T14, T15, T16, T17 |
| 主な成果物 | `src/ocrbench/runner.py`, `src/ocrbench/results.py` |

## 目的

1 run（= 1 モデル × 1 プロンプト × 1 split）を逐次・決定的に実行し、`runs/<UTC>_<run-id>/` に
`manifest.json` と `results.json`（メトリクスの正本）を書き出す。

## スコープ

### 含む
- 逐次実行ループ、再試行（最大 1 回）、時間・token 計測、GPU 配置確認、成果物書き出し
### 含まない
- report.md / analysis.md の生成（T14）、final split の許可判定（T16。ここでは T16 のフックを呼ぶだけ）

## 実装指示

### 1. 結果データモデル（`results.py`、Pydantic。すべて `extra="forbid"`）

```python
class DocumentResult(BaseModel):
    doc_id: str
    source_kind: Literal["scan", "photo"]
    first_attempt_status: ParseStatus
    final_status: ParseStatus
    attempt_count: int                      # 1 or 2（ADR §2: 再試行は最大 1 回）
    prediction: dict[str, Any] | None       # 検証済み OrderDocument の dump（失敗時 None）
    raw_outputs: list[str]                  # 各試行の生出力（機密。リポジトリ外にのみ存在）
    score: dict[str, Any]                   # DocumentScore の直列化（T07）
    needs_review: bool                      # T04 compute_needs_review
    warm: bool                              # load_duration_ns < WARM_THRESHOLD_NS で判定
    timings_ms: dict[str, float]            # preprocess / load / infer / parse_validate / total
    tokens: dict[str, int | None]           # prompt / output
    tokens_per_sec: float | None            # output_tokens / eval_duration
    error: str | None                       # adapter 例外などの実行時エラー

class RunManifest(BaseModel):
    run_id: str
    started_at_utc: str                     # "YYYYMMDDTHHMMSSZ"
    model_tag: str
    model_digest: str                       # 解決済み digest（ADR §6: タグでなく digest を固定）
    prompt_name: str
    prompt_hash: str                        # T13 の内容ハッシュ
    split: Literal["dev", "selection", "final"]
    dataset_fingerprint: str
    preprocess_version: str
    ollama_version: str
    seed: int
    temperature: float
    gpu_fully_loaded: bool | None           # ADR §6: False なら参考枠
    participation: Literal["primary", "reference"]  # gpu_fully_loaded に基づく
    os_info: str                            # platform.platform()
    gpu_info: str | None                    # nvidia-smi --query-gpu=name,driver_version ベストエフォート
    adapter_kind: Literal["real", "fake"]

class RunResults(BaseModel):
    schema_version: int                     # 1
    manifest: RunManifest
    documents: list[DocumentResult]
```

- `DocumentResult` は T08 の `DocResultLike` Protocol を満たすアクセサを持つこと
  （`score` を `DocumentScore` に戻す `parsed_score()` ヘルパー等、突合方法は実装裁量。
  ただし T08 の `aggregate(results)` に渡せる形を必ず提供する）

### 2. 実行ループ（`runner.py`）

```python
def run_benchmark(
    *, adapter: OllamaAdapter, data_root: Path, manifest: Manifest,
    split: Literal["dev", "selection", "final"], model_tag: str,
    prompt: PromptVersion, runs_root: Path, run_id: str | None = None,
) -> Path:
    """run を実行し、出力ディレクトリ（runs_root/<UTC>_<run-id>/）の Path を返す。"""
```

規則（ADR §7）:

1. `model_tag` が `config.ALLOWED_MODELS` に無ければ即エラー
2. 開始時に `resolve_digest` / `server_version` / `gpu_placement` を取得し RunManifest を確定。
   `gpu_placement.fully_on_gpu` が False なら `participation="reference"` とし、警告ログを出して続行
3. 対象 split の docs を **doc_id 昇順**で逐次処理（同時実行 1。並列化しない）
4. 各 doc:
   - `preprocess()` → 計測
   - `adapter.generate_structured(model, prompt.text, png, ollama_format_schema(), seed=config.SEED)`
   - `parse_model_output()` → `NOT_JSON` / `SCHEMA_VIOLATION` なら**同一入力で 1 回だけ再送**
     （正解値・エラー内容・ヒントを一切加えない。ADR §2）→ 最終 status 確定
   - `score_document()`、`compute_needs_review()` を記録
   - 時間: `preprocess`（T09 計測値）、`load`（ChatResult.load_duration_ns 合計）、
     `infer`（prompt_eval + eval）、`parse_validate`（パース・検証・再試行往復の壁時計）、
     `total`（preprocess 開始 → 有効 JSON 確定または最終失敗。ADR §7 の速度定義）
   - `warm` 判定: 全試行の `load_duration_ns` の最大が `WARM_THRESHOLD_NS`（既定 1 秒。定数公開）未満
   - adapter 例外は doc 単位で捕捉し `error` に記録して続行（run 全体は落とさない）
5. 出力: `runs_root/<started_at>_<run_id>/` に `manifest.json` と `results.json`
   （`RunResults.model_dump_json(indent=2)`）。`run_id` 省略時は `uuid4().hex[:8]`
6. 30 秒制約（ADR §7）は**記録のみ**で足切りしない（判定は T14 のレポートで表示）

### 3. 読み戻し

```python
def load_run(run_dir: Path) -> RunResults: ...
```

T14 / T15 / T17 が使う。schema_version 不一致は明示エラー。

## 提供インターフェース

- `RunResults`, `RunManifest`, `DocumentResult`, `run_benchmark()`, `load_run()`, `WARM_THRESHOLD_NS`

## 受け入れ基準（DoD）

- [ ] fake adapter + ミニデータセット（T11 の fixture）で run が完走し、manifest.json / results.json が
      規定スキーマで生成される
- [ ] 1 回目 `NOT_JSON` → 2 回目 OK のシナリオで `attempt_count=2`、`first_attempt_status=not_json`、
      `final_status=ok` になる
- [ ] 2 回とも失敗で `final_status` が失敗のまま、run は継続する
- [ ] fake の呼び出し記録から、再送時にプロンプトが**変わっていない**ことを検証できる
- [ ] `gpu_fully_loaded=False` の fake で `participation="reference"` になる
- [ ] 許可外モデルタグが即エラー
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_runner.py`: 上記 DoD 全ケース（すべて fake + tmp_path。実 Ollama 不使用）
  - results.json を `load_run` で読み戻すラウンドトリップ
  - doc 処理順が doc_id 昇順であること（fake の呼び出し記録で検証）
  - adapter 例外 doc の `error` 記録と続行
  - `DocumentResult` が T08 `aggregate()` にそのまま渡ることの結合テスト 1 本
