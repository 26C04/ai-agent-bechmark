# T08: メトリクス集計と統計

| 項目 | 内容 |
|---|---|
| Phase | 1 コア |
| ADR 参照 | §9 評価指標 |
| 依存 | T07 |
| 後続 | T14, T15, T17 |
| 主な成果物 | `src/ocrbench/metrics.py` |

## 目的

帳票単位の採点結果（T07）と実行計測（T12 が生成、ここでは型だけ前提とする）から、ADR §9 の
主指標・補助指標・信頼区間を**純 Python で決定的に**計算する。numpy / scipy は使わない。

## スコープ

### 含む
- 統計プリミティブ（Wilson / bootstrap / McNemar / percentile）、run 集計、3 回実行の再現一致率
### 含まない
- results.json の読み書き（T12）、Markdown 整形（T14）

## 実装指示

`metrics.py` に以下を実装する。

### 1. 統計プリミティブ

```python
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval。n == 0 は ValueError。"""
```

実装式（実装ミス防止のため固定）: p̂ = successes/n として

```text
center = (p̂ + z²/2n) / (1 + z²/n)
half   = z / (1 + z²/n) × sqrt(p̂(1−p̂)/n + z²/4n²)
→ (max(0, center−half), min(1, center+half))
```

```python
def percentile(values: Sequence[float], q: float) -> float:
    """nearest-rank 法（ceil(q/100 × n) 番目、1-indexed）。決定的。空列は ValueError。"""

def bootstrap_ci(
    per_doc_values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float],
    n_boot: int = 2000,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """帳票単位リサンプルの percentile 法 95% CI（2.5, 97.5）。random.Random(seed) で決定的。"""

def mcnemar_exact(a_only: int, b_only: int) -> float:
    """対応あり比較（exact McNemar, 両側）。

    a_only = モデル A のみ完全一致した帳票数、b_only = B のみ。
    p = 2 × Σ_{k=0..min(a,b)} C(n, k) × 0.5^n  （n = a_only + b_only）を 1.0 で clip。
    n == 0 は p = 1.0。math.comb を使用。
    """
```

### 2. run 集計

T12 が確定する「帳票 1 件分の結果」を受け取る。T12 完了前にこのタスクを実装するため、
**このタスクでは入力を Protocol で定義**し、T12 の `DocumentResult` がそれを満たす形にする:

```python
class DocResultLike(Protocol):
    @property
    def doc_id(self) -> str: ...

    @property
    def source_kind(self) -> str: ...  # "scan" | "photo"

    @property
    def first_attempt_status(self) -> ParseStatus: ...

    @property
    def final_status(self) -> ParseStatus: ...

    @property
    def attempt_count(self) -> int: ...  # 1 or 2

    @property
    def prediction(self) -> Mapping[str, object] | None: ...

    @property
    def warm(self) -> bool: ...

    @property
    def timings_ms(self) -> Mapping[str, float]: ...

    @property
    def tokens(self) -> Mapping[str, int | None]: ...

    @property
    def tokens_per_sec(self) -> float | None: ...

    @property
    def needs_review(self) -> bool: ...

    def parsed_score(self) -> DocumentScore:
        """Restore and return the persisted document score."""
```

```python
@dataclass(frozen=True)
class RunSummary:
    n_docs: int
    exact_match_count: int
    exact_match_rate: float
    exact_match_ci: tuple[float, float]            # Wilson
    schema_valid_rate: float                       # final_status == OK の率
    first_attempt_json_rate: float                 # first_attempt_status == OK の率
    after_retry_success_rate: float | None         # None when no document was retried
    header_field_accuracy: dict[str, float]
    item_field_accuracy: dict[str, float]          # item_field_counts の合算から
    by_source_kind: dict[str, SourceKindSummary]   # scan / photo 別の n・exact率・schema妥当率
    error_tag_counts: dict[str, int]
    latency_warm_p50_ms: float | None
    latency_warm_p95_ms: float | None
    mean_timings_ms: dict[str, float]
    token_totals: dict[str, int]
    mean_tokens_per_sec: float | None
    needs_review_count: int
```

- `after_retry_success_rate` は分母 0（再試行が 1 件も無い）のとき `None`（`float | None`）とする
- p50 / p95 は **warm な帳票の total_ms のみ**を対象にする（ADR §9「ウォーム処理時間の p50 と p95」）
- If no documents are warm, both warm percentile fields are `None`.
- 集計は入力順に依存しない（dict は doc_id ソートで構築）

```python
def aggregate(results: Sequence[DocResultLike]) -> RunSummary: ...
def item_field_bootstrap_ci(results: Sequence[DocResultLike], field: str) -> tuple[float, float]:
    """項目別指標の帳票単位固定 seed bootstrap CI（ADR §9）。"""
def paired_comparison(a: Sequence[DocResultLike], b: Sequence[DocResultLike]) -> PairedComparison:
    """同一帳票集合に対する 2 モデルの対応あり比較。doc_id で突合し、exact_match の
    a_only / b_only / both / neither と mcnemar_exact の p 値を返す。突合不能は ValueError。"""
```

### 3. 再現一致率（ADR §7: 最終候補の 3 回実行）

```python
@dataclass(frozen=True)
class ReproSummary:
    n_docs: int
    identical_prediction_rate: float   # 3 run 全てで正規化後予測 JSON が一致した帳票の率
    exact_match_rates: tuple[float, ...]  # 各 run の完全一致率（振れ幅の確認用）

def reproducibility(runs: Sequence[Sequence[DocResultLike]]) -> ReproSummary: ...
```

- 「正規化後予測 JSON の一致」は、`OrderDocument.model_dump()` を trim（T05）してから
  `json.dumps(sort_keys=True)` した文字列の一致で判定。パース失敗は「失敗ステータス同士なら一致」
  とせず**不一致扱い**とする（保守的に）。docstring に明記

## 提供インターフェース

- 上記すべて（`wilson_interval`, `percentile`, `bootstrap_ci`, `mcnemar_exact`, `DocResultLike`,
  `RunSummary`, `aggregate`, `item_field_bootstrap_ci`, `paired_comparison`, `ReproSummary`, `reproducibility`）

## 受け入れ基準（DoD）

- [ ] Wilson: 既知値と一致（例: 8 成功 / 10 件, z=1.96 → 約 (0.4902, 0.9433)。小数 3 桁で検証）
- [ ] percentile: nearest-rank の定義どおり（[1..10] の p50 = 5, p95 = 10）
- [ ] bootstrap: 同 seed で 2 回呼んで同一区間、異 seed で（一般に）異なる区間
- [ ] McNemar: (a_only=5, b_only=1) の p 値を手計算値と照合、(0,0) → 1.0
- [ ] `aggregate` が全フィールドを正しく埋める（下記テストのミニ fixture で全数値を手計算検証）
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_metrics.py`:
  - 統計プリミティブ 4 種の既知値テスト（上記 DoD）
  - `aggregate`: 6 件程度のミニ結果（scan 3 / photo 3、exact 4、retry 2 件中 1 成功、warm 5 件）で
    全フィールドを手計算と照合。入力順シャッフルで結果不変
  - `after_retry_success_rate` の分母 0 → None
  - `paired_comparison`: doc_id 突合、a_only/b_only の数え上げ、doc_id 不一致で ValueError
  - `reproducibility`: 3 run 一致 / 1 run だけ違う / パース失敗を含む の 3 ケース
- テスト用の `DocResultLike` は軽量なローカル dataclass で偽装してよい（T12 に依存しない）
