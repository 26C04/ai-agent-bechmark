# T15: 候補自動採用・盲検選定・停止判定

| 項目 | 内容 |
|---|---|
| Phase | 2 実行系 |
| ADR 参照 | §8 プロンプト自動改善 |
| 依存 | T08, T13 |
| 後続 | T17、H4（改善ループの判定部として使用） |
| 主な成果物 | `src/ocrbench/selection.py` |

## 目的

プロンプト改善ループの**判定をすべて決定的なコード**にする: Development での自動採用条件、
Selection での盲検最良選択、ループ終了条件。エージェントの裁量を「プロンプト本文の生成」だけに
限定する（ADR §8）。

## スコープ

### 含む
- 採用判定 / 最良選択 / 停止判定 / ループ状態の永続化モデル
### 含まない
- run の実行（T12）、候補プロンプトの生成（エージェント = H4）

## 実装指示

`selection.py` に以下を実装する。入力は T08 の `RunSummary`（+ プロンプト参照）。

### 1. 自動採用判定（Development ループ内。ADR §8 の条件を全て実装）

```python
@dataclass(frozen=True)
class CandidateEval:
    prompt: PromptVersion
    summary: RunSummary

@dataclass(frozen=True)
class AdoptionDecision:
    adopted: bool
    reasons: tuple[str, ...]     # 判定根拠（満たした / 落ちた条件を全て列挙）

def is_adoptable(baseline: CandidateEval, candidate: CandidateEval) -> AdoptionDecision:
    """ADR §8 の自動採用条件:
    1. JSON 不正件数（n - schema_valid 件数）が増えない
    2. 帳票完全一致数が baseline + 1 以上
    3. order_no / part_no / num_pieces の項目精度が悪化しない
       （order_no は header_field_accuracy、part_no / num_pieces は item_field_accuracy）
    4. （タイブレークは pick_best 側。ここでは 1–3 の AND）
    全条件を評価し、reasons に全条件の判定を記録する。"""
```

### 2. 盲検選定（Selection 20 件の一括評価後。ADR §8-5）

```python
def pick_best(candidates: Sequence[CandidateEval]) -> CandidateEval:
    """評価プログラムによる決定的な最良候補選択。ソートキー（優先順）:
    1. 帳票完全一致数（多いほど良い）
    2. JSON 不正件数（少ないほど良い）
    3. order_no / part_no / num_pieces の項目精度の合計（高いほど良い）
    4. 出力 token 合計（少ないほど良い）  ← ADR「同点なら出力トークン数と処理時間が少ない候補」
    5. warm p50 時間（短いほど良い）
    6. prompt_hash の辞書順（最終タイブレーク。決定性保証）
    空列は ValueError。"""
```

- 個別帳票の結果はこの関数に**入力しない**（集計のみで判定 = エージェントに個別結果を返さない
  設計を型で強制する）

### 3. 停止判定（ADR §8 の終了条件）

```python
class LoopState(BaseModel):
    """改善ループの状態。JSON ファイルとして runs 領域に永続化し、CLI 呼び出し間で引き継ぐ。"""
    model_config = ConfigDict(extra="forbid")
    started_at_utc: str
    candidates_tried: int              # 生成済み候補数
    consecutive_no_improve: int
    consecutive_errors: int
    best_exact_match_rate: float
    baseline_prompt_hash: str

class StopReason(StrEnum):
    MAX_CANDIDATES = "max_candidates"          # 最大 10 候補
    NO_IMPROVEMENT = "no_improvement"          # Development で 3 候補連続改善なし
    TIME_LIMIT = "time_limit"                  # 開始から 2 時間経過
    PERFECT = "perfect"                        # 完全一致率 100%
    ERRORS = "errors"                          # 実行エラー 3 回連続

def should_stop(state: LoopState, *, now_utc: datetime) -> StopReason | None:
    """いずれかの終了条件に該当すれば StopReason を返す。now_utc は呼び出し側が渡す（テスト容易性）。"""

def update_after_candidate(state: LoopState, decision: AdoptionDecision) -> LoopState: ...
def update_after_error(state: LoopState) -> LoopState: ...
```

- 上限値（10 候補 / 3 連続 / 2 時間 / 3 エラー）はモジュール定数として公開する
- `LoopState` の読み書きヘルパー `load_loop_state(path)` / `save_loop_state(path, state)` を用意

### 4. 条件を満たす候補が無い場合

`is_adoptable` が全候補 False のとき既存 active を維持するのは**呼び出し側（CLI / H4）の規約**。
`selection.py` には「採用ゼロなら active を変えない」ことをテストで担保するヘルパー
`apply_adoption(registry_root, name, baseline, candidates) -> PromptVersion`（採用時のみ activate を
呼び、採用後の active 版を返す）を実装する。

## 提供インターフェース

- `CandidateEval`, `AdoptionDecision`, `is_adoptable()`, `pick_best()`, `LoopState`, `StopReason`,
  `should_stop()`, `update_after_candidate()`, `update_after_error()`, `apply_adoption()`,
  `load_loop_state()` / `save_loop_state()`

## 受け入れ基準（DoD）

- [ ] 採用条件 1–3 の各違反で不採用になり、reasons に根拠が残る
- [ ] `pick_best` がタイブレーク 6 段をすべて実装し、完全同点でも決定的
- [ ] 停止条件 5 種がそれぞれ単独で発火する
- [ ] `apply_adoption` が採用ゼロで active を変更しない
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_selection.py`:
  - `is_adoptable`: 条件マトリクス（JSON 不正増 / 完全一致 +0 / +1 / 3 項目のどれか悪化 / 全合格）
  - `pick_best`: 各タイブレーク段で決まるケースを 1 つずつ + 完全同点 → hash 辞書順
  - `should_stop`: 境界値（10 個目 / 3 回目 / ちょうど 2 時間 / 100% / エラー 3 連続、およびそれぞれの直前）
  - `LoopState` の保存・読込ラウンドトリップ
  - `apply_adoption`: 採用あり（activate が呼ばれ history に残る）/ 採用なし（active 不変）
