# T18: smoke コマンドと Windows セットアップ手順書

| 項目 | 内容 |
|---|---|
| Phase | 3 統合 |
| ADR 参照 | §6 モデルと実行環境、§7 推論プロトコル、Implementation Order 2 |
| 依存 | T17 |
| 後続 | H1（実機での smoke 実施）、H2 以降の全運用 |
| 主な成果物 | `src/ocrbench/smoke.py`, `docs/windows-setup.md` |

## 目的

Windows 実機で「1 モデル・1 帳票」を検証する `ocrbench smoke` を実装し（Implementation Order 2）、
基準マシンのセットアップ〜smoke 完了までを人間が迷わず実行できる手順書を書く。

## スコープ

### 含む
- smoke サブコマンド、実 Ollama 疎通テスト（`ollama` marker）、Windows 手順書
### 含まない
- データセット一括実行（`run` で実施済み）、ACL 隔離手順（T16 の docs/isolation-windows.md）

## 実装指示

### 1. `ocrbench smoke`（`smoke.py` + cli.py への登録）

```text
ocrbench smoke --model <tag> --image <path> [--prompt-name base]
```

1 枚の画像 / PDF に対しフルパイプラインを 1 回通し、診断情報を人間可読で表示する:

- サーバー: `server_version()`、モデル: tag と `resolve_digest()` の結果
- GPU 配置: `gpu_placement()` — `fully_on_gpu` でなければ **警告を強調表示**
  （ADR §6: 100% GPU でない構成は参考枠）
- 前処理: source_kind / 出力寸法 / PREPROCESS_VERSION / 所要時間
- 推論: load_duration（コールド判定付き）/ infer 時間 / tokens / tokens/sec
- パース結果: ParseStatus、OK なら整形 JSON を表示、`needs_review` の算出値
- 総所要時間と **30 秒制約（ADR §7）の判定**
- 終了コード: パース OK かつ例外なし → 0、それ以外 → 1（ガード拒否は 2）

実装は既存モジュールの組み合わせのみ（preprocess / adapter / parsing / schema）。
`--model` は `ALLOWED_MODELS` 検証をする。ただし smoke に限り `--allow-any-model` で回避可
（動作確認用途。run には存在しない）。

- fake でも動く（`OCRBENCH_ADAPTER=fake`）: E2E テストで使用
- 実 Ollama の統合テスト `tests/test_smoke_real.py` を `@pytest.mark.ollama` で 1 本
  （前提: サーバー起動済み + 対象モデル pull 済み。docstring に明記）

### 2. `docs/windows-setup.md`（H1 の作業手順書）

基準マシン（Windows 11 Home / Core Ultra 7 265K / RAM 64GB / RTX 5070 Ti / NVMe 2TB。ADR §6）
向けの手順を、コピペ可能なコマンド付きで書く:

1. **前提確認**: `nvidia-smi` で GPU / driver 版数確認（記録手順含む）
2. **uv + Python 3.14**: uv インストール → `uv sync` → `uv run ocrbench --version`
3. **Ollama 導入と版固定**: インストール後 `ollama --version` を記録。
   ベンチマーク期間中はアップデートしない（自動更新の無効化手順を含める。ADR §6）
4. **モデル pull**（5 種。ADR §6）:
   ```text
   ollama pull gemma4:12b
   ollama pull qwen3.5:9b
   ollama pull minicpm-v4.5:8b
   ollama pull glm-ocr:bf16
   ollama pull ministral-3:14b
   ```
   pull 後 `ollama list` で **digest を記録**（タグではなく digest を固定する。ADR §6）
5. **環境変数**: `OCRBENCH_DATA_DIR` / `OCRBENCH_RUNS_DIR` の作成と setx 例
   （`OCRBENCH_FINAL_DIR` は docs/isolation-windows.md を参照させる）
6. **smoke 実行**: 手元の匿名サンプル帳票 1 枚で
   `uv run ocrbench smoke --model gemma4:12b --image <path>` → チェックリスト
   （digest 表示 / 100% GPU / 30 秒以内 / ParseStatus）
7. **GPU 常駐確認**: smoke 直後に `ollama ps` で `100% GPU` を目視確認する手順
8. **実 Ollama テスト**: `uv run pytest -m ollama --no-cov` の実行（任意）
9. **トラブルシュート**: VRAM 不足時（CPU offload → 参考枠扱い。ADR §6）、
   think 非対応エラー時の扱い（T10 の方針: 設定変更は人間が判断）
10. 次のステップへの導線: H2（データ準備）→ `tasks/README.md` の運用タスク表

## 提供インターフェース

- `ocrbench smoke` サブコマンド（H1 のチェックリストが依存）

## 受け入れ基準（DoD）

- [ ] fake adapter で smoke が終了コード 0、診断項目がすべて表示される
- [ ] fake の不正応答で終了コード 1、GPU 非 100% 設定で警告表示
- [ ] `--allow-any-model` の有無で許可外モデルの扱いが変わる
- [ ] `docs/windows-setup.md` に手順 1〜10 が揃い、ADR §6 の 5 モデルと digest 記録手順がある
- [ ] 品質ゲート（ruff / mypy strict / pytest / coverage 80%）を全て通過

## テスト要件

- `tests/test_smoke_cmd.py`（fake）: 正常 / 不正応答 / GPU 非 100% / 許可外モデル ±フラグ
- `tests/test_smoke_real.py`（`@pytest.mark.ollama`）: 実サーバーで smoke 相当を 1 回実行し
  終了コード 0（環境前提を docstring に明記。CI では実行されない）
