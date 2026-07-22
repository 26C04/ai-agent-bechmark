# ADR-0001: Ollama を用いた生産指示書 OCR ベンチマーク基盤

- Status: Accepted
- Date: 2026-07-21

## Context

研究室プロジェクトでは、生産指示書の内容を読み取り、将来的に EDI へ自動登録する
Agentic workflow を検討している。その第一段階として、ローカルで動作する複数の
Vision Language Model (VLM) とプロンプトを、再現可能かつ機密情報を漏らさず比較する
ベンチマーク基盤が必要である。

今回のベンチマークは OCR・構造化抽出までを対象とする。EDI への登録、操作成功率、
誤登録防止は別ベンチマークとする。

## Decision

### 1. 評価対象

- 1 ページ、1 レイアウト、印字のみの生産指示書を対象とする。
- 入力はスキャン PDF またはスマートフォン写真とする。
- Ollama 上の VLM へ画像を直接入力し、構造化 JSON を生成する。
- 手書き、複数ページ、未知レイアウト、値段、EDI 自動登録は対象外とする。
- 人が画像だけを見て対象項目を判読できる帳票だけを主データセットに含める。

### 2. 抽出スキーマ

モデルが生成するデータの論理スキーマを次のようにする。

```json
{
  "customer_name": "原本どおりの文字列",
  "order_no": "原本どおりの文字列",
  "delivery_date": "YYYY-MM-DD",
  "items": [
    {
      "part_no": "原本どおりの文字列",
      "material": "原本どおりの文字列",
      "num_pieces": 100
    }
  ]
}
```

以下の評価規則を適用する。

- 文字列は前後の空白だけを除去し、内部空白、全角・半角、大小文字、記号は厳密に比較する。
- `customer_name` は名寄せせず、帳票上の文字列をそのまま保持する。
- `delivery_date` は `YYYY-MM-DD` に正規化する。年は帳票上で一意に読めることを前提とする。
- `num_pieces` は 1 以上の整数とし、桁区切りと単位表記を除去する。
- `items` の順序は評価しないが、同一明細の重複件数は保持する。比較には多重集合を使う。
- 読み取れない値を推測させず `null` とする。
- `needs_review` はモデルに生成させず、必須値の欠落や検証失敗からシステムが算出する。
- JSON Schema では未知項目を禁止する (`additionalProperties: false`)。
- JSON の前後にある説明文、Markdown、コードフェンスは救済せず形式エラーとする。
- JSON Schema 違反時の再試行は最大 1 回とし、正解値やヒントは渡さない。
- バウンディングボックスや根拠座標は出力させない。

### 3. 共通画像前処理

モデル間で同じ最小前処理を使用する。

- PDF は固定 300 DPI で画像化する。
- 写真は EXIF の向きを反映する。
- RGB 形式へ統一し、縦横比を維持する。
- 内容依存の切り抜き、台形補正、傾き補正、鮮明化は初期版では行わない。
- 前処理のバージョンと所要時間を実行結果へ記録する。

### 4. データセットと分割

異なる原本 100 件を目標とし、スキャン PDF 50 件、スマートフォン写真 50 件を収集する。
同一原本の撮り直しや派生画像は別件として数えない。

| Split | 合計 | スキャン | 写真 | 用途 |
|---|---:|---:|---:|---|
| Development | 40 | 20 | 20 | エラー分析とプロンプト改善 |
| Selection | 20 | 10 | 10 | 候補の盲検選定 |
| Final test | 40 | 20 | 20 | 研究発表用の最終評価 |

- 実在する顧客名・品番・材質の候補一覧をモデルへ渡さない。
- 可能なら、開発データにない顧客名も最終テストへ含める。
- 最終テストの正解 JSON は作成者とは別の人が原本と照合する。
- Development と Selection も JSON Schema 検証と抜き取り確認を行う。
- 分割は固定し、各データと manifest のハッシュからデータセット fingerprint を生成する。

### 5. 機密情報と評価汚染の防止

- 実帳票、正解 JSON、個別予測、詳細差分はリポジトリ外の管理対象外ディレクトリへ保存する。
- 最終テストは別 Windows ユーザーと NTFS ACL で隔離し、プロンプト改善エージェントから
  物理的に読み取れないようにする。
- CLI 内のフラグだけをテストロックとして信用しない。
- クラウド型エージェントへ渡すのは匿名化した集計とエラー分類だけとし、画像、正解値、
  誤読値を渡さない。
- `.gitignore` に加えて Lefthook でも、禁止パス、実データ形式、大容量ファイルの誤 commit を拒否する。
- 匿名 fixture は `tests/fixtures/public/` のような明示した場所だけ Git 管理を許可する。
- GitHub リポジトリは研究発表までは非公開とする。

### 6. モデルと実行環境

基準マシンは次の構成とする。

- Windows 11 Home（ネイティブ実行）
- Intel Core Ultra 7 265K
- RAM 64 GB
- NVIDIA GeForce RTX 5070 Ti
- NVMe SSD 2 TB

初期の許可モデルは次の 5 つとする。

- `gemma4:12b`
- `qwen3.5:9b`
- `minicpm-v4.5:8b`
- `glm-ocr:bf16`
- `ministral-3:14b`

モデルタグだけでなく、pull 後に解決された digest を manifest に固定する。モデルと画像・KV cache を
含めて GPU に完全に載り、実行時に 100% GPU と確認できた構成だけを主ベンチマークへ参加させる。
CPU・RAM offload が発生した結果は参考枠とする。

Mistral OCR 4 は公開 Ollama モデルではないため初期許可リストへ含めない。企業向けセルフホスト環境を
利用できる場合に、異なる実行基盤として別部門で評価する。

Ollama のバージョンはベンチマーク期間中固定する。OS と GPU driver の情報は記録するが、それらの
更新だけを理由に全モデルの再実行は強制しない。

### 7. 推論プロトコル

- 同時実行数は 1 とする。
- `temperature` は 0、seed は固定、thinking は無効とする。
- モデル digest、量子化、コンテキスト長、画像前処理、その他の推論設定を固定する。
- エージェントが自動変更できるのはプロンプト本文だけとする。
- ラベル名や「表の材質列」のような意味的な位置ヒントは許可する。
- 絶対座標、特定画像向けの指示、実在値を含む few-shot example は禁止する。
- モデル読込時間とウォーム状態の処理時間を分けて記録する。
- 速度は前処理開始から、再試行と検証を経て有効な JSON が得られるまでを計測する。
- 初期合格ラインはウォーム状態で 1 帳票 30 秒以内、将来目標は 3 秒以内とする。

最終候補だけ固定テストを 3 回実行する。事前指定した第 1 回の結果を主精度とし、残り 2 回は
再現一致率と精度の振れ幅に使用する。多数決で結果を改善しない。

### 8. プロンプト自動改善

特定の `/goal` 実装やエージェント製品には依存しない。エージェントは共通 Python CLI を実行し、
匿名集計からプロンプト候補を作る。

1. 共通初期プロンプトで 5 モデルを Development 40 件に対して一次評価する。
2. 精度と 30 秒制約から上位 2 モデルを選ぶ。
3. 上位 2 モデルについて、Development の匿名集計だけを使い最大 10 候補を生成する。
4. 候補生成終了後に Selection 20 件を一括実行する。
5. Selection の個別結果をエージェントへ返さず、評価プログラムが最良候補を自動選択する。
6. 選択済み構成を、隔離された Final test 40 件で評価する。

各候補プロンプトは削除や破壊的上書きをせず、内容ハッシュ付きで版管理する。`active` の参照だけを
自動更新し、ロールバック可能にする。

自動採用条件は次のとおりとする。

- JSON 不正件数が増えない。
- 帳票単位の完全一致数が 1 件以上改善する。
- `order_no`、`part_no`、`num_pieces` の項目精度が悪化しない。
- 同点なら出力トークン数と処理時間が少ない候補を採用する。
- 条件を満たす候補がなければ既存の active prompt を維持する。

改善ループは、最大 10 候補、Development で 3 候補連続改善なし、2 時間経過、完全一致率 100%、
または実行エラー 3 回連続のいずれかで終了する。

### 9. 評価指標

主指標は帳票単位の完全一致率とする。ヘッダー、明細数、全明細の全項目が正しい場合だけ成功と数える。

補助指標として次を記録する。

- JSON Schema 妥当率と初回 JSON 成功率
- 再試行後成功率
- 項目別正解率
- スキャン・写真別の正解率
- 欠落明細、余分な明細、文字置換などのエラー分類
- ウォーム処理時間の p50 と p95
- モデル読込時間、前処理時間、検証・再試行時間
- prompt/output token 数と生成 token/秒
- 3 回の再現一致率
- GPU 内実行の確認結果

最終テストでは帳票完全一致率に Wilson の 95% 信頼区間を付ける。同じ帳票に対するモデル差は
対応あり比較を使い、項目別指標は帳票単位の固定 seed bootstrap で信頼区間を算出する。

初期版ではモデル精度に根拠のない絶対合格値を置かない。基盤の完成条件は、100 件を再現可能に評価し、
入力種別・精度・速度・安定性・プロンプト履歴を比較できることである。

### 10. レポートと実験履歴

数値と表は Python CLI が決定的に生成し、エージェントは考察だけを生成する。

```text
runs/
└── <UTC timestamp>_<run-id>/
    ├── manifest.json
    ├── results.json
    ├── report.md
    └── analysis.md
```

- `results.json` をメトリクスの正本とする。
- `report.md` は CLI が実験条件と集計表を生成する。
- `analysis.md` はエージェントが匿名集計から失敗傾向と次の改善案を書く。
- 共有可能な成果物には集計値、匿名 ID、モデル・環境情報だけを含める。
- 正解・予測差分と画像参照を含む詳細レポートは Git 管理外のローカル領域へ保存する。
- 初期版では MLflow 等の専用サーバーを導入しない。

### 11. 実装と品質管理

- Python 3.14 と `uv` を使用し、`pyproject.toml` と `uv.lock` を Git 管理する。
- インターフェースは CLI のみとする。
- 公式 Ollama Python client を薄い adapter で包み、接続先を明示的に localhost へ固定する。
- 通常テストでは adapter を fake に差し替える。
- Ruff を lint と format に使用する。
- Mypy は strict mode で使用する。
- pytest と coverage を使用し、全体の coverage 下限を 80% とする。
- 採点、正規化、分割、テストロックの決定的ロジックは重点的にテストする。
- 実 Ollama を使うテストは `ollama` marker の明示実行に限定し、通常の pytest、Lefthook、CI に含めない。

Lefthook は次のように構成する。

- pre-commit: 変更ファイルへ `ruff check --fix` と `ruff format`
- pre-push: 全体へ `ruff check`、`ruff format --check`、`mypy --strict`、`pytest`

GitHub Actions は `ubuntu-latest` 上で lockfile 検証、Ruff、Mypy、pytest を実行する。実データ、GPU、
Ollama model は CI で使用しない。初期版では CD、PyPI 公開、自動デプロイを行わない。

## Implementation Order

1. 匿名 fixture で CLI、JSON Schema、採点、fake Ollama、レポート、CI、Lefthookを構築する。
2. Windows 上で 1 モデル・1 帳票の Ollama smoke test を行う。
3. 外部データディレクトリへ 100 件を準備し、正解と固定分割を検証する。
4. 5 モデルの一次評価を行う。
5. 上位 2 モデルのプロンプトを自動改善する。
6. 盲検 Selection と隔離済み Final test を実行する。
7. 研究発表用レポートを生成する。

## Consequences

### Positive

- OCR 精度、構造化出力、速度を実際の業務項目で比較できる。
- 評価コードと Agentic な改善処理を分離できる。
- Selection と Final test への過学習を抑え、研究発表で評価手順を説明できる。
- 機密帳票を GitHub やクラウド型エージェントへ送らずに済む。
- モデルやエージェント基盤を交換しても同じ CLI とデータ契約を再利用できる。

### Negative

- 初期結果は 1 レイアウト、1 ページ、印字帳票にしか一般化できない。
- Final test の OS レベル分離には、Windows ユーザーと ACL の運用が必要になる。
- 上位 2 モデルの自動改善でも、多数のローカル推論と数時間の GPU 時間を要する可能性がある。
- バウンディングボックス、未知レイアウト、EDI 自動登録の品質は別途評価が必要になる。

## References

- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Vision Models](https://ollama.com/search?c=vision)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Mistral OCR 4](https://mistral.ai/news/ocr-4/)
- [uv project management](https://docs.astral.sh/uv/guides/projects/)
- [Ruff](https://docs.astral.sh/ruff/)
