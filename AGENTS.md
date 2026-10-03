# AGENTS.md — stock_comp_2026_v0 作業ハンドオーバー

この md は新しいセッションへの引き継ぎ用。ここに書かれた確定事項は既にユーザーと詰めたものであり、
勝手に変更しないこと。作業指示は都度ユーザーが行う。

## 1. プロジェクト概要

- **目的**: Stock Competition 2026（日本株498銘柄・翌営業日→翌々営業日の寄付リターンの市場残差を予測、
  日次5分位ロング・ショート + 片道0.1%コストで年率 Sharpe 採点）でスコアを上げる。
  仮提出（+0.754）は完了済み。あとは精度向上のみ。
- **開発方法**: v0.app をコード生成に使い、プログラムを Web アプリ上で公開してもらい DL してローカル反映。
  プロンプトと参照資料は GitHub 公開リポジトリ `naphazoline-hcl/q-tr` に置き、初回メッセージで URL を渡す。
  v0 は 1 セッション 1〜3 メッセージ、1 メッセージ 25 分、fast モード対応。
  v0 は入力データを持たない前提で、実行・計測はローカルで行う。
- **環境**: `.venv`（Python 3.13 + 配布固定 5 パッケージ。配布要件は 3.11 だがバージョン同一のため等価。ただし lightgbm は実採点環境に合わせ **4.1.0**）。
- **現状**: `strategies/v0_multifactor`（仮提出 robust_multifactor のコピー、Valid +0.754）が出発点。

## 2. ファイルマップ

| ファイル | 内容・状態 |
| :--- | :--- |
| `v0/計画.md` | v0.app 運用計画（制約対策・セッション順序・フィードバック・中断対策・実行ログ） |
| `v0/prompts/P1〜P6*.md` | v0.app に投げる6つのプロンプト（日本語・自己完結・1セッション1成果物・React アプリ + ZIP DL 指定） |
| `v0/messages/S1〜S6_初回メッセージ.md` | 各セッションの初回メッセージ定型文（`{...}` は実測値で埋める） |
| `docs/contest_spec.md` | 採点式・提出形式・禁止事項（ローカル作成・v0 参照用） |
| `docs/data_schema.md` | `tools/profile_data.py` の自動生成（列名・型・欠損・実例・target 定義の検算つき） |
| `docs/baseline.md` | サンプル実測 + 自作ベースライン（Train OOS 約+2.4 / Valid +0.754）+ 目標 |
| `docs/coding_conventions.md` | API 契約・因果性・メモリ・進捗ファイル・出力形式 |
| `tools/score.py` | 採点レポート（公式数式そのまま + RankIC/年別/分位）。公式との差は 8e-4 未満 |
| `tools/walkforward.py` | purged walk-forward（2010〜2015・purge 260日・進捗ファイルつき） |
| `tools/check_lookahead.py` | 禁止パターンの静的検査 + predict 実行時の読込ガード |
| `tools/check_truncation.py` | 特徴量の truncation invariance 検査 + make_label の purge 確認 |
| `tools/profile_data.py` | `docs/data_schema.md` 生成（target 定義の検算つき） |
| `tools/make_zip.py` | 提出 zip 作成（create_zip.ipynb と同一ルール）+ zip で採点 |
| `tools/progress.py` | Progress（JSONL + STATE.md）。長時間処理の共通規約 |
| `strategies/v0_multifactor/` | v1 の全ファイル + `alpha.py`（walkforward API のラッパー）+ `walkforward_config.json` |
| `requirements.txt` / `evaluate_script.py` / `input/` / `input_manifest.json` | 配布物（`evaluate_script.py` / `input/` / `input_manifest.json` は触らない。`requirements.txt` は lightgbm のみ実採点環境に合わせ 4.1.0 へ修正済み） |

## 3. 外部サービスの仕様（確定事項・再調査しないこと）

- **v0.app**: ①長文の直接入力ができない → プロンプトは GitHub 公開リポジトリに置き初回メッセージで URL を渡す。
  ②取得できるのは**公開 URL のみ**。③出力 Web アプリはサンドボックス内でしか見られず共有 URL は出ない
  （Blob/ZIP DL でローカル保存）。
- **GitHub 公開は仕様上避けられない**: ただし学習済みモデル（`.txt`、約11MB）や個人情報は push しない。
  `.gitignore` で除外。
- **データの扱い**: `input/` は巨大（約700MB）かつ配布物なので push しない。v0 にはスキーマ資料だけ渡す。
- **採点環境の lightgbm は 4.1.0**（配布 `requirements.txt` の 4.6.0 ではない。仮提出の作業で判明し、
  配布 README の GPU 欄にも 4.1.0 と記載）。ローカルの venv・`requirements.txt` も 4.1.0 に統一済み。
  モデルファイル形式 v4 は 4.1.0 / 4.6.0 で互換なので、既存の学習済みモデルはそのまま使える。

## 4. 絶対ルール（例外なし）

1. 予測コードは `target_1day_*` / `raw_target_1day_*` を読まない（静的 + 実行時ガードで検証）。
2. 特徴量の因果性（負の shift・center rolling・bfill・別日付の順位化の禁止）。
3. **モデル選択は Train 内 purged walk-forward が主、Valid は確認のみ**。
   Valid だけ良くなった変更は採用しない。
4. 数値の捏造禁止。実測できない数値は `{要実行}`。
5. v0 の生成物は必ずローカルで検収（py_compile → lookahead → truncation → walk-forward → score）してから次へ進む。

## 5. 作業上の注意

- 全文書・返答は**日本語**（コード・図ラベルは英語可）。
- v0 プロンプトは「1セッション1成果物・質問禁止・不明点はAssumptions節」。修正も新しいセッションで、
  初回メッセージ末尾に【実測結果】と指示を3〜5行追記する方式。
- 環境再現: `setup_env.ps1` を参照（venv 作成 + 固定バージョンの導入）。
- ローカル scoring コマンド:
  `python tools/score.py --submission strategies/v0_multifactor --split valid --json work/reports/<name>.json`
-長時間ジョブはバックグラウンド起動 + `work/logs/` へログ。`python tools/progress.py show` で進捗確認。
  コマンドの制限時間（約30秒）を超える処理は必ず Start-Process/Start-Job で回す。

## 6. 次の作業

1. **ベースライン計測の完了待ち**: `work/reports/wf_baseline_v1.json`（6フォールド Train OOS。
   2015単独では Sharpe +4.86。全体では文書値の約+2.4になる見込み）。
2. **S1 の送信**: v0/計画.md §2 の順で P1 から開始。`{...}` 欄は実測で埋める。
3. P1 の local 検収（truncation + selftest）→ S2 へ。
4. Valid +1.2 超えで中間報告、+1.5 で最終レポート（P6）→ 提出 zip。

