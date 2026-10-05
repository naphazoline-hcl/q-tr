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
- **現状**: `strategies/v0_multifactor` に **v7 採用版**（v6 + モデル項の再構成: `ensemble_weights.ridge 0` /
  `model_weight 0.10` / `model_smoothing_span 10`。コード変更なし・config のみ）。実測
  **Train OOS +2.4667 / Valid +1.3989 / Valid 回転率 0.0135（コスト 0.34%/年）**。
  v6 は予測側再配合（`sector_neutral false` / quality 0.9 / size_pure 1.8 / value 0.3 / lowrisk 0。
  Train +2.4560 / Valid +1.3783）。**第一目標 Valid +1.2 は達成**、第二目標 +1.5 は未達。
  P7（新ブロック・ロバスト化）は受領・検収済みだが全候補不採用。P8（学習型合成 `blend_learning`・線形成分
  `ensemble_weights.linear`）も受領・検収済み（既定 OFF・v5 ビット一致のまま保持、全候補不採用）。
  提出候補は v7（旧候補は v6 / v5 / v4/S5 / 旧暫定提出は v1）。詳細は `CHANGELOG_v7_lgbm_only.md` /
  `CHANGELOG_v6_quality_tilt.md` / `CHANGELOG_v5v6.md` と `docs/baseline.md`。

## 2. ファイルマップ

| ファイル | 内容・状態 |
| :--- | :--- |
| `v0/計画.md` | v0.app 運用計画（制約対策・セッション順序・フィードバック・中断対策・実行ログ） |
| `v0/prompts/P1〜P8*.md` | v0.app に投げるプロンプト（日本語・自己完結・1セッション1成果物・§0.5 に作業規約を内蔵） |
| `v0/prompts/P0_作業規約.md` | 全プロンプト共通の作業規約（タイムアウト対策・成果物の渡し方）。P1〜P8 §0.5 から参照 |
| `v0/messages/S1〜S8_初回メッセージ.md` | 各セッションの初回メッセージ定型文（`{...}` は実測値で埋める） |
| `v0/messages/再開用_ターン中断時.md` / `引き継ぎ_S7受領後.md` / `引き継ぎ_S8受領後.md` | 中断時の再開メッセージ、受領後の検収手順（新セッション用） |
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
| `strategies/v0_multifactor/` | v1 の全ファイル + `alpha.py`（walkforward API のラッパー）+ `walkforward_config.json`。v2 以降: `alpha_features.py` / `alpha_v2.py` / `submission.py` / `train_v2.py`。P5 追加: `ensemble.py` / `slowdown.py` / `regime_v4.py` / `selftest_p5.py` / `bench_p5.py`。P7 追加: `selftest_p7.py` / `selftest_p7_pipeline.py` / `bench_p7.py` / `CHANGELOG_v4v5.md` / `REPORT_P7.md`。v5: `CHANGELOG_v5_size_pure.md`（config のみの変更）。P8 追加: `blend.py` / `selftest_p8.py` / `CHANGELOG_v5v6.md` / `REPORT_P8.md`（既定 OFF。v0 受領版は S5 ベースのため現行コードへ 3-way マージ済み）。v6: `CHANGELOG_v6_quality_tilt.md`（config のみの変更）。v7: `CHANGELOG_v7_lgbm_only.md`（config のみの変更） |
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
- **v0 の実行特性（タイムアウト対策の根拠）**: ツールを呼ばずに応答を終えるとターン終了扱い。
  1応答が長すぎるとターンが強制終了される（目安4分/出力量、書き込み済みファイルは残る）。
  待機は `sleep 30`、完了判定は `echo __DONE__` の出力で行う。正本は `v0/prompts/P0_作業規約.md` で、
  P1〜P6 §0.5 に内蔵済み。他プロジェクト用には `~/.config/opencode/AGENTS.md` に
  v0 プロンプト作成ルールとして同じ規約を記載している。

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
- 全プロンプトに作業規約（§0.5）と新出力形式（`deliverables/` に実ファイル + 回収アプリ）を内蔵済み。
  初回メッセージへの規約貼り付けは不要。中断時は `v0/messages/再開用_ターン中断時.md` を送る。
- 環境再現: `setup_env.ps1` を参照（venv 作成 + 固定バージョンの導入）。
- ローカル scoring コマンド:
  `python tools/score.py --submission strategies/v0_multifactor --split valid --json work/reports/<name>.json`
-長時間ジョブはバックグラウンド起動 + `work/logs/` へログ。進捗確認は `python tools/progress.py <path>`
  （`show` サブコマンドは無い）。コマンドの制限時間（約30秒）を超える処理は必ず Start-Process/Start-Job で回す。

## 6. 次の作業

1. **S5 反映・検収済み**（2026-10-04）。受領物は `work/s5_package/`（gitignore）。実測は
   `work/reports/sweep_improve2.md` / `improve2.json` / `improve2_final.json` / `improve2_final_valid.json`。
   採用は mw025_ridge06_span20（Train +2.1240 / Valid +0.8030 / 回転率 0.0133）。コミット ffccd38（push 済み）。
2. **P5 後のローカル探索も完了**（コミット 7c4ef36、push 未）: span 微調整・rank 併用・slow_profile を実測し、
   いずれも v4/S5 を上回らず不採用。詳細は `docs/baseline.md` の「P5 後のローカル追加探索」。
3. **P7（改善3: 新ブロックとロバスト化）は受領・検収済み（2026-10-04、不採用）**:
   `v0/prompts/P7_改善3_新ブロックとロバスト化.md` / `v0/messages/S7_初回メッセージ.md` で送信し、
   `work/s7_package/`（gitignore）に受領。`strategies/v0_multifactor/` に反映
   （alpha_v2 / ensemble / train_v2 / submission + selftest_p7 / pipeline / bench_p7 /
   CHANGELOG_v4v5 / REPORT_P7）。OFF 既定のビット一致・pipeline 一貫性・truncation・lookahead（静的/runtime）
   はすべて OK。walkforward 基準 +2.1240 を再現後、8候補を実測するもすべて基準未達で不採用
   （最大 topk25 +2.0929、詳細は `docs/baseline.md` と CHANGELOG §7）。**提出候補は v4/S5 のまま**。
   実測 JSON は `work/reports/improve3_wf_*.json` / `improve3_valid.json`。
4. **size 項の再配合をローカル採用（2026-10-04、v5）**: sample02（pure size: Train +1.94 / Valid +1.18）の
   知見から size 項を再検証。既存 size ブロックの weight 増は OOS が低下する一方、素の `z(rank(-logsize))`
   項は OOS/Valid を同時に改善した。`blocks` を明示上書き（`size_pure: {"logsize": -1}` + v1 の
   value/quality/lowrisk）し weight 2.5 で walkforward が **Train +2.2857 / Valid +1.2150 / 回転率 0.0109**。
   w1.5〜3.0 の OOS カーブ最大が w2.5（コード変更なし・config のみ。`CHANGELOG_v5_size_pure.md`、
   実測は `work/reports/size_grid*.json` / `size_pure_w*.json`）。**第一目標 Valid +1.2 達成**。
   ロールバックは `work/improve3_cfgs/walkforward_config_v4s5_backup.json`。
5. **提出準備完了（2026-10-04）**: 現行 v5 で提出 zip を作成済み（`strategies/v0_multifactor.zip`、22.05MB、
   gitignore）。zip で採点して **+1.21496**（設計実測 +1.2150 と一致）を確認。S6 初回メッセージも v5 の
   実測値に更新済み（コミット `b73d4bd`）。提出はユーザーが zip をアップロードする。
   zip を作り直す場合: `python tools/make_zip.py --name v0_multifactor --score`。
6. **P8（改善4: 学習型合成と線形成分）は受領・検収済み（2026-10-05、不採用）**: 受領 ZIP は S5 コードベース
   だったため P7 コードへ 3-way マージして反映（`blend.py` / `selftest_p8.py` 追加。既定 OFF で
   `selftest_p8.py --reference work/ref_v5` が max|diff|=0、walkforward 基準 +2.2857 を再現）。実測は
   linear 5候補が OOS +2.2402〜+2.2837、blend blocks が OOS **+2.3057** / Valid **+1.1742**、
   blend ensemble+model が OOS +2.1807 → すべて不採用。v5 復元後に Valid +1.2150 を再現。
   実測 JSON は `work/reports/improve4_*.json`。**提出候補は v5 のまま**（zip 作成済み・再作成不要）。
7. **v6 採用（2026-10-05、ローカル: 予測側の再配合）**: P9 のテーマ選定前に v5 の成分別診断
   （`work/diag_v5.py`、成分は `work/sweep_improve2/` を v5 config で再生成）を行い、予測側パラメータを
   `work/search_v5_diag*.py` で 214 候補探索 → 最良 OOS を直接パイプラインで確認。
   `sector_neutral false` / quality 0.9 / size_pure 1.8 / value 0.3 / lowrisk 0 / model_weight 0.15 / ridge 0.4 で
   **Train OOS +2.4560 / Valid +1.3783 / 回転率 0.0130**（2020 −0.84 → +0.35）。コード変更なし・config のみ。
   詳細は `CHANGELOG_v6_quality_tilt.md`、実測は `work/reports/direct_v6a_q09_lr0_*.json`。**提出 zip は要再作成**
   （ユーザー指示時: `python tools/make_zip.py --name v0_multifactor --score`）。
8. **v7 採用（2026-10-05、ローカル: モデル項の再構成）**: v6 後の探索（短期ラベル・全特徴量・ブロック定義
   拡充・LGBM 容量増）はすべて不採用。モデル項を LGBM / Ridge に分解した 144 候補の再合成評価で
   「ridge 0 / model_weight 0.10 / span 10」が最良となり、直接パイプラインで
   **Train OOS +2.4667 / Valid +1.3989 / 回転率 0.0135** を確認（改善幅は小さい。OOS +0.011 / Valid +0.021）。
   詳細は `CHANGELOG_v7_lgbm_only.md`、実測は `work/reports/v7b_lgbmonly_*.json`。**提出 zip は要再作成**。
9. push / 提出 zip 作成はユーザーの明示指示があった場合のみ実行する。

