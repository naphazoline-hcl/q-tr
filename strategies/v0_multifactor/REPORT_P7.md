# REPORT_P7: 改善ラウンド3（受け入れチェック・検証・時間見積もり・Assumptions）

下の数字はすべて**合成データでの動作確認**と、この作業環境（**2 CPU**）での時間計測だけ。実データの
Sharpe などは一切測っていない（採否はローカルの walkforward / sweep で判断する）。
検証環境: Python 3.13.11 / lightgbm 4.1.0 / numpy 2.5.3 / pandas 3.0.6 / scikit-learn 1.9.1
（リポジトリの固定は Python 3.11 / numpy 2.4.6 / pandas 3.0.3 / scikit-learn 1.9.0。ローカルでの再実行を推奨）。

## 1. 受け入れチェックリスト

- [x] `block_set: "v3"`（momentum / revision / liquidity）を追加し、v1 / v2 は不変（v3 は v1 の dict をコピーして追加）
- [x] `sample_decay_halflife` / `feature_top_k` を `params` に追加（既定 None = OFF）
- [x] 既定 OFF のとき現行 v4/S5 とシグナルがビット一致（合成データ・4 構成で `max|diff| = 0`、§2）
- [x] `selftest_p7.py` で新機能の ON/OFF 切替と因果性を確認（`RESULT: OK`）
- [x] `feature_top_k` ON の学習時間見積もり ≤ 10 分/フォールド（約 5.1 分）、推論 ≤ 30 分（約 2.2 分 + 特徴量構築）（§3）
- [x] `CHANGELOG_v4v5.md` に変更点・根拠・既定値を記録
- [x] 禁止パターンなし（`tools/check_lookahead.py`: `RESULT: OK (no ERROR patterns)`、16 ファイル）、Valid ラベルを読まない
  （追加コードは target ファイルを読まない。重み・列選択は学習行だけから計算）

## 2. 検証結果（実行ログからの転記）

`selftest_p7.py --reference <v4/S5 の alpha_v2.py>`（合成: train 22,400 行 / predict 9,600 行）:

| 検査 | 結果 |
|---|---|
| [1] OFF: S5 / K1 / S5+rank 0.2 / S5+regime_mix の新旧比較（旧版は旧 ensemble.py 等と一緒に読込） | 4 構成とも `max|diff| = 0.000e+00`（`np.array_equal` で一致） |
| [2] `block_set_v3`（Ridge 経由のみ） | S5 との差 0.5879、Ridge 7 列 |
| [2] `v3_direct`（block_weights に新ブロック） | S5 との差 0.8630 |
| [2] `decay_250` | S5 との差 0.0761、weight_min 0.461 / mean 0.696 |
| [2] `top_k_20` | S5 との差 0.1088、列 24（20 + カテゴリ 4） |
| [2] `all_on`（v3 + decay + top_k + rank 0.2） | S5 との差 1.0604、LGBM・rank とも選択列 |
| [3] 因果性（all_on、予測期間の後半を切り落とし） | `max|diff| = 0.00e+00` |
| [4] train_v2 の経路（select_features → 選択列で学習）vs fit_model の 2 パス | 列一致、`max|diff| = 0.00e+00` |
| [4] `sweep_improve2.compose` vs `predict_signal` + EWMA（v3 / v3_direct / all_on） | 3 構成とも `max|diff| = 0.00e+00` |

`selftest_p7_pipeline.py`（全部 ON、一時フォルダで train_v2 → meta_v2.json → submission）: 7 項目すべて OK。
meta の `features` = `feature_selection.columns` = fit_model の選択列（24 列）、rank 4 本も同じ列、Ridge は v3 の 7 ブロック、
submission の出力 = fit_model + predict_signal + 最終 EWMA（`max|diff| = 0.00e+00`、欠損 0）。

## 3. 時間見積もり（`bench_p7.py`、合成パネル 1500 日 x 498 銘柄、"v1" 51 列、この環境 2 CPU）

| 処理（1 モデル） | 実測（秒） |
|---|---|
| LGBM fit 全 51 列（1 パス目） | 30.1 |
| 同 + 減衰重み | 32.6 |
| 列選択（gain 平均順位） | 0.0 |
| LGBM fit 選択 24 列（2 パス目） | 20.6 |
| 同 + 減衰重み | 21.5 |
| Ridge fit（v3 の 7 ブロック + 減衰重み） | 0.2 |
| LGBM predict 選択列（Valid 規模の行） | 21.8 |

- **学習 1 フォールド（6 本 = 2 ホライズン x 3 seed）**: `feature_top_k` ON = `6 x 30.1 + 6 x 20.6 + 0.2 ≈ 304 秒（約 5.1 分）`、
  減衰重みも ON で `≈ 325 秒（約 5.4 分）`。現行（1 パス）は `≈ 181 秒（約 3.0 分）`。rank モデルも ON にする場合は
  REPORT_P5 の実測（lambdarank 2 本 68.4 秒、51 列）x 2 ホライズンを足しても `≈ 462 秒（約 7.7 分）` で **10 分以内**。
  これは Train 全期間（train_v2）規模の上限で、walkforward の各フォールドは学習行がこれより少ない。
  特徴量構築・順位化は従来どおり別計上。n_estimators の調整や選択の簡略化は不要と判断した。
- **推論**: 推論式は不変で、選択列のモデルは列が少ないぶん軽い。`6 x 21.8 ≈ 131 秒（約 2.2 分）` + 特徴量構築
  （規約上 1〜3 分）で **30 分に十分収まる**。v3 のブロック合成は列の断面順位 13 本分が増えるだけ。

## 4. Assumptions（不明点は質問せずに以下で決めた）

1. **減衰の単位は営業日**: 経過日数 = 学習行の最終 Date からの営業日数（学習行の Date で作ったカレンダー上の位置の差）。
   このリポジトリの日数パラメータ（`purge_days` / `turnover_window` / k126・k250）がすべて営業日のため。250 ≈ 1 年。
2. **基準日** = fit_model に渡された学習行の最終 Date。walkforward・train_v2 とも「ラベルのある行」だけを渡すので、
   実質「最後にラベルがある日」。予測期間の日付は使わない。
3. **重みは正規化しない**（仕様の式どおり `0.5 ** (経過 / halflife)`）。LightGBM の `reg_lambda=1`・Ridge の `alpha=10` は
   重みの合計（数十万行規模）に比べて小さく、正規化の有無の影響は小さいと判断した。
4. 減衰重みは LGBM・Ridge・rank モデル（ON のとき）に同じものを渡す。同一 Date で一定なので Ridge の同日デミーンは不変。
5. **列選択の単位は学習期間全体で 1 セット**。gain は非カテゴリ列の合計で正規化 → モデル内順位（同値は平均順位）→
   horizon x seed で平均。同順位は平均 gain シェア → 入力列順で決める。出力は元の列順（meta / rank_for_model と一致）。
6. `SECTOR_COLUMNS`（カテゴリ 4 列）は常に残し、k に数えない（`feature_top_k: 20` → 24 列）。
7. 1 パス目は最終モデルと同じ seed・n_estimators・重みで学習する（時間に余裕があるため簡略化しない）。
   選択列が全列と同じなら 2 パス目を省略。LGBM の重みが 0、または 1 パス目のモデルが無いときは選択しない（全列）。
8. train_v2 は列選択を 1 回だけ行い `models_v2/feature_selection.json` に保存。`--resume` は設定ハッシュと学習行数が
   一致するときだけ再利用する。meta_v2.json に `feature_selection` / `sample_decay`（OFF なら null）を追加。
9. v3 の新ブロックには業種内順位を使わない（`sector_rank_blocks` の既定 size / value を変えないため）。
   符号は仕様どおりで、`alpha_features.py` の式と整合することを確認した（CHANGELOG §1）。
10. `walkforward_config.json` は v4/S5 のまま変更しない。`tools/walkforward.py` / `tools/sweep_improve2.py` も変更不要
    （fit_model が内部で処理。sweep のキャッシュ `work/sweep_improve2/` は学習設定を変えたら退避・削除が必要）。

## 5. 次にやること（ローカル）

1. 再検査: `python strategies/v0_multifactor/selftest_p7.py --reference <v4/S5 の alpha_v2.py>`（旧版は ensemble.py 等と同じ
   フォルダに置く。例: `git worktree add ../q-tr-s5 <S5 のコミット>`）と `python strategies/v0_multifactor/selftest_p7_pipeline.py`。
2. walkforward（Train OOS）: CHANGELOG §6 の params を 1 つずつ試す（v3 Ridge 経由 → v3 直接合成 → decay 250 / 500 / 750 →
   top_k 20 / 30 → 組み合わせ）。
3. sweep（Valid）: 学習設定ごとに `work/sweep_improve2/` を分けて成分を作り、`block_weights` の直接合成は再合成で比較。
4. 採用条件は従来どおり Train OOS ≥ 現行（+2.124）かつ Valid 改善かつ回転率 ≤ 0.017。採用したら train_v2 → submission。

## 6. 未完了・制約

- 実データでの walkforward / sweep / train_v2 は実行していない（v0 は検証しない方針・データなし）。性能の数値は `{要実行}`。
- 時間は合成データでの計測。実データでは木の形が違うため、ローカルの実測で確認すること。
