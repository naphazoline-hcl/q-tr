# CHANGELOG v4 → v5 候補（P7 改善ラウンド3: 新ブロックとロバスト化）

基準は v4/S5（K1 + Ridge アンサンブル `ridge: 0.6` + `model_weight: 0.25` + `model_smoothing_span: 20`、
Train OOS +2.1240 / Valid +0.8030 の実測値はユーザー提供）。今回の追加は **すべて既定 OFF** で、
既定値のままなら v4/S5 と **シグナルがビット一致**する（`selftest_p7.py --reference` で確認、§5）。
採否はローカルの walkforward / sweep で判断する（ここでは性能の数値を出していない）。

## 0. 追加したキーと既定値

| キー | 既定 | 値 | 効く場所 |
|---|---|---|---|
| `block_set` | `"v1"`（walkforward_config.json。コード既定 `"v2"` も不変） | `"v1"` / `"v2"` / **`"v3"`**（新規） | ブロック合成・Ridge の説明変数 |
| `sample_decay_halflife` | `None`（OFF） | `None` / 正の営業日数（例 250） | LGBM・Ridge・rank の学習重み |
| `feature_top_k` | `None`（OFF） | `None` / 正の整数（例 20） | LGBM（と rank）の入力列 |

`BLOCK_SETS["v1"]` / `["v2"]`、`block_weights` の既定、`sector_rank_blocks` の既定、既存キー
（`ensemble_weights` / `regime_weights` / `turnover_cap` / `slow_profile` など）・API 契約は変更していない。

## 1. A: `block_set: "v3"`（新ブロック 3 つ）

v1 の 4 ブロック（size / value / quality / lowrisk）を**そのままコピー**し、次の 3 ブロックを足した。
各ブロックは従来どおり「符号付き断面順位の等ウェイト平均」（`block_scores`）。業種内順位は使わない
（`sector_rank_blocks` は既定の size / value のまま）。

| ブロック | 列: 符号 | 符号の根拠（alpha_features.py の式） |
|---|---|---|
| momentum | `rcc120` +1 | `_rolling(rcc, 120, 84, "sum")`。rcc = 調整後終値の日次リターン → 過去 120 日の累積リターン。高い = 勝ち組 → 中期モメンタム |
| | `rcc250` +1 | 同 250 日（約 12 か月）の累積リターン → 12 か月モメンタム |
| | `mom` +1 | `cum[250] - cum[20]` = 直近 1 か月を除く 12 か月リターン（12-1 モメンタム。短期リバーサルを除外） |
| | `res250` +1 | `sum(raw - beta * topix)` の 250 日 = 市場ベータを除いた残差モメンタム |
| revision | `eps_revision` +1 | `_revision`: 同じ対象年度の予想 EPS の前回開示からの改訂率 `(new - prev) / |prev|`。上方修正 = 正 → 修正後ドリフト |
| | `opprofit_revision` +1 | 同じ式で予想営業利益の改訂率 |
| | `eps_fwd_chg` +1 | `eps_fwd - shift(125)`（分割調整済み予想 EPS の約半年変化、円/株）。上昇 = 正 |
| | `sales_yoy_acc` +1 | `sales_yoy - shift(250)` = 売上成長率の加速。加速 = 正 |
| | `op_margin_yoy_chg` +1 | 営業利益率（累計ベース）の前年同期差 `opm - opm_prev`。改善 = 正 |
| liquidity | `illiq20` +1 | `mean(|raw| / TurnoverValue)` 20 日（Amihud）。大きい = 非流動 → 非流動性プレミアム |
| | `illiq60` +1 | 同 60 日（v1 の size ブロックと同じ列・同じ符号） |
| | `logturn20` -1 | `mean(log TurnoverValue)` 20 日。大きい = 流動的 → 符号は負（低売買代金ほどプレミアム） |
| | `logturn60` -1 | 同 60 日（v1 の size ブロックと同じ列・同じ符号） |

使い方:
- **Ridge 経由（狙い）**: `block_weights` に新ブロックを入れない（既定の重み 0 = 直接合成には効かない）。
  `ensemble.fit_extras` が `block_scores` の**全ブロック z** を Ridge の説明変数にするので、`block_set: "v3"` に
  するだけで新ブロックは Ridge の係数として自動的に使われる（Ridge は 7 列になる）。
- **直接合成**: `block_weights` に `momentum` / `revision` / `liquidity` の重みを足せば、従来のブロック合成にも入る
  （`block_weights` は dict の部分上書きなので、既存 4 ブロックの重みはそのまま残る）。
- 注意: liquidity は size と `illiq60` / `logturn60` を共有し相関が高い。Ridge（alpha=10）では係数が分け合われうる。
  `eps_fwd_chg` は円/株の差分なので株価水準の影響が残る（断面順位にしているので外れ値の影響は小さい）。

## 2. B1: `sample_decay_halflife`（学習サンプルの時間減衰）

- 重み `w = 0.5 ** (経過日数 / halflife)`（正規化しない。式どおり）。
- **経過日数 = 学習行の最終 Date からの営業日数**。学習に渡された行の Date だけでカレンダーを作り
  （`pd.factorize(Date, sort=True)`）、`(日付数 - 1) - 位置` を経過日数とする。予測期間の日付は見ない。
- 同一 Date の行は同じ重み → 断面内の相対関係は変えず、古い日付ほど軽くするだけ。Ridge の `label_demean`
  （同日平均を引く）も重みの有無で変わらない。
- 渡し先: LGBM（`_fit_boosters` の `sample_weight`）、Ridge（`ensemble.fit_ridge(..., sample_weight=)` の重み付き
  最小二乗）、rank モデル（`ensemble.fit_rank(..., sample_weight=)`）。LightGBM 4.1.0 + numpy 2 の copy 例外を
  避けるため、重みは連続 float32 にしてから渡す。
- walkforward（fit_model に全ホライズン）と train_v2（ジョブごとに fit_model）のどちらでも、基準日は
  「その学習に渡した特徴量行の最終 Date」で同じ。どちらも「どれかのラベルがある行」だけを渡すので、基準日は
  実質「最後にラベルがある日」（strict ラベルなら学習末尾 − 125 営業日前後）になる。`model["sample_decay"]` / meta の `"sample_decay"` に要約
  （最小・平均重み、実効行数の割合）を残す。
- 単位を営業日にした理由: `purge_days` / `turnover_window` / ホライズン k126・k250 など、このリポジトリの
  日数パラメータはすべて営業日。250 ≈ 1 年。

## 3. B2: `feature_top_k`（gain importance による列選択・2 パス学習）

1. 1 パス目: 従来どおり全列で (horizon, seed) の LGBM を学習（`sample_decay_halflife` も同じ重みで効く）。
2. 各モデルの gain importance を**非カテゴリ列の合計で正規化** → モデル内で降順の順位（同値は平均順位）→
   horizon × seed で**平均順位**を計算 → 平均順位の小さい順に上位 k 列（同順位は平均 gain シェアの大きい順 →
   入力列順）。選択は学習期間全体で 1 セット。
3. `SECTOR_COLUMNS`（sector17 / sector33 / market / margin。カテゴリ列）は順位の競争に入れず**常に残す**
   （k に数えない）。列の並びは元の `model_columns` の順のまま（rank_for_model・meta の列順と一致）。
4. 2 パス目: 選択列だけで最終モデルを学習し、`result["columns"]` = 選択列。rank モデルも同じ選択列で学習
   （`fit_extras` が `model["columns"]` を使う）。選択列が全列と同じなら 2 パス目は省略。
5. LGBM の重みが 0、または全ホライズンが `min_label_days` 未満で 1 パス目のモデルが無いときは選ばない（全列）。

一貫性: `rank_for_model` / `model_prediction` / `submission.ranked_matrix` は従来どおり `model["columns"]` /
`meta["features"]` をそのまま使う。
- `train_v2.py`: ジョブの前に `alpha.select_features`（= fit_model の 1 パス目と同じ学習）で 1 回だけ選び、
  `models_v2/feature_selection.json` に保存（`--resume` は設定ハッシュと学習行数が一致すれば再利用）。各ジョブは
  `model_features=選択列, feature_top_k=None` で学習し、`meta_v2.json` の `"features"` = 選択列、
  `"feature_selection"` に選択結果（平均順位・平均 gain シェア）を残す。
- `submission.py`: 推論式は不変。`feature_selection.columns` と `features` の一致を `load_meta` で確認する
  ガードだけ追加（OFF / 旧 meta は null なので素通り）。

## 4. 変更ファイル

| ファイル | 変更 |
|---|---|
| `alpha_v2.py` | `BLOCK_SETS["v3"]`、`DEFAULT_PARAMS` に 2 キー、`fit_model` を `_fit_boosters` に整理（OFF の呼び出しは同一）、`sample_weights` / `decay_summary` / `select_columns` / `select_features` を追加、`fit_extras` が重みを渡す |
| `ensemble.py` | `fit_ridge` / `fit_rank` に `sample_weight=None` 引数（None なら従来と同一） |
| `train_v2.py` | feature_top_k の 1 パス目（`load_or_select`）→ 選択列でジョブ学習、meta に `feature_selection` / `sample_decay` |
| `submission.py` | `load_meta` に選択列の一致ガード（推論式は不変） |
| `selftest_p7.py` / `selftest_p7_pipeline.py` / `bench_p7.py` | 新規（合成データの検査、train_v2 → meta → submission の一貫性、時間計測） |

walkforward_config.json は**変更していない**（v4/S5 のまま）。`tools/walkforward.py` / `tools/sweep_improve2.py` も
変更不要（fit_model が 2 パス・重みを内部で処理し、成分 `blk_<block>` は v3 の 7 ブロックを自動で含む）。

## 5. ビット一致の確認（合成データ）

`selftest_p7.py --reference <v4/S5 の alpha_v2.py>`: 旧版を旧 ensemble.py 等と一緒に読み込み、OFF の 4 構成
（S5 / K1 / S5+rank 0.2 / S5+regime_mix）で新旧のシグナルが `np.array_equal` で一致することを確認した
（結果は REPORT_P7.md）。

## 6. ローカル sweep の例（walkforward_config.json の params に足す）

```json
{"block_set": "v3"}
{"block_set": "v3", "block_weights": {"momentum": 0.2, "revision": 0.2}}
{"sample_decay_halflife": 500}
{"feature_top_k": 25}
{"block_set": "v3", "sample_decay_halflife": 750, "feature_top_k": 30}
```

学習側を変える候補（v3 / decay / top_k）は学習のやり直しが要る。`sweep_improve2.py` は成分を
`work/sweep_improve2/` にキャッシュするので、学習設定を変えたら**そのフォルダを退避・削除してから**実行する。
`block_weights` の直接合成は v3 で学習した成分から再合成できる（`compose` は全 `blk_*` を重み `get(b, 0)` で足す）。

## 7. ローカル実測（2026-10-04、S7 受領後の検収）

- 基準再現: 現行 v4/S5 のまま walkforward → **+2.1240**（`work/reports/improve3_wf_baseline.json`、採用値と一致）。
  P7 反映後の Valid（config 変更なし）も **+0.8030** / 回転率 0.0133 で維持（`work/reports/improve3_valid.json`）。
- P7 候補の walkforward 実測（`work/run_improve3.py` で一括実行、config は一時ファイルで上書きなし）。
  いずれも基準 **+2.1240** 未達のため不採用。提出候補は v4/S5 のまま:

  | 候補 | 追加 params | Train OOS | レポート |
  |---|---|---|---|
  | v3 | block_set v3 | +2.0222 | improve3_wf_v3.json |
  | v3_direct | v3 + block_weights momentum/revision 0.2 | +2.0867 | improve3_wf_v3_direct.json |
  | decay250 | sample_decay_halflife 250 | +2.0889 | improve3_wf_decay250.json |
  | decay500 | sample_decay_halflife 500 | +2.0884 | improve3_wf_decay500.json |
  | topk25 | feature_top_k 25 | +2.0929 | improve3_wf_topk25.json |
  | v3_decay500 | v3 + decay 500 | +2.0551 | improve3_wf_v3_decay500.json |
  | v3_topk25 | v3 + top_k 25 | +2.0487 | improve3_wf_v3_topk25.json |
  | all_on | v3 + decay 750 + top_k 30 | +2.0463 | improve3_wf_all_on.json |

- 検証: selftest_p7（OFF 4構成ビット一致・ON切替・因果性・再合成一致）RESULT OK、
  selftest_p7_pipeline（全部 ON の train_v2 → meta → submission 一貫性）RESULT OK、
  truncation OK、lookahead 静的・runtime OK。
