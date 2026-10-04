# CHANGELOG v5 → v6 候補（P8 改善4: 学習型合成と線形成分）

## 前提（v5 = 現行採用版。数値はユーザー実測）

- v5 = v4/S5（K1 + ensemble ridge 0.6 + model_weight 0.25 + model_smoothing_span 20）に `blocks` 明示上書きで
  `size_pure {"logsize": -1}`（重み 2.5）を配合し旧 size ブロックを廃止したもの。**コード変更なし**（config のみ）。
- Train OOS **+2.2857** / Valid **+1.2150**（回転率 0.0109・コスト 0.27%/年・最大DD −3.91%）。
- 本変更のベースコードは GitHub main の v4/S5 コード（`alpha_v2.py` は `blocks` 上書きに対応済み）。

## 変更ファイル

| ファイル | 変更 |
| :--- | :--- |
| `ensemble.py` | `MODEL_KEYS` / `DEFAULT_ENSEMBLE_WEIGHTS` に `linear`（既定 0.0、末尾に追加）、`DEFAULT_LINEAR`、`fit_linear` / `predict_linear` |
| `blend.py`（新規） | 学習型合成の数値部（`settings` / `apply` / `target_vector` / `solve` / `rescale_mix` / `ratio_mix` / `inner_cut` / `before_cut`） |
| `alpha_v2.py` | `DEFAULT_PARAMS` に `linear` / `blend_learning`、`fit_extras` で linear・blend を学習、`linear_columns` / `_inner_predictions` / `fit_blend`、`signal_parts` に linear、`predict_signal` / `estimate_spans` で `blend.apply` |
| `train_v2.py` | (k, seed) ごとの LGBM 学習では linear / blend を無効化、extras で 1 回だけ学習し `meta_v2.json` の `linear` / `blend` に保存 |
| `submission.py` | `blend.apply(config, meta["blend"])`、`meta["linear"]` の係数で線形成分を予測（LGBM と同じ順位行列を再利用）。同梱物に `blend.py` |
| `selftest_p8.py`（新規） | 合成データでビット一致・ON/OFF・purge・因果性を検査 |

## A. 学習型合成 `blend_learning`（既定 `enabled: false`）

手動で決めていた合成重み（`block_weights` / `ensemble_weights` / `model_weight`・`sector_weight`）を、
**各 fit_model の学習期間だけ**で推定し `model["blend"]`（train_v2 では `meta_v2.json` の `"blend"`）に保存する。
予測側は `blend.apply` で params を上書きするだけ（重みは定数なので予測は因果的）。

| キー | 既定 | 意味 |
| :--- | :--- | :--- |
| `enabled` | `false` | OFF なら学習も上書きもしない（v5 とビット一致） |
| `targets` | `["blocks"]` | `blocks` / `ensemble` / `model` の部分集合 |
| `shrink` | `0.5` | 0 = 手動重み、1 = 学習重み（L1 合計を手動に揃えた後に凸結合） |
| `l2` | `0.1` | 項の共分散への ridge（平均対角に対する比） |
| `nonneg` | `true` | 符号反転を禁止（active-set で負の項を落として解き直す） |
| `holdout_frac` | `0.3` | `ensemble` / `model` 用の内側ホールドアウト（ラベルのある日付の後ろ 30%） |
| `min_holdout_dates` | `60` | 内側ホールドアウトの最小日数（足りなければ手動重みのまま、`notes` に記録） |
| `inner_seeds` | `[0]` | 内側学習の LGBM seed |
| `inner_n_estimators` | `null` | 内側 LGBM の木の本数（null = `model_params` と同じ） |
| `max_model_ratio` | `4.0` | 学習した `model_weight` / `sector_weight` の上限（手動値の倍数） |

- 目的関数: 項 T_j と「clip 後・同日 demean・ホライズン平均」ラベル y の相関を最大にする合成
  `beta = (C + l2·mean(diag C)·I)^-1 c`（C = 項の共分散、c = 項と y の共分散）。採点はクロスセクションの
  順位だけで決まるため、同日 demean した y への回帰がそのまま「IC 最大の合成」になる。
- `blocks`: ブロックはパラメータを持たない（順位平均）ので**学習期間全体**が正直な標本。手動重みが非ゼロの
  ブロックだけを対象にし（重み 0 のブロックを足すと欠損が伝播するため）、学習重みを手動の L1 合計に揃える
  → ブロックとモデル項のバランスは v5 のまま、ブロック間の配分だけを学習する。
- `ensemble` / `model`: LGBM の学習期間内予測は過学習で必ず勝つため、**内側ホールドアウト**を使う。
  cut 以降をホールドアウト、窓 t..t+k−1 が cut より前に終わる行（ホライズンごとの purge）だけで
  LGBM（`inner_seeds`）・Ridge・linear を学習し直し、ホールドアウトを予測して重みを推定する。
  rank モデルは内側で学習し直さず、手動重みのまま（既定 OFF のため）。
- `model`: ホールドアウト上で「ブロック合成（`blocks` 学習後の重み）」「z(p)」「z(p − 業種平均)」の 3 項を解き、
  `model_weight = beta_model / beta_blocks`、`sector_weight = beta_sector / beta_blocks` を `shrink` で手動値と混ぜる。
  p は本番と同じ式（`ensemble.combine` → 銘柄ごと EWMA(`model_smoothing_span`)）。
- walk-forward では fold ごとに学習し直す（fold の train_end 以前のラベルだけを使う）。

## B. 線形成分 `ensemble_weights.linear`（既定 `0.0`）+ `linear`

| キー | 既定 | 意味 |
| :--- | :--- | :--- |
| `ensemble_weights.linear` | `0.0` | 0 なら学習も予測もしない（v5 とビット一致） |
| `linear.l2` | `0.01` | 平均二乗誤差スケールの L2（中心化順位の分散は約 1/12 = 0.083） |
| `linear.label_demean` | `true` | 同日 demean したラベルに回帰 |
| `linear.features` | `null` | null = LGBM 入力列から `SECTOR_COLUMNS` を除いた列。指定時は LGBM 入力列の部分集合 |
| `linear.drop_features` | `[]` | 追加で除く列 |
| `linear.chunk_rows` | `200000` | 正規方程式を積み上げる行チャンク（float64 の全体コピーを作らない） |

- モデル: 同日断面順位を中心化した X（rank − 0.5、欠損 0）に対するホライズンごとのプール ridge。
  予測はホライズンごとに `X @ coef` → 同日 z → ホライズン平均（Ridge / LGBM と同じ合成規則）。
- 根拠: Ridge 成分は 4 ブロック z だけの線形結合、LGBM は木（局所的・相互作用あり・単調な滑らかな効果の近似が粗い）。
  全入力列の**大域的・加法的・単調**な効果は両者と関数形が違い、分散が小さく木との相関も低い成分になる。
- 入力は LGBM と同じ順位行列を再利用（推論の追加コストは行列積 1 回）。

## OFF 時のビット一致の担保

- `linear` は `DEFAULT_ENSEMBLE_WEIGHTS` の末尾に追加: `ensemble.combine` は重み dict の順に加算するため、
  既存メンバーの加算順は不変。重み 0 のメンバーは学習も予測もしない（`signal_parts` の順位化条件も不変）。
- `blend.apply` は OFF / 未学習なら config を**そのまま返す**。`fit_extras` は `enabled` のときだけ学習する。
- `selftest_p8.py --reference <v5 の戦略フォルダ>`（旧モジュール一式を別プロセスで読む）で
  `np.array_equal` による完全一致を確認（合成データ・v5 の params）。

## sweep 候補（ローカル walkforward / sweep 用。採否は Train OOS と Valid の両方で v5 超え + 回転率 ≤ 0.017）

```json
{"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0, "linear": 0.3}}
{"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0, "linear": 0.6}, "linear": {"l2": 0.03}}
{"blend_learning": {"enabled": true, "targets": ["blocks"], "shrink": 0.5}}
{"blend_learning": {"enabled": true, "targets": ["ensemble", "model"], "shrink": 0.5}}
{"ensemble_weights": {"linear": 0.3}, "blend_learning": {"enabled": true, "targets": ["blocks", "ensemble", "model"], "shrink": 0.5}}
```

- `linear` の重みと `l2` は学習 1 回で再合成できる（成分 `p_linear` を保存すればよい）。`blend_learning` は
  fold ごとに重みが変わるため、walkforward（fit_model → predict_signal）で評価する。

## 実測結果（ローカル検収、2026-10-05・不採用）

受領パッケージは GitHub の v4/S5 コードベースだった（v0 から見て P8 プロンプト URL が 404）ため、現行の
P7 コードと 3-way マージして反映した。マージ後の検証と実測（採用条件: Train OOS > +2.2857 かつ
Valid > +1.2150 かつ回転率 ≤ 0.017）:

- py_compile / 静的 lookahead / truncation / runtime lookahead OK。
- `selftest_p8.py --reference work/ref_v5`: **RESULT OK**（既定 OFF・明示 OFF でビット一致 max|diff|=0、
  ON/OFF 切替、内側 purge −1 取引日、因果性 max|diff|=0）。
- `selftest_p7.py` は通常実行（現行 v5 config を読む）では v5 の明示 `blocks` とテスト前提（block_set v3/v1）が
  食い違うため NG 表示になるが、config なしの隔離コピーでは **RESULT OK**（OFF ビット一致・ON 切替・因果性・
  train_v2 経路・sweep 再合成一致）。これは selftest_p5 の既知 NG と同種（テストが旧 config 前提）で、
  P8 マージによる回帰ではない。受領 alpha_v2.py にあった日本語 1 行の文字化け（U+FFFD）は復元した。
- walkforward 基準再現: **+2.2857**（`work/reports/improve4_wf_baseline.json`）。

| 候補 | Train OOS | 回転率 | 判定 |
|---|---|---|---|
| `ensemble_weights.linear` = 0.1 / 0.3 / 0.5 / 1.0 | +2.2837 / +2.2486 / +2.2474 / +2.2402 | 0.0128〜0.0130 | 不採用（OOS 未達。重み増で単調低下） |
| `linear` = 0.6 + `linear.l2` = 0.03 | +2.2569 | 0.0129 | 不採用 |
| `blend_learning` targets=["blocks"] shrink 0.5 | **+2.3057** | 0.0123 | OOS は基準超えも **Valid +1.1742**（グロス +4.22% / 回転率 0.0106 / maxDD −3.90% / 勝率 54.1%）で未達 → 不採用 |
| `blend_learning` targets=["ensemble","model"] shrink 0.5 | +2.1807 | 0.0126 | 不採用 |
| `linear` 0.3 + blend all | 未実行（成分単独がいずれも未達のため） | — | 不採用 |

- 実測 JSON: `work/reports/improve4_lin_*.json` / `improve4_blend_*.json` / `improve4_blend_blocks_valid.json`。
- v5 復元後: train_v2 → score valid で **+1.2150 / 回転率 0.0109 / グロス +4.37%** を再現
  （`work/reports/improve4_restore_v5_valid.json`）。
- 結論: **どちらの新機能も v5 を上回らず不採用。提出候補は v5 のまま**。コードは既定 OFF・v5 ビット一致の
  まま保持する。linear / blend は P7 の `sample_decay_halflife` に未対応（受領版が S5 ベースのため。必要なら
  `fit_linear` / 内側再学習へ sample_weight を通す改修を追加すること）。
