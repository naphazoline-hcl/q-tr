# MODEL_DESIGN — alpha_v2（v0_multifactor v2）の設計判断

## 0. 出発点（実測済みの事実だけ）

| 項目 | 値 | 出典 |
|---|---|---|
| v1 再現値（Train OOS, purged WF） | **+1.993**（回転率 0.032） | docs/baseline.md |
| 94列を v1 の alpha.py にそのまま投入 | **+1.871**（回転率 0.0358） | S1 実測 |
| 特徴量 | 51 → 94列、609,960行×94列で truncation 検査一致 | S1 実測 |
| NaN 率 top3 | progress_1q 75.6% / opprofit_revision 32.5% / eps_revision 28.7% | S1 実測 |
| inf | eps_fwd_yoy に 4,425行（式は v1 由来） | S1 実測 |
| v1 の Valid 弱点 | 2020 −0.32 / 2021 −0.51、2018-2021 +0.38 | docs/baseline.md |

S1 では列を足すとグロスが下がり、回転率が上がった（0.032 → 0.0358）。そこで v2 は列を足すのではなく、
**列の取捨・業種中立化・重み設計**で勝負する。v2 の性能は未計測（`{要実行}`）。

## 1. ラベル設計

- **k 日平均残差ラベル**: `label[t] = mean(target[t..t+k-1])`（k ∈ {126, 250}）。シグナルが数ヶ月
  スケールで動くようになるので回転率が下がる。v1 と同じ考え方。
- **窓が end を超える行は NaN（v1 から変更）**。v1 は `min_periods=1` で、end の手前では短い窓
  （1日ラベルに近い値）になっていた。v2 は取引日カレンダー上の窓の最終日が end より後なら NaN にする。
  - 帰結（暦からの概算）: Train の開始は 2008-11-04 なので、fold 2010 は train_end が開始から約20営業日後になる。
    k=126/250 とも有効ラベルが 0 行なので、モデルなし（ブロック合成のみ）で予測する。fold 2011 は k126 だけ
    学習する（約140日分）。k250 は約20日分しかなく、`min_label_days=100` 未満なのでスキップする。
    実際の日数は `fit_model` の戻り値 `label_days` / `skipped` に残る。
  - 上場廃止で窓内の観測が `label_min_frac`（0.5）未満の行も NaN（残りの日数の平均を使わない）。
- **clip ±0.05**（`params.label_clip`、未指定なら config 最上位の値）。k≥126 の平均でこの幅に
  届くのは稀なので、実質的には k=63 などの短いホライズン向けの保険になる。
- **ボラ割りラベル（`label_transform="vol"`）は実装済み・既定 OFF**。分母は vol60 を同じ日の中央値で
  割った相対ボラなので、日付間の水準差は持ち込まない。不採用の理由: 低リスクブロック（beta, vol60）が
  すでに大ボラ銘柄を減点している。採点は分位内が等ウェイトで、生リターンを評価するので、
  リスク調整後リターンを学ぶと目的がずれうる。
- **順位ラベル（`label_transform="rank"`）は実装済み・既定 OFF**。採点が順位だけで決まることと整合し、
  日付ごとのばらつきの差と外れ値を消せるので、最初に試す候補。既定を raw（仕様の基準ラベル）に
  しているのは、v1 との差分を切り分けやすくするため。

## 2. モデル

- LightGBM 回帰（仕様の既定ハイパラ、`n_jobs=-1`）を `len(horizons) × len(seeds)` = 2×3 = 6本学習する。
  ホライズンごとに seed 平均を取り、**同日断面で z 化してからホライズン平均**する（`horizon_combine="z"`）。
  k126 と k250 の予測はばらつきの大きさが違うので、単純平均（v1、`"mean"`）だと片方が支配しうる。
- 入力は日次断面の順位（0〜1）。`sector17, sector33, market, margin` は順位化せず `categorical_feature`
  として素通しする（v1 と同じ）。
- **列の取捨: 94 → 65列（`model_features="v2_slow"`）**。除外する29列と理由:

| 除外グループ | 列 | 理由 |
|---|---|---|
| REGIME（6） | usdjpy_chg20/60, consumer_level_z, consumer_diff, topix_cum20, topix_vol60 | 同日は全銘柄が同じ値。断面で順位化すると全銘柄が同順位になり、値は 0.5+1/(2n)（n=その日の銘柄数）だけになる。中身は**時期の代理変数**なので、木がこれで期間を切り分けて過学習する。S1 の 94列投入ではこの6列もモデルに入っていた |
| EVENT（4） | disclosed_5d, valid_ratio20, disc_days_bd, progress_1q | 同順位が多い列は、順位の値がその日の同順位の割合で決まるので、決算シーズンの代理変数になる。disc_days_bd は disc_days の重複。progress_1q は NaN 75.6% で、1Q しか値がない |
| FAST（19） | r1, rev1, rev_oc, rev_co, mret, aret, rcc3/5/10, range1, clv1, volratio, res20, gap20, skew20, kurt20, vol5_20, illiq5, volume_z20 | 1〜20日窓の列は順位が毎日入れ替わる。126/250日平均ラベルにはほぼ効かず、回転率を上げるだけになりやすい。S1 では回転率が上がった |

- 残すのは v1 の遅い列（rcc20 以上の窓・財務・流動性・beta など）と、新規の遅い列（hi250/lo250、
  res60/250、長期リバーサル、TTM バリュー、クオリティ成長、改訂率など）。改訂率は NaN 率が約3割あるが、
  LightGBM は欠損を分岐で扱えるので残す。`"v1"`（51列）/ `"all"`（REGIME を除く88列）/ 明示リストへの
  切替と、`drop_features` による追加除外もできる。

## 3. セクター中立化（同一日付内の操作だけ）

1. **ファクター側**（`sector_rank_blocks=["size","value"]`）: 規模・バリューブロックの各列について、
   sector33 内の中心化順位 `(rank-0.5)/n-0.5` を作り、全体順位と一緒にブロック平均へ入れる。
   業種の偏りは約半分になる。この定義なら n=1 の業種でも 0 になり、小さい業種で偏らない。
   有効銘柄が `sector_min_count`（3）未満の業種や、業種コードが欠損の行は NaN にして、全体順位だけを使う。
   P1 の `size_in_sector33` は logsize の業種内順位と同じ順序なので、同じ定義で全列をそろえて再計算する。
2. **モデル側**（`sector_neutral=true`）: `pred - mean(pred | Date, sector33)` を同日断面で z 化して加える。
3. **重み**: 仕様の式どおり `w_model·z(model) + w_sector·z(model_centered)` で、既定は 0.5 + 0.5。
   **モデルの実効重みは v1（0.5）の約2倍になる**ので、`model_weight=0.25, sector_weight=0.25`
   （v1 と同じ総量）との比較を最初のアブレーションに入れる。

## 4. 合成と重み

`signal = Σ_b w_b·z(block_b) + w_model·z(model) + [sector_neutral] w_sector·z(model − 業種平均)`
（z は同日断面の z、±3 で clip）。重みの既定は v1 と同じ（size 1.0 / value 0.5 / quality 0.5 /
lowrisk 0.3 / model 0.5 / sector 0.5）で、すべて `params` で変えられる。

- **`block_set="v2"`（既定）: 累計値を TTM に置き換える**。v1 の sp / ep_a / cfy / roe は四半期の累計値
  （1Q=3か月 … 4Q=12か月）なので、開示のたびに水準がのこぎり状に跳ぶ。決算月や開示日は銘柄ごとに違うので、
  割安度と無関係な順位の入れ替わり（ノイズと回転）が起きる。value は sp_ttm / ep_ttm / cfy_ttm に、
  quality は roe_ttm に置き換える。bp・ep_f・div_y（通期予想）・cfo_ta（TTM 版が特徴量に無い）は v1 のまま。
  `block_set="v1"` で v1 と同じ定義に戻せる。`blocks` に `{block: {列: 符号}}` を渡せば自由に定義できる。
- **欠損ブロック**: v1 と同じく NaN を伝播させる（採点の fillna(0) で中立）。`fill_missing_blocks=true`
  なら、欠損ブロックだけを 0 として扱う。
- 重み 0 のブロックは、欠損を持ち込まずに寄与を 0 にする（`_add_weighted`）。

## 5. マクロ局面配分（実装済み・既定 OFF）

- 拡張期 = `consumer_level_z > consumer_z_min（0）` かつ `topix_vol60 < 学習期間の日次 topix_vol60 の
  vol_quantile（0.5）分位点`。分位点は `fit_model` が学習行だけから決め、モデルに保存する（予測期間は見ない）。
  拡張期は `expansion_block_weights`（size 0.5 / value 0.25 / quality 0.5 / lowrisk 0.3）に切り替える。
- **OFF の理由**: v1 が負けた 2020 年はコロナ禍で消費者態度が急落し、ボラも高かったので、この定義では
  「通常期」になる。弱点を直接は救わない可能性が高い。2局面・月次指標では判定の回数が少なく、
  Train 内で根拠を作りにくい。walk-forward で改善を確認してから ON にする。

## 6. 時点管理（禁止事項への対応）

- `make_label` は `target_1day_train.parquet` だけを読む（Valid 側のファイル名はコードに一度も出てこない）。
  end より後は読み込み直後に捨て、窓がはみ出す行は NaN。逆順 rolling の行には `# check_lookahead: train-ok`。
- 学習・予測には負の shift・center rolling・bfill が無い。順位化・z 化・業種中心化はすべて `groupby(Date)`
  か `groupby(Date, sector33)`。モデル入力は P1 の features だけ。±inf は全列 NaN に統一した
  （置換件数は `features.attrs["inf_replaced"]`）。
- 合成データでの検査: 期間の先頭だけで作ったシグナルが、全期間で作ったシグナルの同じ行と完全に一致した
  （差 0）。日付をまたぐ操作が無いことの確認になる。

## 7. 1フォールドの計算量（目標 5 分以内）

- **行数（概算）**: 最大のフォールド（2015）の学習行は train_end（2014年1月頃）までで、全 609,960行の
  約7割（≈43万行）。ラベルのある行はこれより少ない（k126 は最後の約125日分、k250 は最後の約249日分が NaN）。
- **本数・列**: 2ホライズン × 3 seed = 6本。65列（うち4列は categorical）。350木・31葉・max_bin 127・
  subsample 0.8・colsample 0.7。
- **この sandbox（2 vCPU）での合成データ計測**: 45万行×65列で `rank_for_model` 6.6秒、
  LightGBM 1本（350木）18.7秒、12万行の予測 1.5秒。
- **見積もり**: 6本 × 約19秒 + 順位化 約7秒 + ブロック・業種内順位とラベル作成で数秒 + 予測 約10秒
  ≈ **約2〜2.5分/フォールド（2 vCPU）**で、目標の 5 分に収まる。ローカルで v1 の WF 全体
  （6フォールド・特徴量構築込み）は 241.1秒、94列版は 386秒だった（docs/baseline.md の実測）。
  v2 は列数が両者の間（65列）なので、同程度になる見込み。v2 の実測時間は `{要実行}`。
- **メモリ**: 順位化の作業配列は 65列 × float32 = 1行あたり 260B（45万行で約117MB）。提出時は
  `predict_signal` に予測対象期間の行だけを渡すと、ピーク 2GB の目標に収まりやすい。

## 8. 推奨アブレーション（ローカルの walk-forward で実行。値はすべて `{要実行}`）

| # | params の差分（既定からの変更） | 見たいこと | Train OOS |
|---|---|---|---|
| A | `model_features:"v1", block_set:"v1", sector_rank_blocks:[], sector_neutral:false` | v1 相当（ラベル NaN 化・inf→NaN だけが違う） | {要実行} |
| B | A から `model_features:"v2_slow"` | 列の取捨の効果 | {要実行} |
| C | 既定 | v2 一式 | {要実行} |
| D | `model_weight:0.25, sector_weight:0.25` | モデル総量を v1 にそろえた場合 | {要実行} |
| E | `label_transform:"rank"` / `"vol"` | ラベル変換 | {要実行} |
| F | `block_set:"v1"` | TTM 置換の効果 | {要実行} |
| G | `sector_rank_blocks:[]` / `sector_neutral:false` | 中立化を1つずつ外す | {要実行} |
| H | `regime_mix:true` | 局面配分（C を上回り、かつ Valid の 2020-21 が改善した場合だけ採用） | {要実行} |

採用基準: Train OOS が v1（+1.993）を上回り、かつ年別 Sharpe の最小値が悪化しないこと。Valid は最終確認だけに使う。

## 9. Assumptions

- `alpha_v2.py` は `strategies/v0_multifactor/` に置き、`alpha_features.py`（P1）が同じ場所にある前提。
  実行は `python tools/walkforward.py --strategy strategies/v0_multifactor --module alpha_v2`。
- `make_label` は params を受け取らないので（walkforward の呼び方）、clip などは
  `strategies/v0_multifactor/walkforward_config.json` から読む。`--config` で別ファイルを渡した場合、
  ラベル側には反映されない（`fit_model` 側は params の clip で再度 clip する）。
- target の Date は tz-naive、Code は特徴量と同じ型（v1 の make_label と同じ前提）。
- 業種コードはそのまま categorical に渡す（v1 と同じ）。LightGBM の「sparse values」警告は出るが、
  動作には影響しない。
