# ベースライン実測値（v0 が超えるべき水準）

v0 は数値を捏造してはいけない。以下の数値はローカルで実測済みの事実である。

## 1. 配布サンプル（Valid 2016-04-01 〜 2026-07-31）

| 戦略 | 主に使うデータ | 日次回転率 | Valid Sharpe |
|---|---|---|---|
| sample01_short_reversal | raw_return / beta / topix | 1.337 | −6.35 |
| sample02_size_liquidity | prices / fins / listed_info | 0.013 | +1.12 |
| sample03_fundamental_longhorizon | fins / prices | 0.033 | +0.55 |
| sample04_multifactor | prices / fins / listed_info | 0.030 | +1.14 |
| sample05_macro_regime | consumer / fins / prices | 0.036 | +0.80 |
| 参考: 等ウェイト無条件 | — | — | ≈ 0 |

## 2. 現在の自作ベースライン（`strategies/v0_multifactor`、コピー元は仮提出 robust_multifactor）

設計: 4ブロックのファクター合成（規模・流動性 / バリュー / クオリティ / 低リスク） +
LightGBM 6本（k=126,250 × seed 0,1,2、長期平均ラベル・クリップ±5%）+ EWMA(span=5)。

### Train 内 purged walk-forward（2010〜2015 を毎年検証・未来ラベル除外）

**仮提出 README の記載値（参考・当時の探索条件）**:

| 構成 | Train OOS Sharpe |
|---|---|
| LGBM（1日ラベル・平滑化なし） | −1.79（RankIC +0.027 でもコスト年率19%） |
| LGBM（k=126 ラベル・purge・span5） | +1.34 |
| LGBM（k=250 ラベル・purge・span5） | +1.44 |
| 規模ブロックのみ | +1.99 |
| クオリティブロックのみ | +1.11 |
| 低リスクブロックのみ | +0.59 |
| バリューブロックのみ | +0.25 |
| **本戦略（ブロック + モデル0.5 + span5）** | **約 +2.4** |

> ※ 上表は仮提出時の探索メモ（リンク先の条件での値）。本リポジトリの走査基盤
> `tools/walkforward.py` で再計測した v1 の再現値（`work/reports/wf_baseline_v1.json`）は
> **Sharpe +1.993**（グロス +8.52% / コスト +0.81% / 回転率 0.032 / 年別:
> 2010 +0.90 / 2011 +1.86 / 2012 +0.88 / 2013 +2.14 / 2014 +2.71 / 2015 +4.56）。
> 差分は purge（当時は k+2 日・本基盤は一律 260 日）とラベル末尾の扱いの違いによる。
> **v1 の正式な比較基準は再現値 +1.993 とする**。
> ※ 実採点環境に合わせた lightgbm 4.1.0 でも再計測し、値が完全一致することを確認済み
> （`work/reports/wf_baseline_v1_lgb410.json`、Sharpe +1.993 / elapsed 241.1s）。

### v2（alpha_v2）の検証と選択構成

P2 の alpha_v2 を実データで検証した（既定構成 = +1.879 で v1 未満）。MODEL_DESIGN §8 の
アブレーション 12 構成を実行し、Train OOS と年別 Sharpe の最小値で次を採用した。

| 構成 | Train OOS | 年別最小 | 回転率 | 備考 |
|---|---|---|---|---|
| v1（比較基準） | +1.993 | +0.88 | 0.0320 | `work/reports/wf_baseline_v1.json` |
| P2 既定（v2_slow 特徴量・TTM ブロック・セクター中立化 ON） | +1.879 | −0.30 | 0.0250 | `work/reports/wf_v2_alpha.json` |
| **採用**（v1 特徴量/blocks + rank ラベル + 中立化 OFF） | **+2.186** | +0.89 | 0.0227 | `work/reports/wf_v2_final.json`、`strategies/v0_multifactor/walkforward_config.json` |
| 参考: 同構成で raw ラベル | +2.192 | +0.75 | 0.0224 | 総合は僅差で最良だが 2012 が +0.75 に低下 |

主な所見（すべて Train OOS、詳細は `work/reports/abl_*.json`）:

1. 新規特徴量・TTM ブロック・セクター中立化は Train OOS を下げた
   （94列そのまま v1 モデル接続 = +1.871、v2_slow 65列 = +2.15、中立化 ON で −0.11〜−0.14）。
2. 改善は基盤側から: ホライズンごとの断面 z 平均（`horizon_combine="z"`）、厳密ラベル
   （窓が end を超える行は NaN）、inf→NaN。
3. ラベル変換 rank は raw とほぼ同等（+2.186 / +2.192）で年別のばらつきが小さく、採用。
4. fold 2010 は厳密ラベルにより学習 0 行（ブロック合成のみ）。全構成で共通の条件。

実行した全構成（既定 C からの差分。詳細は `work/reports/abl_*.json`）:

| # | 構成 | Train OOS | 年別最小 | 回転率 |
|---|---|---|---|---|
| A | v1 特徴量 / v1 blocks / 中立化 OFF（raw ラベル） | +2.1925 | +0.75 | 0.0224 |
| **J2** | **A + rank ラベル（採用）** | **+2.1861** | **+0.89** | **0.0227** |
| I2 | A + TTM blocks | +2.1498 | +0.84 | 0.0198 |
| B | v2_slow 特徴量 / v1 blocks / 中立化 OFF | +2.1478 | +0.64 | 0.0212 |
| I3 | A + model_weight 0.25 | +2.1367 | +0.89 | 0.0193 |
| I1 | A + セクター中立化 ON | +2.0844 | +0.04 | 0.0298 |
| J3 | A + vol ラベル | +2.0371 | +0.77 | 0.0206 |
| F | 既定 + blocks v1 | +2.0074 | −0.16 | 0.0275 |
| G1 | 既定 + 業種内順位 OFF | +2.0049 | +0.10 | 0.0249 |
| G2 | 既定 + モデル中立化 OFF | +2.0039 | +0.45 | 0.0186 |
| v1 | 比較基準（再現値） | +1.9930 | +0.88 | 0.0320 |
| D | 既定 + 重み 0.25/0.25 | +1.9617 | +0.45 | 0.0197 |
| C | P2 既定（v2_slow / TTM blocks / 中立化 ON） | +1.8793 | −0.30 | 0.0250 |

#### Valid 実測（2016-04-01 〜 2026-07-31、提出候補の最終確認）

| 構成 | Train OOS | Valid Sharpe | 回転率 | グロス/コスト | RankIC |
|---|---|---|---|---|---|
| **v5 size_pure w2.5（現行 config: v4/S5 + size項再配合）** | **+2.286** | **+1.215** | 0.0109 | +4.37% / 0.27% | +0.0076 |
| v4 S5（旧: K1 + ridge0.6 + model_weight 0.25 + span 20） | +2.124 | +0.803 | 0.0133 | +3.27% / 0.33% | +0.0075 |
| v3 K1（旧候補: I1 + model_smoothing_span 10） | +2.039 | +0.759 | 0.0161 | +3.39% / 0.41% | +0.0079 |
| v1（旧暫定提出） | +1.993 | +0.754 | 0.0169 | +3.42% / 0.43% | +0.0084 |
| v2 I1 | +2.084 | +0.747 | 0.0216 | +3.47% / 0.54% | +0.0081 |
| v2 A（中立化 OFF） | +2.192 | +0.698 | 0.0182 | +3.39% / 0.46% | +0.0088 |
| v2 J2（中立化 OFF + rank ラベル） | +2.186 | +0.633 | 0.0175 | +3.09% / 0.44% | +0.0086 |

- **提出候補は v4 S5（Train +2.124 / Valid +0.803 / 回転率 0.0133）**。K1 に P5 の Ridge アンサンブル
  （`ensemble_weights.ridge: 0.6`、モデル予測を同日断面 z にしてから合成）・モデル項の
  `model_weight: 0.25`（K1 は 0.5）・`model_smoothing_span: 20`（K1 は 10）を適用した。
  Train OOS・Valid の両方を上回り、Valid 回転率も 0.0161 → 0.0133 へ低下（コスト 0.41% → 0.33%）。
- P5 のスイープ（`work/reports/sweep_improve2.md`、成分保存→再合成で 25 候補を一括評価）の主な実測:
  | 候補 | Train OOS | Valid | 回転率 | 採否 |
  |---|---|---|---|---|
  | **mw025_ridge06_span20（採用）** | **+2.124** | **+0.803** | 0.0133 | 採用条件すべて OK |
  | mw025_ridge07_span20 | +2.110 | +0.804 | 0.0130 | 同水準（推定差の範囲内） |
  | mw02_ridge06_span20 | +2.086 | +0.806 | 0.0135 | 同水準 |
  | ridge10（ridge のみ 1.0、他は K1） | +2.109 | +0.794 | 0.0132 | OK（単一変更ではこれが最良） |
  | ridge06 | +2.096 | +0.771 | 0.0138 | OK |
  | neutral_only_ridge03_span20 | +2.011 | +0.797 | 0.0143 | Train OOS が +2.039 未満のため不採用 |
  | E1（ridge0.3 のみ、P5 出荷時既定） | +2.069 | +0.747 | 0.0145 | Valid が K1 未満のため不採用 |
  | regime_v4 / regime_v4_ridge03 | +1.972 / +2.019 | +0.744 / +0.751 | 0.0165 / 0.0149 | 不採用（改善なし） |
  | span5 | +2.074 | +0.767 | 0.0179 | 回転率 > 0.017 のため不採用 |
- v4 S5 の Valid 年別: 2016 +1.49 / 2017 +3.00 / 2018 +1.65 / 2019 +0.87 / 2020 −0.39 / 2021 −0.26 /
  2022 +1.37 / 2023 +1.79 / 2024 +0.48 / 2025 +0.46 / 2026 −0.14
  （分位 Q1 −0.41 / Q2 +0.87 / Q3 +0.02 / Q4 +1.15 / Q5 +3.30 bp/日）。
- K1 の Valid 年別: 2016 +1.19 / 2017 +2.84 / 2018 +1.61 / 2019 +0.90 / 2020 −0.40 / 2021 −0.08 /
  2022 +1.58 / 2023 +1.72 / 2024 +0.43 / 2025 +0.29 / 2026 −0.22
  （分位 Q1 −0.33 / Q2 +0.68 / Q3 −0.06 / Q4 +1.23 / Q5 +3.41 bp/日）。
- P4 のスイープ（`work/reports/sweep_improve1.md`、26 候補を Train OOS / Valid 一括評価）:
  - 採用条件（Train OOS ≥ +2.0 / Valid > +0.754 / 回転率 ≤ 0.017）合格は K1 と msmooth20
    （+2.036 / +0.7544 / 0.0144）。Train OOS 優先で K1 を採用。
  - Valid がより高い候補（neutral_only +0.814 / model_half +0.775 / mw025 +0.773）は回転率 0.019〜0.021 で条件外。
    単独成分では size ブロックが Valid +0.847 / 回転率 0.0071（Train OOS +1.883）。P5 のアンサンブル材料。
- P5 後のローカル追加探索（2026-10-04、実測。いずれも v4/S5 を上回らず不採用。`work/reports/sweep_improve2.*`:
  rank/span 検証、`profile_span20.json` / `profile_span5.json`）:
  - span 微調整: span 10〜20 で Valid +0.797〜+0.813 のプラトー（span12 +2.090 / +0.813 / 0.0140、
    span20 +2.124 / +0.803 / 0.0133）。Train OOS 最大の span20 を維持。
  - rank 併用（lambdarank、成分は実測済み）: 最大 mw025_ridge06_rank06_span20 +2.104 / **+0.812** / 0.0129
    だが Train OOS が採用版 +2.124 未満のため不採用（Valid の +0.01 は標準誤差 ≈0.02 の範囲）。
  - slow_profile（ラベル設計の変更）: span20（k250 のみ学習）Train OOS +2.096 / span5（k126 のみ）
    +2.108（回転率 0.0207）。どちらも採用版 +2.124 未満で不採用。
   → 現行設計の局所探索は出尽くし。次は新ブロック・ロバスト化・学習スキームなど構造変更が必要（P7 候補）。
- P7（改善3: 新ブロックとロバスト化）の実測（2026-10-04、`work/reports/improve3_wf_*.json`、基準は v4/S5 +2.1240 再現確認済み）。
  いずれも基準未達のため不採用。提出候補は v4/S5 のまま:
  | 候補 | Train OOS | 備考 |
  |---|---|---|
  | v3（block_set v3、Ridge 経由のみ） | +2.022 | 新ブロックは Ridge 係数としてのみ使用 |
  | v3_direct（v3 + 直接合成 momentum/revision 0.2） | +2.087 | |
  | decay250 / decay500（sample_decay_halflife） | +2.089 / +2.088 | 学習重みの時間減衰のみ |
  | topk25（feature_top_k 25） | +2.093 | 2パス学習。8候補中の最大だが基準未達 |
  | v3+decay500 / v3+topk25 / all_on（v3+decay750+topk30） | +2.055 / +2.049 / +2.046 | 組み合わせは単独より低下 |
  - OFF 既定でのビット一致は合成データで確認（selftest_p7: S5/K1/S5+rank/S5+regime_mix の4構成で max|diff|=0）。
    実データでも P7 反映後の Valid は +0.8030（回転率 0.0133）で v4/S5 と同一（`work/reports/improve3_valid.json`）。
  - selftest_p7_pipeline（全部 ON の train_v2 → meta → submission 一貫性）OK、truncation OK、lookahead 静的・runtime OK。

#### ローカル採用: size 項の再配合（v5、2026-10-04）

v4/S5 は Valid で size/流動性の頑健なプレミアムを取り切れていない疑いがあった（配布 sample02 は pure size で
Train +1.94 / Valid +1.18。単独 size ブロックは Valid +0.847 で、illiq60/logturn60 の等ウェイト混合が
Valid で薄まる）。**コード変更なし・config の blocks 上書きのみ**で size 項を再配合した:

- `params.blocks = {"size_pure": {"logsize": -1.0}, value/quality/lowrisk は v1 のまま}`
  （旧 `size` ブロック `{logsize -1, illiq60 +1, logturn60 -1}` を置き換え）
- `params.block_weights = {"size_pure": 2.5, "value": 0.5, "quality": 0.5, "lowrisk": 0.3}`
- Ridge は新 4 ブロックで再学習（Train のみ）。LGBM は特徴量ベースで不変。

探索（すべて実測、`work/reports/size_grid*.json/md`）:

| 段階 | 内容 | 結果 |
|---|---|---|
| size_grid / grid2 | 既存 size ブロックの weight 増（1.5〜3.0）と model_weight 併用 | Valid +0.87 まで改善するが OOS は +2.04 前後へ低下。採用条件（OOS > +2.124）に届かず |
| size_grid3b | 素の z(-logsize) / z(rank(-logsize)) 項を per-fold EWMA で正確に再評価 | OOS/Valid 同時改善（rank 版 w2.0: OOS +2.238 / Valid +1.064） |
| walkforward（blocks 上書き + Ridge 再学習） | size_pure weight 1.5 / 2.0 / 2.5 / 3.0 | OOS +2.2801 / +2.2833 / **+2.2857** / +2.2248。OOS 最大の **w2.5 を採用** |

採用の実測（`work/reports/size_pure_w25.json` / `size_pure_w25_valid.json`）:

- Train OOS **+2.2857**（回転率 0.0128、年別: 2010 +1.75 / 2011 +2.83 / 2012 +0.73 / 2013 +2.67 / 2014 +2.01 / 2015 +3.91）
- Valid **+1.2150**（回転率 0.0109、コスト 0.27%、グロス +4.37%、maxDD −3.91%、勝率 54.4%、RankIC +0.0076）
- Valid 年別: 2016 +2.65 / 2017 +3.53 / 2018 +1.54 / 2019 +1.87 / 2020 −0.84 / 2021 +0.78 /
  2022 +2.30 / 2023 +1.94 / 2024 +0.83 / 2025 +0.78 / 2026 +0.17
  （分位 Q1 −0.36 / Q2 −0.18 / Q3 +0.15 / Q4 +1.20 / Q5 +4.12 bp/日）
- 検証: train_v2 → score（--guard）一致、lookahead 静的・runtime OK、truncation OK。
- 注意: 2020 は −0.39 → −0.84 と悪化（それ以外の年はほぼ改善）。ロールバックは
  `work/improve3_cfgs/walkforward_config_v4s5_backup.json` を live config に戻して train_v2 再実行。
- **第一目標（Valid +1.2）達成。**次は +1.5 を目指す。

#### P8（改善4: 学習型合成と線形成分）の検収と実測（2026-10-05、不採用）

v0 の受領パッケージ（`work/s8_package`）は GitHub の v4/S5 コードベースだったため、現行の P7 コードと
3-way マージして反映（`alpha_v2.py` / `ensemble.py` / `submission.py` / `train_v2.py` 更新 +
`blend.py` / `selftest_p8.py` / `CHANGELOG_v5v6.md` / `REPORT_P8.md` 追加）。マージ後も既定 OFF で
v5 とビット一致することを確認（`selftest_p8.py --reference work/ref_v5` で max|diff|=0、walkforward 基準
+2.2857 を再現）。その上で候補を実測:

| 候補 | Train OOS | 回転率 | 判定 |
|---|---|---|---|
| `ensemble_weights.linear` = 0.1 / 0.3 / 0.5 / 1.0 | +2.2837 / +2.2486 / +2.2474 / +2.2402 | 0.0128〜0.0130 | 不採用（OOS 未達、重み増で単調低下） |
| `linear` = 0.6 + `linear.l2` = 0.03 | +2.2569 | 0.0129 | 不採用 |
| `blend_learning` targets=["blocks"] shrink 0.5 | **+2.3057** | 0.0123 | OOS は基準超えも **Valid +1.1742**（グロス +4.22% / 回転率 0.0106 / maxDD −3.90%）で未達 → 不採用 |
| `blend_learning` targets=["ensemble","model"] shrink 0.5 | +2.1807 | 0.0126 | 不採用 |
| `linear` 0.3 + blend all | 未実行（成分単独がいずれも未達のため） | — | 不採用 |

- 実測 JSON: `work/reports/improve4_wf_baseline.json`（基準 +2.2857）、`improve4_lin_*.json`、
  `improve4_blend_*.json`、`improve4_blend_blocks_valid.json`、`improve4_restore_v5_valid.json`
  （v5 復元後に Valid +1.2150 / 回転率 0.0109 / グロス +4.37% を再現）。
- 結論: P8 の 2 機能はどちらも v5 を上回らず、**提出候補は v5 のまま**。コードは既定 OFF・v5 ビット一致の
  まま保持（将来の再利用用）。linear / blend は P7 の `sample_decay_halflife` に未対応（受領版が S5 ベースのため）。
- 参考: `selftest_p7.py` は現行 v5 config の明示 `blocks` とテスト前提が食い違うため通常実行では NG 表示だが、
  config なし隔離コピーでは RESULT OK（P8 マージによる回帰なし）。受領 alpha_v2.py の文字化け 1 行は復元。

### Valid 実測（2016-04-01 〜 2026-07-31）

> ※ lightgbm 4.1.0（実採点環境）で再計測しても値は完全一致
> （`work/reports/baseline_v1_valid_lgb410.json`、Sharpe +0.7538）。

```text
SHARPE            : +0.754
グロス年率        : +3.42%
コスト年率        : +0.43%   （日次回転率 0.0169）
年率ボラティリティ: 3.98%
日次勝率          : 52.6%
RankIC            : +0.0084
年別Sharpe: 2016 +1.35 / 2017 +3.04 / 2018 +1.62 / 2019 +0.96 / 2020 −0.32 /
            2021 −0.51 / 2022 +1.29 / 2023 +1.73 / 2024 +0.42 / 2025 +0.39 / 2026 −0.22
```

### 弱点（改善の狙い目）

1. **2020〜2021 のグロース相場**で負け（サブ期間 2018-2021 の Sharpe は +0.38）。
   規模・バリュー偏重が裏目に出ている。局面判定（`consumer_attitude_index` や
   市場ボラ・モメンタム）でブロック配分を切り替える余地。
2. Train OOS +2.4 に対し Valid +0.754 とギャップが大きい。
   レジーム変化（低金利・グロース優位）に対するロバスト性が不足。
3. セクター中立化をしていないため、業種への偏りがリスクになっている。
4. 財務は四半期累計値のまま使っており、TTM 化・加速度が未実装。
5. モデルは LGBM のみ。線形モデルや別損失（順位回帰・分位回帰）との
   アンサンブルは未検討。

## 3. 目標

- 第一目標: Valid Sharpe を **+1.2 以上**（配布サンプル最高 +1.14 を超える）
- 第二目標: **+1.5 以上**（Train OOS と Valid のギャップを縮める）
- 制約: 予測コードの実行 30 分以内・低メモリ・先読み禁止・CPU のみ
