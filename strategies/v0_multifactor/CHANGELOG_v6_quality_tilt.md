# CHANGELOG v5 → v6（予測側の再配合: quality tilt / sector 中立化 OFF / lowrisk 撤去）

## 変更内容（**コード変更なし・config の予測側パラメータのみ**）

2026-10-05 のローカル成分診断（P8 後の追加作業）で採用。学習側（特徴量・ラベル・LGBM / Ridge の学習条件・
seeds・モデル構造）は v5 と完全に同一で、予測の合成側だけを変更した。

| パラメータ | v5 | v6 | 意図 |
|---|---|---|---|
| `sector_neutral` | `true` | **`false`** | 2020/2024-26 で中立化が逆効果。OOS / Valid の両方で改善 |
| `block_weights.size_pure` | 2.5 | **1.8** | size 偏重を緩和（単独では強いが 2020 に弱い） |
| `block_weights.quality` | 0.5 | **0.9** | 2020 のグロース相場・直近のディフェンシブ局面で有効 |
| `block_weights.value` | 0.5 | **0.3** | 直近のバリュー逆風を軽減 |
| `block_weights.lowrisk` | 0.3 | **0.0** | 単独 Valid −0.140 / OOS +0.599 と弱く、合成の足を引っ張っていた |
| `model_weight` | 0.25 | **0.15** | モデル項はブロックに比べ 2020/2024-26 で劣化 |
| `ensemble_weights.ridge` | 0.6 | **0.4** | LGBM 項の比率を維持しつつリッジを縮小 |
| `model_smoothing_span` | 20 | 20 | 変更なし |
| `smoothing_span`（最終 EWMA） | 5 | 5 | 変更なし |

## 探索方法（過学習を避けるための手順）

1. `tools/sweep_improve2.py` で v5 config の成分（`blk_*` / `p_lgbm` / `p_ridge` / `sector33`、OOS 6 fold + Valid）を
   **再生成**（旧キャッシュは v5 以前のブロック定義なので退避）。
2. 予測側パラメータだけを変更して再合成し、Train OOS と Valid を同時評価（`work/search_v5_diag.py` 112 候補 +
   `work/search_v5_diag2.py` 102 候補。選抜は Train OOS 主・Valid 確認。回転率 ≤ 0.017 を必須）。
3. 最良 OOS 候補 `v6a_q09_lr0` を**直接パイプライン**（walkforward → train_v2 → score valid）で再確認。

予測側パラメータ（block_weights / sector_neutral / model_weight / model_smoothing_span / ensemble_weights）は
学習済みモデルに影響しないため、成分からの再合成は walkforward と同一の数値を与える（v5_full が +1.2150 を
再現することで確認済み）。

## 実測（採用値）

| 指標 | v5 | **v6（採用）** |
|---|---|---|
| Train OOS Sharpe | +2.2857 | **+2.4560** |
| Train OOS 回転率 | 0.0128 | 0.0135 |
| Valid Sharpe | +1.2150 | **+1.3783** |
| Valid 回転率（コスト） | 0.0109（0.27%/年） | 0.0130（0.33%/年） |
| Valid グロス | +4.37% | +4.63% |
| Valid maxDD / 勝率 | −3.91% / 54.4% | −3.58% / 54.3% |
| Valid RankIC | +0.0076 | +0.0074 |

- Train OOS 年別（v6）: 2010 +1.96 / 2011 +2.97 / 2012 +1.20 / 2013 +2.67 / 2014 +2.28 / 2015 +3.87
- Valid 年別（v6）: 2016 +2.55 / 2017 +3.74 / 2018 +1.24 / 2019 +2.39 / **2020 +0.35** / 2021 +0.66 /
  2022 +1.79 / 2023 +1.58 / 2024 +0.98 / 2025 +0.70 / 2026 +0.20
  （v5 比: 2020 −0.84 → +0.35、2016/2017/2019/2024 改善。2018 1.54→1.24 / 2022 2.30→1.79 / 2023 1.94→1.58 は低下）

## 検証

- `tools/walkforward.py`: +2.4560（`work/reports/direct_v6a_q09_lr0_wf.json`）
- `train_v2.py` → `tools/score.py --split valid --guard`: +1.3783（`work/reports/direct_v6a_q09_lr0_valid.json`）
- 採用条件（Train OOS > +2.2857 かつ Valid > +1.2150 かつ回転率 ≤ 0.017）をすべて満たす。
- ロールバック: `work/improve4_cfgs/walkforward_config_v5_backup.json` を live config に戻して train_v2 再実行。

## 次の課題

- Valid +1.378 で第二目標 +1.5 まで残り +0.12。残る弱点は 2018 / 2022 / 2023 / 2025-26。
- 予測側の再配合はほぼ上限。次は新しい alpha 源（速いホライズンの成分、新特徴、レジーム対応）が必要。
- 提出 zip は config / meta 変更後に作り直す（`python tools/make_zip.py --name v0_multifactor --score`）。
