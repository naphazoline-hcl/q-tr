# CHANGELOG v6 → v7（モデル項の再構成: LGBM のみ / span 10 / 重み 0.1）

## 変更内容（**コード変更なし・config のみ**）

| パラメータ | v6 | v7 | 意図 |
|---|---|---|---|
| `ensemble_weights.ridge` | 0.4 | **0.0** | モデル項を LGBM のみに単純化（Ridge の追加価値が薄い） |
| `model_weight` | 0.15 | **0.10** | ブロック主体の構成を維持 |
| `model_smoothing_span` | 20 | **10** | LGBM 単独では短めが有効 |
| その他（blocks / quality 0.9 / size_pure 1.8 / sector 中立 OFF） | — | 変更なし | v6 のまま |

## 根拠（v6 後のローカル探索）

- モデル項を LGBM と Ridge に分解し、成分ごとの EWMA スパン×重みを 144 候補で再合成評価
  （`work/search_v7_model.py`。blocks は v6 固定。Valid は正式パイプラインと一致、OOS は fold 境界の
  扱いが異なるため相対比較のみ）。合格したのは ridge を使わないケースのみで、LGBM span 10・重み 0.1 が最良。
- 採用候補は直接パイプライン（walkforward → train_v2 → score --guard）で再確認した。
- 他の v6 後探索（すべて OOS 不採用）: horizons [63,126,250] +2.4474 / `model_features all` +2.4396 /
  quality へ op_margin・prof_margin 追加 +2.3129 / quality へ TTM 追加 +2.3448 / value へ TTM 追加 +2.4309 /
  momentum ブロック +2.3400 / revision ブロック +2.3261 / LGBM 容量増（num_leaves63・n_estimators500・
  lr0.025）は OOS +2.4636 も Valid +1.3719 で不採用。実測: `work/reports/exp_*_wf.json` / `v7_*_wf.json`。

## 実測（採用値）

| 指標 | v6 | **v7（採用）** |
|---|---|---|
| Train OOS Sharpe | +2.4560 | **+2.4667** |
| Valid Sharpe | +1.3783 | **+1.3989** |
| Valid 回転率（コスト） | 0.0130（0.33%/年） | 0.0135（0.34%/年） |
| Valid グロス | +4.63% | +4.70% |
| Valid maxDD / 勝率 | −3.58% / 54.3% | −3.57% / 54.3% |
| Valid RankIC | +0.0074 | +0.0074 |

- Train OOS 年別（v7）: 2010 +1.96 / 2011 +3.00 / 2012 +1.29 / 2013 +2.71 / 2014 +2.28 / 2015 +3.75
- Valid 年別（v7）: 2016 +2.68 / 2017 +3.76 / 2018 +1.27 / 2019 +2.38 / 2020 +0.38 / 2021 +0.66 /
  2022 +1.70 / 2023 +1.62 / 2024 +1.02 / 2025 +0.73 / 2026 +0.21
- 実測 JSON: `work/reports/v7b_lgbmonly_wf.json` / `v7b_lgbmonly_valid.json`
- **注意**: 改善幅は OOS +0.011 / Valid +0.021 と小さく、Valid の推定誤差（±0.02 程度）と同規模。
  「Train OOS と Valid の両方で改善」という採用条件に基づく乗り換え。ロールバックは
  `work/improve4_cfgs/walkforward_config_v6_backup.json` を live config に戻して train_v2 再実行。
- 提出 zip は config / meta 変更後に作り直す（`python tools/make_zip.py --name v0_multifactor --score`）。
