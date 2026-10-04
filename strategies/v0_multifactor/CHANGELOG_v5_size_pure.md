# CHANGELOG v5（ローカル採用）: size 項の再配合

基準は v4/S5（Train OOS +2.1240 / Valid +0.8030 / 回転率 0.0133）。**コード変更なし**、
`walkforward_config.json` の `params.blocks` / `params.block_weights` の上書きのみ。

## 変更内容

```json
"blocks": {
  "size_pure": {"logsize": -1.0},
  "value":  {"bp": 1.0, "sp": 1.0, "ep_f": 1.0, "ep_a": 1.0, "cfy": 1.0, "div_y": 1.0},
  "quality": {"roe": 1.0, "cfo_ta": 1.0, "eq_ratio": 1.0},
  "lowrisk": {"beta": -1.0, "vol60": -1.0}
},
"block_weights": {"size_pure": 2.5, "value": 0.5, "quality": 0.5, "lowrisk": 0.3}
```

- 旧 `size` ブロック（`logsize -1, illiq60 +1, logturn60 -1` の等ウェイト平均）を廃止し、
  **logsize 単独**のブロックに置き換えた。`blocks` は REPLACE キーのため、config の `block_set` より優先される。
- `size_pure` は `sector_rank_blocks`（既定 `["size","value"]`）に含めない（業種内順位は従来どおり value のみ…実際は size という名前のブロックが無くなったため size_pure には掛からない）。
- Ridge は新 4 ブロックで再学習（Train のみ）。LGBM は feature ベースで変更なし。推論式・API 契約は不変。

## 根拠（すべて実測。work/ は gitignore）

1. 配布 sample02（pure size/liquidity）は Train +1.94 / Valid +1.18。単独 size ブロックは Valid +0.847 で、
   illiq60/logturn60 の混合が Valid で薄まる疑いがあった（`docs/baseline.md`）。
2. 既存 size ブロックの weight 増（1.5〜3.0）は Valid +0.87 まで改善するが OOS は +2.04 前後に低下
   （`work/reports/size_grid.md` / `size_grid2.md`）。
3. 素の `z(-logsize)` / `z(rank(-logsize))` 項を per-fold EWMA で正確に再評価すると OOS/Valid が同時改善
   （rank 版 w2.0: OOS +2.2379 / Valid +1.0639 / `size_grid3b.json`）。
4. 実装どおり（blocks 上書き + Ridge 再学習）の walkforward:
   w1.5 +2.2801 / w2.0 +2.2833 / **w2.5 +2.2857** / w3.0 +2.2248 → OOS 最大の w2.5 を採用
   （`work/reports/size_pure_w15..w30.json`）。

## 採用の実測（`work/reports/size_pure_w25.json` / `size_pure_w25_valid.json`）

| 指標 | v4/S5 | v5 採用 |
|---|---|---|
| Train OOS Sharpe | +2.1240 | **+2.2857** |
| Train OOS 回転率 | 0.0174 | 0.0128 |
| Valid Sharpe | +0.8030 | **+1.2150** |
| Valid 回転率 / コスト | 0.0133 / 0.33% | 0.0109 / 0.27% |
| Valid グロス | +3.27% | +4.37% |
| Valid maxDD | −4.88% | −3.91% |
| Valid RankIC | +0.0075 | +0.0076 |

- Train OOS 年別: 2010 +1.75 / 2011 +2.83 / 2012 +0.73 / 2013 +2.67 / 2014 +2.01 / 2015 +3.91
- Valid 年別: 2016 +2.65 / 2017 +3.53 / 2018 +1.54 / 2019 +1.87 / 2020 −0.84 / 2021 +0.78 /
  2022 +2.30 / 2023 +1.94 / 2024 +0.83 / 2025 +0.78 / 2026 +0.17
- 検証: train_v2 → score valid（--guard）一致、lookahead 静的・runtime OK、truncation OK。

## 注意・ロールバック

- 2020 のみ悪化（−0.39 → −0.84）。その他の年はほぼ改善。
- ロールバック: `work/improve3_cfgs/walkforward_config_v4s5_backup.json` を
  `strategies/v0_multifactor/walkforward_config.json` に戻し、`train_v2.py` を再実行する。
