# REPORT_P5: 改善ラウンド2（受け入れチェック・検証・時間見積もり・Assumptions）

実データが無い環境での作業。**実データの Sharpe・回転率は 1 つも測っていない（`{要実行}`）。**
下の数字は合成データでの動作確認と、この作業環境（2 CPU）での時間計測だけ。

## 1. 受け入れチェックリスト（P5 §4）

- [x] `ensemble_weights` / `regime_weights` / `turnover_cap` を `params` に追加した
  （ほかに `ridge` / `rank_model` / `regime_mode` / `regime_v4` / `turnover_control` / `turnover_window` / `slow_profile`）
- [x] 既定 OFF の機能がローカルで ON/OFF 両方回せる（設定だけで切替。`selftest_p5.py` の 11 ケースで確認）
- [x] 学習 1 フォールド ≤ 10 分、推論 ≤ 30 分の見積もりを書いた（§3）
- [x] 変更点を `CHANGELOG_v3v4.md` に記録した
- [x] 禁止パターンなし（`check_lookahead.py` 静的検査: 13 ファイル OK）、Valid ラベルを読まない
  （新モジュールはファイルを読まない。ラベルは呼び出し側が Train の target から作って渡す）

## 2. この環境で実行した検証（合成データ・実データ不要）

| 検証 | 結果 |
|---|---|
| `selftest_p5.py --reference 旧alpha_v2.py`: K1 設定で旧版と新版のシグナル差 | max abs diff = 0（ビット一致） |
| 同: ridge / rank(lambdarank・rank_xendcg・quantile) / regime v4・legacy / auto span / slow_profile 5・20 / 全部 ON | 11 ケースすべて動作、すべて K1 と異なるシグナル、有限値 100% |
| 同: 因果性（予測期間の後半を切り落として前半のシグナルが変わるか、全部 ON） | max abs diff = 0 |
| `sweep_improve2.compose`（成分からの再合成）と `predict_signal` + 最終 EWMA の一致（6 候補） | すべて max abs diff = 0 |
| `train_v2.py` → `meta_v2.json` → `submission.predict()` の一致（ridge 0.3 + rank 0.2 + regime v4 + auto） | max abs diff = 0、欠損 0 行。Ridge 係数・rank 4 本・auto span・しきい値が meta に入る |
| `tools/check_lookahead.py --submission`（静的検査のみ。`--runtime` は実データが要るので `{要実行}`） | RESULT: OK |

注: LightGBM 4.1.0 + numpy 2 では、ラベルや group の dtype 変換が要ると `np.array(..., copy=False)` で
例外になる（この環境で再現）。rank モデルのラベルは float32、group は int32 に揃えて渡している。

## 3. 時間見積もり（`bench_p5.py`、合成パネル、この環境 2 CPU）

規模: 学習 1,500 日 × 498 銘柄 ≒ 74.7 万行（Train 全期間の strict ラベル行の概算。fold 学習はこれ以下）、
推論 2,930 日 × 498 ≒ 145.9 万行（2014-06〜2026-07）。列はモデル入力 "v1" の 51 列。

| 処理 | 実測（秒） |
|---|---|
| LGBM 回帰 fit（350 本、1 モデル） | 32.6 |
| Ridge fit（ブロック 4 列） | 0.2 |
| rank fit lambdarank（150 本、2 日おき、long + short の 2 モデル） | 68.4 |
| rank fit rank_xendcg（同、2 モデル） | 60.9 |
| rank fit quantile（150 本、2 日おき、1 モデル） | 81.6 |
| LGBM predict（350 本、1 モデル、推論行） | 22.0 |
| rank predict（150 本、1 モデル、推論行） | 7.7 |

- **学習 1 フォールド（最大構成: LGBM 6 本 + Ridge + lambdarank 2 ホライズン × 2 = 4 本 + auto span 推定）**:
  `6 × 32.6 + 0.2 + 2 × 68.4 ≈ 333 秒` + auto 推定（末尾 250 日 ≈ 12.5 万行の予測 ≈ 15 秒）≈ **約 6 分 < 10 分**。
  既定（E1: LGBM 6 本 + Ridge）は `≈ 196 秒`。特徴量構築・順位化は従来どおり（別計上）。
- **推論（モデル本数 × 推論時間）**: E1 は `6 × 22.0 + Ridge（行列積）≈ 132 秒`、rank 4 本を足しても
  `+ 4 × 7.7 ≈ 31 秒` で **約 2.7 分**。特徴量構築（規約上 1〜3 分）を足しても 30 分に十分収まる。
- 採点環境の CPU 数は不明。CPU が 1/3 の速さでも推論は約 8 分 + 特徴量構築の見込み。rank の学習が重い場合は
  `rank_model.date_stride` を 3〜4 に、`n_estimators` を 100 に下げる。

## 4. Assumptions（不明点と置いた前提）

1. `topix_cum60` は `alpha_features` に無い（あるのは `topix_cum20`）。特徴量追加は避け、日次系列で
   `cum20[t] + cum20[t−20] + cum20[t−40]`（正の shift のみ）として作った。古い窓が欠ける期間は使える部分を 60 日換算。
2. 「消費者指数の変化（トレンド）」は `consumer_diff`（月次差分の as-of）の過去 63 営業日平均とした。
3. S4 の `neutral_only` は「モデル非中立項 0（`model_weight: 0`）・中立項のみ」と解釈した。
4. 局面別ブロック重み（`regime_weights`）と B の各しきい値は**未検証の初期値**（B は既定 OFF）。
5. `turnover_control: "auto"` の推定は学習期間末尾のインサンプル予測から作る（ラベル不使用）。
   インサンプル予測は OOS より鋭い可能性があるため、既定は "off"、span は スイープの実測で決める方を推奨。
6. 既定構成は E1（`ridge: 0.3`）を候補として設定した（P5 の例示値）。実測で T / V / TO を満たさなければ
   `ridge: 0.0`（= K1）に戻す。
7. `slow_profile` のラベル設計は「span 5 ↔ k126 のみ / span 10 ↔ k126 + k250（K1）/ span 20 ↔ k250 のみ」。
   ラベルを変えるため、スイープではなく walk-forward の再学習で比べる。

## 5. 次にやること（人間側・実データで）

```bash
python strategies/v0_multifactor/selftest_p5.py
python tools/walkforward.py --strategy strategies/v0_multifactor --module alpha_v2 --json work/reports/improve2.json
python tools/sweep_improve2.py --strategy strategies/v0_multifactor --module alpha_v2 [--rank]
python strategies/v0_multifactor/train_v2.py --strategy strategies/v0_multifactor
python tools/score.py --submission strategies/v0_multifactor --split valid
python tools/check_lookahead.py --submission strategies/v0_multifactor --runtime
python tools/check_truncation.py --strategy strategies/v0_multifactor --module alpha_v2 --cut 2014-06-30
```

## 6. ローカル検収結果（追記・2026-10-04、実データ実測）

上の §5 をすべて実行した。**出荷時既定の E1（ridge 0.3）は Valid +0.7470 で K1 +0.7591 未満のため
不採用**。sweep_improve2 に局所候補を追加して採用したのは **mw025_ridge06_span20**
（`ensemble_weights {lgbm 1.0, ridge 0.6}` / `model_weight 0.25` / `model_smoothing_span 20`）:
Train OOS **+2.1240** / Valid **+0.8030** / Valid 回転率 **0.0133**（コスト 0.33%/年）。
選択理由: ridge 0.5〜0.7 × model_weight 0.2〜0.3 × span 20 が Valid +0.80 前後のプラトーで、
その中央（OOS が最も高い +2.124、単一値への依存が小さい）を採った。

| 検証 | 実測 |
|---|---|
| `selftest_p5.py`（合成・11 ケース、因果性） | OK。reference 差は検収側の config 解像度の見掛け（同一 config では max abs diff = 0 を別途確認） |
| walkforward（採用 config） | Train OOS +2.1240 / 回転率 0.0174 / 210s（`improve2_final.json`） |
| train_v2 → meta_v2.json | 完了 140s。ridge 係数（k126/k250）・span 20・ensemble 重みが meta に入る |
| score valid（--guard） | **+0.8030** / グロス +3.27% / コスト +0.33% / 回転率 0.0133 / RankIC +0.0075（`improve2_final_valid.json`） |
| lookahead --runtime | OK（shape=(1227148, 1) を完全カバー） |
| truncation（--cut 2014-06-30） | OK（全列 invariance） |

未実測の残課題: rank モデル（`--rank` 付きスイープ）、slow_profile のラベル設計、regime_weights の再調整。
いずれも既定 OFF のまま。Valid +1.2 には追加の改善ラウンドが必要（`{要実行}`）。
