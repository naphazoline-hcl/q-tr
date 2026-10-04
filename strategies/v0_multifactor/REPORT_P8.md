# REPORT_P8 — 改善4（学習型合成と線形成分）実装報告

> 実データは持っていないので、性能（Sharpe 等）は一切測っていない。下の数値はすべて合成データでの
> 動作確認の実測値。採否はローカルの walkforward / sweep で判断する（v0 は検証しない）。

## 受け入れチェック

| 項目 | 結果 |
| :--- | :--- |
| A. 学習型合成 `blend_learning`（blocks / ensemble / model）を追加、既定 `enabled: false` | OK |
| B. 線形成分 `ensemble_weights.linear`（既定 0.0）+ `linear` 設定を追加 | OK |
| 既定 OFF で v5 とシグナルがビット一致（旧モジュール一式を別プロセスで読む `--reference`） | OK（`np.array_equal` = True、max\|diff\| = 0） |
| キーを明示 OFF にしても既定と完全一致 | OK（max\|diff\| = 0） |
| 各機能が設定だけで ON/OFF でき、シグナルが変わる | OK（7 ケース） |
| 内側ホールドアウトの purge（内側学習のラベル窓が cut より前に終わる） | OK（窓の最終日 − cut = −1 取引日） |
| 因果性（予測期間の後半を切っても前半が不変、全部 ON） | OK（max\|diff\| = 0） |
| 提出経路の一致（meta_v2.json を JSON 往復 + LGBM をファイル保存→読込 → `submission.predict_split`） | OK（4 構成で `predict_signal` + EWMA5 と max\|diff\| = 0。sandbox 内の一時スクリプトで確認） |
| 静的 lookahead（`tools/check_lookahead.py --submission`） | OK（15 ファイル、ERROR なし） |
| py_compile（変更・新規の全 .py） | OK |
| 既存 `selftest_p5.py` の回帰 | 新旧で全行同じ数値（秒数以外）。`RESULT: NG ['turnover_auto', 'reference']` は**旧 v4/S5 コードでも同じく出る**既存の仕様（selftest_p5 は walkforward_config.json = K1 を前提にしており、現行 config の span 20 / ridge 0.6 と食い違う）で、今回の変更による回帰ではない |
| 学習時間 ≤ 10 分/フォールド、推論 ≤ 30 分（見積もり） | OK（下記） |
| Valid ラベルを読まない・禁止パターンなし・LightGBM 4.1.0 | OK（ファイル読み込みの追加なし。sandbox は lightgbm 4.1.0 / numpy 2.4.6 / pandas 3.0.3 / scikit-learn 1.9.0） |

### selftest_p8.py の出力（sandbox 実測、合成データ 500 日 × 60 銘柄）

```text
synthetic: train=22,800 predict=7,200 rows
v5_default           1.5s finite=1.000 max|d v5|=0
v5_explicit_off      1.5s finite=1.000 max|d v5|=0
linear_0.3           1.5s finite=1.000 max|d v5|=0.1145
linear_only          0.7s finite=1.000 max|d v5|=0.707
blend_blocks         1.3s finite=1.000 max|d v5|=0.5756
blend_ensemble       1.9s finite=1.000 max|d v5|=0.2372
blend_model          1.8s finite=1.000 max|d v5|=2.552
all_on               2.0s finite=1.000 max|d v5|=5.588
inner purge: max(window end) - cut = -1 trading days (must be < 0)
causality (all_on, truncated tail): max|diff| = 0.00e+00
v5 (OFF) vs reference /tmp/p8test/ref: bit-identical=True max|diff|=0.00e+00
RESULT: OK
```

（reference = GitHub main の v4/S5 コード一式。params は v5 構成: size_pure 2.5 + ridge 0.6 + mw 0.25 + span 20）

## 時間見積もり（REPORT_P5 の実測 LGBM 1 本 32.6 秒・推論 1 本 22.0 秒を基準にした見積もり。実測ではない）

- 既定（v5）: 変化なし（約 196 秒/フォールド）。
- linear ON: 正規方程式の積み上げ（47 列 × 約 100 万行、20 万行チャンク）と行列積だけ。順位行列は LGBM と共有。
  **+数秒/フォールド**、推論 +数秒（`{要実行}`）。
- blend `blocks`: ブロック順位の再計算 + 4 項の解 → **+10〜30 秒/フォールド**（`{要実行}`）。
- blend `ensemble` / `model`: 内側 LGBM（2 ホライズン × `inner_seeds` 1 本 × 約 6 割の行）+ 内側 Ridge / linear +
  ホールドアウト予測 → **約 +60〜90 秒/フォールド**。全部 ON でも約 4〜5 分/フォールド < 10 分。
  重い場合は `inner_n_estimators`（例 150）で削れる。推論は重みが定数なので追加なし。

## Assumptions（不明点と置いた前提）

1. **指定 URL（`v0/prompts/P8_改善4_学習型合成と線形成分.md`）は GitHub main に存在せず 404**（最新コミットは P7）。
   初回メッセージの要約（A: 手動の合成重みの学習化、B: LGBM と別種の線形成分、全部既定 OFF・OFF で v5 と
   ビット一致、採否はローカル）と P7 の書式から仕様を組み立てた。キー名・既定値は本実装の提案であり、
   P8 本文と違う点があれば初回メッセージで指示してほしい。
2. ベースコードは GitHub main の v4/S5 コード（v5 は config のみの変更とのことなので同一とみなした）。
   v5 の `sector_rank_blocks` は不明のため selftest では `["value"]` を仮定（一致検査の結果には影響しない）。
3. 成果物のファイル名は P7 の慣例に合わせ `CHANGELOG_v5v6.md` / `REPORT_P8.md` / `selftest_p8.py`、
   ZIP は `improve4_package.zip`（リポジトリ相対パスで格納）。
4. 学習型合成の「honest な標本」: ブロックはパラメータが無いので学習期間全体、LGBM を含む項は内側ホールドアウト
   （後ろ 30%・ホライズン別 purge）。rank モデルは内側で学習し直さず手動重みのまま（既定 OFF のため）。
5. 線形成分の入力は LGBM と同じ列（`SECTOR_COLUMNS` 除く）に限定した（提出側で順位行列を再利用し、
   メモリと時間を増やさないため）。別の列を使う場合は `model_features` 側を広げる。
6. `tools/sweep_improve2.py` は変更していない（ローカルで拡張済みの版と衝突しないため）。linear の重み sweep は
   `components()` に `p_linear`（`alpha.ensemble.predict_linear(ranked, model.get("linear"))`）を足し、
   `compose()` のキー列に `"linear"` を加えれば再合成できる（学習時は `fit_params` で `linear: 1.0`）。

## 次にやること（人間側の検収）

1. 展開 → `git diff --no-index` で差分レビュー（変更 5 ファイル + 新規 `blend.py` / `selftest_p8.py`）。
2. py_compile → `tools/check_lookahead.py` → `python strategies/v0_multifactor/selftest_p8.py --reference <v5 のコピー>`。
3. walkforward（基準 v5 +2.2857）で CHANGELOG の sweep 候補を評価 → Train OOS と Valid の両方で v5
   （+2.2857 / +1.2150、回転率 ≤ 0.017）を上回ったものだけ採用 → config 更新 → train_v2 → score valid。
