# P1 特徴量ライブラリ v2 — 作業報告

成果物: `alpha_features.py`（94列）/ `selftest_features.py` / `FEATURES.md` / 本ファイル。

## 1. 受け入れチェック（§8）

| 項目 | 結果 | 根拠 |
|---|---|---|
| 既存51列の列名を変えていない | OK | `V1_COLUMNS` を先頭に同名・同順で出力。合成データで `rf_features.build_features()` と **51列すべてビット一致** |
| 新規列を含め 75〜95 列 | OK | 51 + 43 = **94列** |
| `end` 引数が動作する | OK | 全入力を読込時に `Date <= end` で絞る。selftest 5 で最大日付 <= end を検査 |
| `FEATURE_GROUPS` を定義 | OK | 6グループ（12 / 9 / 7 / 4 / 5 / 6 列）＝新規43列と一致 |
| `selftest_features.py` が6項目を出力 | OK | 合成データで実行し exit 0。`shift(-1)` を故意に混入すると検査5が検出し exit 1 |
| 禁止パターン | OK | リポジトリの `tools/check_lookahead.py --submission deliverables` → `RESULT: OK (no ERROR patterns)`。逆順処理・train-ok マーカーは不使用 |
| 財務の基準そろえ（R） | OK | `_rfactor` を開示日に as-of。合成データの2:1分割で `bp`・`ep_a` が連続、`eps_revision` に偽の −50% が出ないことを確認 |
| マクロは `Date` で backward as-of | OK | `_macro_asof()`。`ObservationDate` は読まない |
| `FEATURES.md` に全列の定義表 | OK | 94列すべて（定義・使用データ・因果性の根拠） |

## 2. 検証の内容と限界

- 実データはこの環境に無いため、`docs/data_schema.md` の列名・型（tz-aware の開示日、`(Date, Pair)` index の ECB、
  23:59:59 JST の消費者態度指数、string の listed_info）を再現した**合成データ**で、採点と同じ
  pandas 3.0.3 / numpy 2.4.6 / pyarrow 24.0.0 を使って実行した。
- 合成データでの確認: TTM の手計算一致（例: 2Q 累計 + 前年通期 − 前年 2Q）、1Q 進捗率、営業日カウント、
  分割前後の連続性、因果性検査（end=2014-06-30）で全94列一致。
- 実データでの実行時間・ピークメモリ・NaN 率・記述統計は **{要実行}**:
  `python selftest_features.py --data-dir input`（既定の因果性検査は `--cut 2014-06-30`）。

## 3. Assumptions（§9）

1. **同日複数開示の順序**: v1 は `sort_values("Date")`（安定でない）だったため、`(Date, DisclosureNumber)` の安定ソートに固定した。
   v1 列の値が変わりうるのは「同一銘柄・同日に複数開示」がある行だけ（実データでの件数は {要実行}）。
2. **マクロの時刻**: パネル行を取引日の 00:00 とみなす保守的な結合。23:00 / 23:59:59 JST に利用可能になった値は翌営業日から使う。
   消費者態度指数の調査方法変更（`MethodRegime`: direct_visit → mail）による水準差は補正していない。
3. **TTM**: 期間長は `CurrentPeriodStartDate/EndDate` の月数（3/6/9/12）、会計年度は `CurrentFiscalYearStartDate`。
   前年同期・前年通期が無い場合や期間長が 3/6/9/12 以外（決算期変更など）は、その時点で最新の通期実績で代用。
4. **予想の対象年度**: 通期決算の行では `NextYearForecast*` を翌年度の予想として扱う（schema の欠損率から、
   通期行では当期予想が空で翌期予想が入る前提）。改訂率は同じ対象年度の前回予想との比較のみ。
5. **改訂率の式**: `new/prev − 1` の代わりに `(new − prev)/|prev|` を採用（prev>0 なら同値。赤字予想でも符号が正しい）。
6. `volume_z20` は対数出来高（Volume>0）で計算。`gap20` は仕様どおり `rev_co` の20日平均（`co20` とほぼ比例し冗長）。
7. `progress_1q` は純利益（`Profit / ForecastProfit`）。予想 <= 0 のときは NaN。`div_chg` は250営業日前の予想配当との比。
8. `size_in_sector33` は (0,1] のパーセント順位（中心化しない）。`SECTOR_COLUMNS` は整数コードを float32 で保持（v1 と同じ）。
9. `start` の意味は v1 と同じ（rolling の助走は start から）。マクロ表と決算短信は z スコア・as-of の履歴のため start で絞らない（どちらも過去方向のみ）。
10. `disc_days_bd` は開示日がパネル初日より前のとき NaN（営業日を数えられないため）。
11. メモリ: 列は作った時点で float32 化し、最後に F 順の float32 配列 1 つへ dict を空けながら詰める（DataFrame 化でコピーしない）。実測ピークは {要実行}。

## 4. 次にやること

1. 実データで `selftest_features.py` を実行し、時間・メモリ・NaN 率を確認（v1 比 +43列で目標 1〜3分 / 2GB 以下）。
2. `tools/walkforward.py` で v2 特徴量のモデルを再学習し、v1 の Train OOS +1.993 / Valid +0.754 と比較。
3. 冗長な列（`gap20` と `co20` など）と寄与の小さい列を、グループ単位の除外実験で整理。
