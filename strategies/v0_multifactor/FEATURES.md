# alpha_features v2 特徴量一覧（94列 = v1 51列 + 新規 43列）

- 出力: `build_features(splits=("train","valid"), start=None, end=None)` → index `(Date, Code)`、全列 float32。
- 列順: v1 の51列（同名・同順・同値）→ 新規43列（`FEATURE_GROUPS` の順）。`ALL_COLUMNS` で参照できる。
- `SECTOR_COLUMNS = ["sector17","sector33","market","margin"]` は整数コード値のまま（float32 で保持。順位化しない）。
- `end` を渡すと全入力を `Date <= end` で読み込み時に絞るので、返り値も `Date <= end` の行のみ。

## 記号と共通ルール

| 記号 | 意味 |
|---|---|
| `AO/AH/AL/AC` | `AdjustmentOpen/High/Low/Close`（配布時点の株式数基準へ調整済み） |
| `r1` | `raw_return_1day.Return`（t−1 寄付 → t 寄付。t の寄付で確定） |
| `R` | `AdjustmentClose / Close` を開示日に as-of で持った値（`_rfactor`）。1株当たり値 × R、株数 ÷ R |
| `株数` | `NumberOfIssuedAndOutstandingShares...IncludingTreasuryStock ÷ R` |
| `時価総額` | `AC × 株数` |
| `rolling(N, m)` | 銘柄ごと・過去方向のみの N 営業日窓（最低 m 行）。t を含む |
| as-of（財務） | 決算短信の `Date`（開示日、tz-aware）を naive JST にして backward `merge_asof`。銘柄内で前方埋めのみ |
| as-of（マクロ） | `Date`（日本時間の利用可能日時）で backward。パネル行は 00:00 扱いなので当日夜の公表は翌営業日から |

因果性の根拠は「t 行は t までに確定した値だけを使う」ことを、列ごとに使う操作で示す。
禁止操作（負の shift・center 付き rolling・未来方向の穴埋め・逆順処理・日付をまたぐ順位化）は一切使っていない
（`tools/check_lookahead.py` で ERROR 0 件）。

## v1 の51列（定義は rf_features.py と同一）

| 列名 | 定義 | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `r1` | `r1[t]` | raw_return | t の寄付で確定した過去リターン |
| `rev1` | `AC[t]/AC[t−1] − 1` | prices | 正の shift(1) のみ |
| `rev_oc` | `AC[t]/AO[t] − 1` | prices | 当日終値で確定 |
| `rev_co` | `AO[t]/AC[t−1] − 1`（ギャップ） | prices | shift(1) |
| `mret` | 前場 `MorningAdjClose/MorningAdjOpen − 1` | prices | 当日確定 |
| `aret` | 後場 `AfternoonAdjClose/AfternoonAdjOpen − 1` | prices | 当日確定 |
| `rcc3`〜`rcc250` | `rev1` の rolling(N, 0.7N) 和（N=3,5,10,20,60,120,250） | prices | 過去方向 rolling |
| `mom` | `rcc250 − rcc20` | prices | 同上 |
| `oc20` / `co20` | `rev_oc` / `rev_co` の rolling(20,12) 和 | prices | 同上 |
| `vol20` / `vol60` | `r1` の rolling(20,15) / (60,45) 標準偏差 | raw_return | 同上 |
| `dvol60` | `min(r1,0)` の rolling(60,45) 標準偏差 | raw_return | 同上 |
| `range20` / `range1` | `(AH−AL)/AC` の rolling(20,15) 平均 / 当日値 | prices | 同上 |
| `clv1` | `(AC−AL)/(AH−AL)`、[0,1] にクリップ | prices | 当日確定 |
| `logturn20` / `logturn60` | `log(TurnoverValue)` の rolling(20,15)/(60,45) 平均 | prices | 過去方向 rolling |
| `illiq20` / `illiq60` | Amihud `|r1|/TurnoverValue` の rolling 平均 | prices, raw_return | 同上 |
| `volratio` | `Volume / rolling(20,15) 平均 Volume`（Volume>0 のみ） | prices | 同上 |
| `logsize` | `log(時価総額)` | prices, fins | 財務は開示日 as-of、R で基準そろえ |
| `bp` | `(Equity/株数)/AC` | prices, fins | 同上 |
| `ep_f` | `ForecastEarningsPerShare × R / AC` | prices, fins | 同上 |
| `ep_a` | `EarningsPerShare × R / AC`（期間累計 EPS） | prices, fins | 同上 |
| `sp` | `NetSales/株数/AC`（累計売上） | prices, fins | 同上 |
| `cfy` | `CFO / 時価総額`（累計営業CF） | prices, fins | 同上 |
| `div_y` | `ForecastDividendPerShareAnnual × R / AC` | prices, fins | 同上 |
| `roe` | `Profit / Equity`（累計） | fins | 開示日 as-of |
| `cfo_ta` | `CFO / TotalAssets` | fins | 同上 |
| `eq_ratio` | `EquityToAssetRatio` | fins | 同上 |
| `op_margin` / `prof_margin` | `OperatingProfit/NetSales`、`Profit/NetSales`（累計） | fins | 同上 |
| `sales_yoy` / `profit_yoy` | as-of 値 ÷ 250営業日前の as-of 値 − 1 | fins | as-of 後に正の shift(250) |
| `eps_fwd_chg` | `予想EPS×R − 125営業日前の値` | fins, prices | 正の shift(125) |
| `eps_fwd_yoy` | `予想EPS×R ÷ 250営業日前 − 1` | fins, prices | 正の shift(250) |
| `cf_yoy` | `CFO ÷ 250営業日前 − 1` | fins | 正の shift(250) |
| `disc_days` | 最新開示日からの暦日数 | fins | as-of で得た過去の開示日 |
| `scale` | `ScaleCategory` → Core30=5, Large70=4, Mid400=3, Small1=2, Small2=1（他は NaN） | listed_info | 同日付結合 |
| `sector17` / `sector33` / `market` / `margin` | 各コード文字列を整数化（カテゴリ特徴） | listed_info | 同日付結合 |
| `beta` | `beta_1day.Return[t]` | beta | 配布値（t 行の値） |

## 新規43列

### price_reversal（12列）

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `hi250` | `AC / rolling(250,175) max(AC) − 1`（52週高値からの距離。アンカー効果） | prices | 過去方向 rolling |
| `lo250` | `AC / rolling(250,175) min(AC) − 1`（52週安値からの距離） | prices | 同上 |
| `res20` / `res60` / `res250` | `resid = r1 − beta[t] × topix_return[t]` の rolling(N, 0.7N) 和（市場中立化した過去リターン） | raw_return, beta, topix | 3系列とも t で確定した配布値。過去方向 rolling |
| `lrev120_20` | `rcc120 − rcc20`（直近1か月を除いた中期リターン。長期リバーサル） | prices | 過去方向 rolling の差 |
| `lrev250_60` | `rcc250 − rcc60` | prices | 同上 |
| `gap20` | `rev_co` の rolling(20,12) 平均（`rev_co` がギャップと同義のため、その20日平均） | prices | 同上 |
| `ac1_60` | 銘柄内 `corr(r1[t], r1[t−1])` の rolling(60,45)（短期の過剰反応/反応不足） | raw_return | 正の shift(1) と過去方向 rolling |
| `skew20` / `kurt20` | `r1` の rolling(20,15) 歪度・尖度（宝くじ選好・テールリスク） | raw_return | 過去方向 rolling |
| `updays60` | `r1 > 0` の比率、rolling(60,45)（r1 が NaN の日は除外） | raw_return | 同上 |

### risk_liquidity（9列）

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `vol5_20` | `std5(r1) / vol20`（短期ボラの相対的上昇＝不確実性イベント） | raw_return | 過去方向 rolling |
| `uvol60` | `max(r1,0)` の rolling(60,45) 標準偏差（上方ボラ。`dvol60` と対） | raw_return | 同上 |
| `dvol_uvol60` | `dvol60 / uvol60`（下方リスクの偏り） | raw_return | 同上 |
| `turn_trend` | `logturn20 − logturn60`（売買代金のトレンド＝注目度の変化） | prices | 同上 |
| `illiq5` | Amihud の rolling(5,4) 平均（短期の非流動性） | prices, raw_return | 同上 |
| `volume_z20` | `(logV[t] − mean20(logV)) / std20(logV)`、`logV = log(Volume>0)`、窓は t を含む過去20日 | prices | 過去方向 rolling のみ |
| `highlow60` | `(AH−AL)/AC` の rolling(60,45) 平均（`(High−Low)/Close` と同値。調整係数は同日内で相殺） | prices | 同上 |
| `valid_ratio20` | 過去20行で `AC` が NaN でない比率（売買停止・欠損の代理） | prices | 過去方向 rolling(20,1) |
| `size_in_sector33` | `logsize` の同一 Date・同一 `sector33` 内パーセント順位（0,1]（業種内の相対規模） | prices, fins, listed_info | 同一 Date の断面のみで順位化 |

### value（7列、TTM 基準）

TTM: 各開示行で `今回累計 + 前年通期 − 前年同期累計`（通期行は通期値）。前年同期・前年通期は
「同一銘柄・会計年度キー（`CurrentFiscalYearStartDate` の年月）−12か月・同じ期間長（`CurrentPeriodStart/EndDate` の月数）」
を開示日以前の行から backward as-of で引く。欠ける場合は、その時点で最新の通期実績で代用。

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `ep_ttm` | `Profit_ttm / 時価総額`（季節性のない益回り） | fins, prices | 開示行で過去の開示のみから TTM → 開示日 as-of |
| `sp_ttm` | `NetSales_ttm / 時価総額` | fins, prices | 同上 |
| `cfy_ttm` | `CFO_ttm / 時価総額`（営業CF は主に 2Q・通期で開示） | fins, prices | 同上 |
| `roe_ttm` | `Profit_ttm / Equity`（Equity>0） | fins | 同上 |
| `op_margin_ttm` | `OperatingProfit_ttm / NetSales_ttm`（NetSales_ttm>0） | fins | 同上 |
| `accrual` | `(Profit_ttm − CFO_ttm) / TotalAssets`（大きいほど利益の質が低い） | fins | 同上 |
| `ep_gap` | `ep_f − ep_a`（予想と実績の乖離＝改訂余地） | fins, prices | v1 列の差 |

### quality_growth（4列）

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `equity_ratio_chg` | `EquityToAssetRatio` − 前年同期の値（財務体質の改善） | fins | 前年同期は開示日以前の行のみ |
| `op_margin_yoy_chg` | 営業利益率（累計）− 前年同期の営業利益率（同じ期間長） | fins | 同上 |
| `sales_yoy_acc` | `sales_yoy − sales_yoy[t−250]`（成長の加速度） | fins | 正の shift(250) |
| `div_chg` | `予想年間配当×R ÷ 250営業日前の値 − 1`（分母>0。増配・減配の方向） | fins, prices | 正の shift(250) |

### revision（5列）

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `eps_revision` | 同じ対象年度の予想EPS×R について `(今回 − 前回)/|前回|`（改訂後ドリフト） | fins, prices | 銘柄×対象年度内の正の shift(1)。予想の無い開示行は直前の値を持ち越し |
| `opprofit_revision` | 予想営業利益で同様 | fins | 同上 |
| `progress_1q` | `1Q累計 Profit / 最新の通期予想 Profit`（最新実績が 1Q＝3か月累計のときのみ、予想>0） | fins | 前方埋めした過去の値のみ |
| `disclosed_5d` | 最新開示から 5 営業日以内（0〜4）なら 1、それ以外 0 | fins | 過去の開示日のみ |
| `disc_days_bd` | 最新開示からの経過営業日（パネルの取引日カレンダー。休日開示は翌営業日起点） | fins | 同上 |

予想の対象年度: 通期決算の行は `NextYearForecast*`（翌年度が対象）、それ以外の行は `Forecast*`（当年度が対象）。
対象年度が変わった最初の予想（期初ガイダンス）は改訂ではないので NaN。

### regime（6列、全銘柄に同じ値）

| 列名 | 定義（仮説） | 使用データ | 因果性の根拠 |
|---|---|---|---|
| `usdjpy_chg20` / `usdjpy_chg60` | USD/JPY（ECB 由来）を取引日へ as-of し、20 / 60 取引日前比 − 1（円安局面＝輸出株・大型株優位） | ecb_fx_rates | `Date` で backward as-of、正の shift のみ |
| `consumer_level_z` | 消費者態度指数の過去60公表分（5年）z スコア（最低24本） | consumer_attitude_index | `Date` で backward as-of、過去方向 rolling |
| `consumer_diff` | 消費者態度指数の前回公表差 | consumer_attitude_index | 同上（`diff(1)`） |
| `topix_cum20` | `topix_return` の rolling(20,15) 和（市場モメンタム） | topix_return | 過去方向 rolling |
| `topix_vol60` | `topix_return` の rolling(60,45) 標準偏差（市場ボラ） | topix_return | 同上 |

前提と判断の詳細は `REPORT.md` の Assumptions を参照。
