# 配布データ スキーマ（自動生成）

- 生成日時: 2026-10-04 04:02
- 生成スクリプト: `tools/profile_data.py`（ローカル実行・再生成可）
- `input_manifest.json` as_of: 2026-07-31
- ファイル数: 20

このファイルは v0.app に渡す参照資料である。v0 は実データを取得できないため、
**ここに書かれた列名・型・欠損・値の例が唯一の根拠**になる。推測で列名を作らないこと。

## ファイル一覧

| ファイル | 行数 | 列数 | サイズ |
|---|---|---|---|
| `beta_1day_train.parquet` | 809,636 | 1 | 6.7MB |
| `beta_1day_valid.parquet` | 1,227,148 | 1 | 10.5MB |
| `consumer_attitude_index_train.parquet` | 89 | 15 | 21.1KB |
| `consumer_attitude_index_valid.parquet` | 125 | 15 | 24.7KB |
| `ecb_fx_rates_train.parquet` | 5,691 | 9 | 324.0KB |
| `ecb_fx_rates_valid.parquet` | 7,938 | 9 | 438.6KB |
| `fins_statements_train.parquet` | 15,926 | 106 | 2.0MB |
| `fins_statements_valid.parquet` | 22,348 | 106 | 3.1MB |
| `listed_info_train.parquet` | 5,274,692 | 11 | 152.7MB |
| `listed_info_valid.parquet` | 10,415,399 | 11 | 304.3MB |
| `prices_daily_quotes_train.parquet` | 809,636 | 40 | 64.5MB |
| `prices_daily_quotes_valid.parquet` | 1,227,148 | 40 | 103.5MB |
| `raw_return_1day_train.parquet` | 809,636 | 1 | 5.0MB |
| `raw_return_1day_valid.parquet` | 1,227,148 | 1 | 9.2MB |
| `raw_target_1day_train.parquet` | 809,636 | 1 | 5.0MB |
| `raw_target_1day_valid.parquet` | 1,227,148 | 1 | 9.2MB |
| `target_1day_train.parquet` | 809,636 | 1 | 6.7MB |
| `target_1day_valid.parquet` | 1,227,148 | 1 | 10.5MB |
| `topix_return_1day_train.parquet` | 1,814 | 1 | 34.2KB |
| `topix_return_1day_valid.parquet` | 2,523 | 1 | 47.6KB |

## 派生系列の検算（実データで確認した事実）

- `raw_target[t] == raw_return[t+2]`: 一致（最大絶対差 0.000e+00, 比較可能 1,226,150 行）
- `target[t] == raw_target[t] - beta[t+2] * topix_return[t+2]`: 一致（最大絶対差 0.000e+00, 比較可能 1,223,593 行）

| ファイル | 行数 |
|---|---|
| target_1day_valid | 1,227,148 |
| raw_return_1day_valid | 1,227,148 |
| beta_1day_valid | 1,227,148 |
- `raw_return_1day_valid` に無い (Date, Code) 行（target 基準）: 0
- `beta_1day_valid` に無い (Date, Code) 行（target 基準）: 0

→ 特徴量とラベルは **index の intersection** で揃えること（sample02 が実例）。
- valid 期間の銘柄数（target 基準）: 498（ユニコード文字列。例 `13320`）
- 1日あたり行数: 中央値 485 / 最小 473 / 最大 498

### `beta_1day_train.parquet`

- 行数: 809,636 / サイズ: 6.7MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.7% | 804124 | 1.0263934999144984 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2008-11-04 13320  1.026393
           14140  0.698478
           16050  1.280980
```

</details>

### `beta_1day_valid.parquet`

- 行数: 1,227,148 / サイズ: 10.5MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.2% | 1224535 | 0.8041474566175564 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2016-04-01 13320  0.804147
           13330  0.570363
           14140  0.836941
```

</details>

### `consumer_attitude_index_train.parquet`

- 行数: 89 / サイズ: 21.1KB / 列数: 15
- 索引: `['Date']`
- Date 範囲: 2008-11-12 23:59:59+09:00 .. 2016-03-08 23:59:59+09:00（89 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `ObservationDate` | datetime64[ns] | 0.0% | 89 | 2008-10-01 00:00:00 |
| `ConsumerAttitudeIndex` | float64 | 0.0% | 61 | 29.4 |
| `PublishedAtJst` | str | 0.0% | 89 | 2008-11-12T23:59:59+09:00 |
| `Unit` | str | 0.0% | 1 | DI |
| `SeasonalAdjustment` | str | 0.0% | 1 | original |
| `HouseholdScope` | str | 0.0% | 1 | two_or_more_person_households |
| `MethodRegime` | str | 0.0% | 2 | direct_visit |
| `PublicationDateKind` | str | 0.0% | 1 | official_estat_date_conservative_eod |
| `ValueVintageKind` | str | 0.0% | 1 | as_published |
| `DatasetSplit` | str | 0.0% | 1 | train |
| `StatInfId` | str | 0.0% | 89 | 000001720437 |
| `SourceUrl` | str | 0.0% | 89 | https://www.e-stat.go.jp/stat-search/file-downlo |
| `SourceFile` | str | 0.0% | 89 | 000001720437_0.xlsx |
| `ContentSha256` | str | 0.0% | 89 | fb02389263df44773e62ceccf4c2888ddfb083ac5b9ada8b |
| `RevisionId` | str | 0.0% | 89 | d2bd4622d53ed3622dc3e79a143192577edc5c0f541e23c7 |

<details><summary>先頭3行の値</summary>

```text
                          ObservationDate  ConsumerAttitudeIndex             PublishedAtJst Unit SeasonalAdjustment                 HouseholdScope  MethodRegime                   PublicationDateKind ValueVintageKind DatasetSplit     StatInfId                                                                             SourceUrl           SourceFile                                                     ContentSha256                                                        RevisionId
Date                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
2008-11-12 23:59:59+09:00      2008-10-01                   29.4  2008-11-12T23:59:59+09:00   DI           original  two_or_more_person_households  direct_visit  official_estat_date_conservative_eod     as_published        train  000001720437  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000001720437&fileKind=0  000001720437_0.xlsx  fb02389263df44773e62ceccf4c2888ddfb083ac5b9ada8baefc7b681cc0b7af  d2bd4622d53ed3622dc3e79a143192577edc5c0f541e23c7c480a1a49e4ee9bb
2008-12-12 23:59:59+09:00      2008-11-01                   28.4  2008-12-12T23:59:59+09:00   DI           original  two_or_more_person_households  direct_visit  official_estat_date_conservative_eod     as_published        train  000002032356  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000002032356&fileKind=0  000002032356_0.xlsx  25b9d9e103ad1537eb484146ea082d197cc77beb86bae16400df6d5d2518dcd4  adb86fa7dae1ddc7bd8175bb119ab70c279a8311f519f5a73ad1e0dde0dd5dab
2009-01-20 23:59:59+09:00      2008-12-01                   26.2  2009-01-20T23:59:59+09:00   DI           original  two_or_more_person_households  direct_visit  official_estat_date_conservative_eod     as_published        train  000002048837  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000002048837&fileKind=0  000002048837_0.xlsx  fe5f70ea11996f834a11d279e5fa7f33cd519bc699f109817dc65d1a6181e206  e109d7995aa56864ae85b2bb05eefc21bb9ea4988970e3aa588f457071941718
```

</details>

### `consumer_attitude_index_valid.parquet`

- 行数: 125 / サイズ: 24.7KB / 列数: 15
- 索引: `['Date']`
- Date 範囲: 2016-04-08 23:59:59+09:00 .. 2026-07-30 23:59:59+09:00（125 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `ObservationDate` | datetime64[ns] | 0.0% | 125 | 2016-03-01 00:00:00 |
| `ConsumerAttitudeIndex` | float64 | 0.0% | 88 | 41.6 |
| `PublishedAtJst` | str | 0.0% | 125 | 2016-04-08T23:59:59+09:00 |
| `Unit` | str | 0.0% | 1 | DI |
| `SeasonalAdjustment` | str | 0.0% | 1 | original |
| `HouseholdScope` | str | 0.0% | 1 | two_or_more_person_households |
| `MethodRegime` | str | 0.0% | 2 | mail |
| `PublicationDateKind` | str | 0.0% | 1 | official_estat_date_conservative_eod |
| `ValueVintageKind` | str | 0.0% | 1 | as_published |
| `DatasetSplit` | str | 0.0% | 1 | valid |
| `StatInfId` | str | 0.0% | 125 | 000031402413 |
| `SourceUrl` | str | 0.0% | 125 | https://www.e-stat.go.jp/stat-search/file-downlo |
| `SourceFile` | str | 0.0% | 125 | 000031402413_0.xlsx |
| `ContentSha256` | str | 0.0% | 125 | 50b44f1fb844e602fd6b9b716357984c73c0f54f6d877819 |
| `RevisionId` | str | 0.0% | 125 | 8a741355ab17c8b3618ab942378cb6f0ecb572f973cf39c5 |

<details><summary>先頭3行の値</summary>

```text
                          ObservationDate  ConsumerAttitudeIndex             PublishedAtJst Unit SeasonalAdjustment                 HouseholdScope MethodRegime                   PublicationDateKind ValueVintageKind DatasetSplit     StatInfId                                                                             SourceUrl           SourceFile                                                     ContentSha256                                                        RevisionId
Date                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
2016-04-08 23:59:59+09:00      2016-03-01                   41.6  2016-04-08T23:59:59+09:00   DI           original  two_or_more_person_households         mail  official_estat_date_conservative_eod     as_published        valid  000031402413  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000031402413&fileKind=0  000031402413_0.xlsx  50b44f1fb844e602fd6b9b716357984c73c0f54f6d87781980baad6bd7dfded7  8a741355ab17c8b3618ab942378cb6f0ecb572f973cf39c51fa3107ab6e9e0f9
2016-05-09 23:59:59+09:00      2016-04-01                   40.7  2016-05-09T23:59:59+09:00   DI           original  two_or_more_person_households         mail  official_estat_date_conservative_eod     as_published        valid  000031408604  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000031408604&fileKind=0  000031408604_0.xlsx  82de2a5364f93fbe1db1083e4fd13d6074a4851c62afb912d874009f1c16adba  0378f2e4720bc253e028f7071cd27cd4ca199ebfe477852178ee29f9b20d4331
2016-06-02 23:59:59+09:00      2016-05-01                   41.5  2016-06-02T23:59:59+09:00   DI           original  two_or_more_person_households         mail  official_estat_date_conservative_eod     as_published        valid  000031416170  https://www.e-stat.go.jp/stat-search/file-download?statInfId=000031416170&fileKind=0  000031416170_0.xlsx  b576867472ccec381e64e0c10d011d933268411d0d8785b8cc1d22258f72463a  33b3173320644898701bb192141533a16d603be5b476eb33b125ec580338f48a
```

</details>

### `ecb_fx_rates_train.parquet`

- 行数: 5,691 / サイズ: 324.0KB / 列数: 9
- 索引: `['Date', 'Pair']`
- Date 範囲: 2008-11-01 00:00:00+09:00 .. 2016-03-31 23:00:00+09:00（1897 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `ObservationDate` | datetime64[ns] | 0.0% | 1897 | 2008-10-31 00:00:00 |
| `Rate` | float64 | 0.0% | 4806 | 124.97 |
| `PublishedAtFrankfurt` | str | 0.0% | 1897 | 2008-10-31T16:00:00+01:00 |
| `IsDerived` | bool | 0.0% | 2 | False |
| `SourceSeriesKey` | str | 0.0% | 3 | D.JPY.EUR.SP00.A |
| `SourceValues` | str | 0.0% | 4806 | EUR/JPY=124.97 |
| `SourceUrl` | str | 0.0% | 1 | https://data.ecb.europa.eu/data/datasets/EXR |
| `DatasetSplit` | str | 0.0% | 1 | train |
| `RevisionId` | str | 0.0% | 5691 | 83d1eb73b76facbddca16a904b198edc4295066ddad942ff |

<details><summary>先頭3行の値</summary>

```text
                                  ObservationDate        Rate       PublishedAtFrankfurt  IsDerived                    SourceSeriesKey                   SourceValues                                     SourceUrl DatasetSplit                                                        RevisionId
Date                      Pair                                                                                                                                                                                                                                                                    
2008-11-01 00:00:00+09:00 EUR/JPY      2008-10-31  124.970000  2008-10-31T16:00:00+01:00      False                   D.JPY.EUR.SP00.A                 EUR/JPY=124.97  https://data.ecb.europa.eu/data/datasets/EXR        train  83d1eb73b76facbddca16a904b198edc4295066ddad942ffaa491400a7798218
                          EUR/USD      2008-10-31    1.275700  2008-10-31T16:00:00+01:00      False                   D.USD.EUR.SP00.A                 EUR/USD=1.2757  https://data.ecb.europa.eu/data/datasets/EXR        train  cc7331a5ce699faf633ca864af8bf514f20173707ba9d992316844e4dfc5b3d2
                          USD/JPY      2008-10-31   97.961903  2008-10-31T16:00:00+01:00       True  D.JPY.EUR.SP00.A/D.USD.EUR.SP00.A  EUR/JPY=124.97;EUR/USD=1.2757  https://data.ecb.europa.eu/data/datasets/EXR        train  9da8c680ca83f23510c9c4f9f32630f2b39a667e0da0bc97ba445680195c9bc4
```

</details>

### `ecb_fx_rates_valid.parquet`

- 行数: 7,938 / サイズ: 438.6KB / 列数: 9
- 索引: `['Date', 'Pair']`
- Date 範囲: 2016-04-01 23:00:00+09:00 .. 2026-07-31 23:00:00+09:00（2646 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `ObservationDate` | datetime64[ns] | 0.0% | 2646 | 2016-04-01 00:00:00 |
| `Rate` | float64 | 0.0% | 6129 | 128.07 |
| `PublishedAtFrankfurt` | str | 0.0% | 2646 | 2016-04-01T16:00:00+02:00 |
| `IsDerived` | bool | 0.0% | 2 | False |
| `SourceSeriesKey` | str | 0.0% | 3 | D.JPY.EUR.SP00.A |
| `SourceValues` | str | 0.0% | 6131 | EUR/JPY=128.07 |
| `SourceUrl` | str | 0.0% | 1 | https://data.ecb.europa.eu/data/datasets/EXR |
| `DatasetSplit` | str | 0.0% | 1 | valid |
| `RevisionId` | str | 0.0% | 7938 | dcf988bf8a10a30217090dfd305e4e3a12ae8f2558095aef |

<details><summary>先頭3行の値</summary>

```text
                                  ObservationDate        Rate       PublishedAtFrankfurt  IsDerived                    SourceSeriesKey                   SourceValues                                     SourceUrl DatasetSplit                                                        RevisionId
Date                      Pair                                                                                                                                                                                                                                                                    
2016-04-01 23:00:00+09:00 EUR/JPY      2016-04-01  128.070000  2016-04-01T16:00:00+02:00      False                   D.JPY.EUR.SP00.A                 EUR/JPY=128.07  https://data.ecb.europa.eu/data/datasets/EXR        valid  dcf988bf8a10a30217090dfd305e4e3a12ae8f2558095aef1f573ccb7c38757a
                          EUR/USD      2016-04-01    1.143200  2016-04-01T16:00:00+02:00      False                   D.USD.EUR.SP00.A                 EUR/USD=1.1432  https://data.ecb.europa.eu/data/datasets/EXR        valid  925191cd10c108d73567096a5be804d6d30eed744fa4b8dcc88635ded7d85569
                          USD/JPY      2016-04-01  112.027642  2016-04-01T16:00:00+02:00       True  D.JPY.EUR.SP00.A/D.USD.EUR.SP00.A  EUR/JPY=128.07;EUR/USD=1.1432  https://data.ecb.europa.eu/data/datasets/EXR        valid  31bf4d04d8d61aa025ce431f7977e00532153e352be8d8c83317c107fae7c504
```

</details>

### `fins_statements_train.parquet`

- 行数: 15,926 / サイズ: 2.0MB / 列数: 106
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00+09:00 .. 2016-03-31 00:00:00+09:00（1451 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `DisclosureNumber` | int64 | 0.0% | 15926 | 20081104099402 |
| `TypeOfDocument` | category | 0.0% | 19 | EarnForecastRevision |
| `TypeOfCurrentPeriod` | category | 0.0% | 5 | FY |
| `CurrentPeriodStartDate` | datetime64[ns] | 0.0% | 118 | 2008-04-01 00:00:00 |
| `CurrentPeriodEndDate` | datetime64[ns] | 0.0% | 179 | 2009-03-31 00:00:00 |
| `CurrentFiscalYearStartDate` | datetime64[ns] | 0.0% | 118 | 2008-04-01 00:00:00 |
| `CurrentFiscalYearEndDate` | datetime64[ns] | 0.0% | 119 | 2009-03-31 00:00:00 |
| `NextFiscalYearStartDate` | datetime64[ns] | 79.9% | 107 |  |
| `NextFiscalYearEndDate` | datetime64[ns] | 79.9% | 108 |  |
| `NetSales` | float64 | 27.2% | 11301 | 105056000000.0 |
| `OperatingProfit` | float64 | 30.7% | 9783 | 18464000000.0 |
| `OrdinaryProfit` | float64 | 33.0% | 9528 | 18969000000.0 |
| `Profit` | float64 | 27.6% | 9740 | 11835000000.0 |
| `EarningsPerShare` | float64 | 27.6% | 8583 | 35.32 |
| `DilutedEarningsPerShare` | float64 | 68.4% | 4260 |  |
| `TotalAssets` | float64 | 27.2% | 11373 | 415786000000.0 |
| `Equity` | float64 | 27.2% | 11316 | 341692000000.0 |
| `EquityToAssetRatio` | float64 | 27.3% | 941 | 0.821 |
| `BookValuePerShare` | float64 | 53.8% | 7129 | 1018.96 |
| `CashFlowsFromOperatingActivities` | float64 | 55.0% | 6755 |  |
| `CashFlowsFromInvestingActivities` | float64 | 55.0% | 6622 |  |
| `CashFlowsFromFinancingActivities` | float64 | 55.0% | 6509 |  |
| `CashAndEquivalents` | float64 | 55.0% | 6826 |  |
| `ResultDividendPerShare1stQuarter` | float64 | 99.4% | 16 |  |
| `ResultDividendPerShare2ndQuarter` | float64 | 45.9% | 188 | 14.0 |
| `ResultDividendPerShare3rdQuarter` | float64 | 99.7% | 15 |  |
| `ResultDividendPerShareFiscalYearEnd` | float64 | 80.0% | 236 |  |
| `ResultDividendPerShareAnnual` | float64 | 80.2% | 262 |  |
| `DistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `ResultTotalDividendPaidAnnual` | float64 | 80.5% | 2389 |  |
| `ResultPayoutRatioAnnual` | float64 | 82.8% | 805 |  |
| `ForecastDividendPerShare1stQuarter` | float64 | 99.9% | 2 |  |
| `ForecastDividendPerShare2ndQuarter` | float64 | 83.1% | 165 | 88.0 |
| `ForecastDividendPerShare3rdQuarter` | float64 | 99.7% | 15 |  |
| `ForecastDividendPerShareFiscalYearEnd` | float64 | 46.2% | 276 | 88.0 |
| `ForecastDividendPerShareAnnual` | float64 | 46.8% | 320 | 176.0 |
| `ForecastDistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `ForecastTotalDividendPaidAnnual` | float64 | 100.0% | 0 |  |
| `ForecastPayoutRatioAnnual` | float64 | 100.0% | 0 |  |
| `NextYearForecastDividendPerShare1stQuarter` | float64 | 99.8% | 15 |  |
| `NextYearForecastDividendPerShare2ndQuarter` | float64 | 83.3% | 161 |  |
| `NextYearForecastDividendPerShare3rdQuarter` | float64 | 99.9% | 13 |  |
| `NextYearForecastDividendPerShareFiscalYearEnd` | float64 | 82.7% | 201 |  |
| `NextYearForecastDividendPerShareAnnual` | float64 | 82.6% | 235 |  |
| `NextYearForecastDistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `NextYearForecastPayoutRatioAnnual` | float64 | 84.2% | 659 |  |
| `ForecastNetSales2ndQuarter` | float64 | 76.1% | 2011 | 178900000000.0 |
| `ForecastOperatingProfit2ndQuarter` | float64 | 76.9% | 1054 | -860000000.0 |
| `ForecastOrdinaryProfit2ndQuarter` | float64 | 76.9% | 1080 | 260000000.0 |
| `ForecastProfit2ndQuarter` | float64 | 76.1% | 1010 | -440000000.0 |
| `ForecastEarningsPerShare2ndQuarter` | float64 | 76.7% | 3104 | -1.59 |
| `NextYearForecastNetSales2ndQuarter` | float64 | 84.3% | 1269 |  |
| `NextYearForecastOperatingProfit2ndQuarter` | float64 | 84.9% | 623 |  |
| `NextYearForecastOrdinaryProfit2ndQuarter` | float64 | 84.9% | 666 |  |
| `NextYearForecastProfit2ndQuarter` | float64 | 84.4% | 564 |  |
| `NextYearForecastEarningsPerShare2ndQuarter` | float64 | 84.4% | 2122 |  |
| `ForecastNetSales` | float64 | 33.1% | 2922 | 1665000000000.0 |
| `ForecastOperatingProfit` | float64 | 36.1% | 1400 | 70000000000.0 |
| `ForecastOrdinaryProfit` | float64 | 37.4% | 1418 | 60000000000.0 |
| `ForecastProfit` | float64 | 32.9% | 1355 | 26500000000.0 |
| `ForecastEarningsPerShare` | float64 | 34.5% | 6318 | 45.75 |
| `NextYearForecastNetSales` | float64 | 81.8% | 1348 |  |
| `NextYearForecastOperatingProfit` | float64 | 82.7% | 734 |  |
| `NextYearForecastOrdinaryProfit` | float64 | 83.3% | 766 |  |
| `NextYearForecastProfit` | float64 | 81.8% | 676 |  |
| `NextYearForecastEarningsPerShare` | float64 | 81.9% | 2577 |  |
| `MaterialChangesInSubsidiaries` | bool | 0.0% | 2 | False |
| `SignificantChangesInTheScopeOfConsolidation` | bool | 0.0% | 1 | False |
| `ChangesBasedOnRevisionsOfAccountingStandard` | bool | 0.0% | 2 | False |
| `ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard` | bool | 0.0% | 2 | False |
| `ChangesInAccountingEstimates` | bool | 0.0% | 2 | False |
| `RetrospectiveRestatement` | bool | 0.0% | 2 | False |
| `NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock` | float64 | 27.3% | 1732 | 351136165.0 |
| `NumberOfTreasuryStockAtTheEndOfFiscalYear` | float64 | 28.8% | 10277 | 16120220.0 |
| `AverageNumberOfShares` | float64 | 33.2% | 10140 | 335073058.0 |
| `NonConsolidatedNetSales` | float64 | 94.3% | 859 |  |
| `NonConsolidatedOperatingProfit` | float64 | 94.5% | 827 |  |
| `NonConsolidatedOrdinaryProfit` | float64 | 94.3% | 854 |  |
| `NonConsolidatedProfit` | float64 | 94.3% | 860 |  |
| `NonConsolidatedEarningsPerShare` | float64 | 94.3% | 843 |  |
| `NonConsolidatedTotalAssets` | float64 | 94.3% | 862 |  |
| `NonConsolidatedEquity` | float64 | 94.3% | 863 |  |
| `NonConsolidatedEquityToAssetRatio` | float64 | 94.3% | 539 |  |
| `NonConsolidatedBookValuePerShare` | float64 | 94.5% | 838 |  |
| `ForecastNonConsolidatedNetSales2ndQuarter` | float64 | 96.8% | 454 | 170000000000.0 |
| `ForecastNonConsolidatedOperatingProfit2ndQuarter` | float64 | 97.3% | 311 | -1890000000.0 |
| `ForecastNonConsolidatedOrdinaryProfit2ndQuarter` | float64 | 96.8% | 346 | -650000000.0 |
| `ForecastNonConsolidatedProfit2ndQuarter` | float64 | 96.8% | 325 | -850000000.0 |
| `ForecastNonConsolidatedEarningsPerShare2ndQuarter` | float64 | 97.0% | 461 | -3.06 |
| `NextYearForecastNonConsolidatedNetSales2ndQuarter` | float64 | 94.8% | 579 |  |
| `NextYearForecastNonConsolidatedOperatingProfit2ndQuarter` | float64 | 96.0% | 307 |  |
| `NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter` | float64 | 94.8% | 351 |  |
| `NextYearForecastNonConsolidatedProfit2ndQuarter` | float64 | 94.8% | 304 |  |
| `NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter` | float64 | 94.8% | 738 |  |
| `ForecastNonConsolidatedNetSales` | float64 | 79.8% | 2836 | 2010000000.0 |
| `ForecastNonConsolidatedOperatingProfit` | float64 | 81.0% | 2388 | 669000000.0 |
| `ForecastNonConsolidatedOrdinaryProfit` | float64 | 79.7% | 2496 | 658000000.0 |
| `ForecastNonConsolidatedProfit` | float64 | 79.7% | 2454 | 383000000.0 |
| `ForecastNonConsolidatedEarningsPerShare` | float64 | 80.4% | 2746 | 4497.07 |
| `NextYearForecastNonConsolidatedNetSales` | float64 | 94.2% | 693 |  |
| `NextYearForecastNonConsolidatedOperatingProfit` | float64 | 95.6% | 378 |  |
| `NextYearForecastNonConsolidatedOrdinaryProfit` | float64 | 94.2% | 424 |  |
| `NextYearForecastNonConsolidatedProfit` | float64 | 94.2% | 379 |  |
| `NextYearForecastNonConsolidatedEarningsPerShare` | float64 | 94.2% | 857 |  |
| `FiscalYear` | int64 | 0.0% | 10 | 2008 |
| `Period` | float64 | 0.0% | 34 | 2008.75 |

<details><summary>先頭3行の値</summary>

```text
                                 DisclosureNumber        TypeOfDocument TypeOfCurrentPeriod CurrentPeriodStartDate CurrentPeriodEndDate CurrentFiscalYearStartDate CurrentFiscalYearEndDate NextFiscalYearStartDate NextFiscalYearEndDate  NetSales  OperatingProfit  OrdinaryProfit  Profit  EarningsPerShare  DilutedEarningsPerShare  TotalAssets  Equity  EquityToAssetRatio  BookValuePerShare  CashFlowsFromOperatingActivities  CashFlowsFromInvestingActivities  CashFlowsFromFinancingActivities  CashAndEquivalents  ResultDividendPerShare1stQuarter  ResultDividendPerShare2ndQuarter  ResultDividendPerShare3rdQuarter  ResultDividendPerShareFiscalYearEnd  ResultDividendPerShareAnnual  DistributionsPerUnit(REIT)  ResultTotalDividendPaidAnnual  ResultPayoutRatioAnnual  ForecastDividendPerShare1stQuarter  ForecastDividendPerShare2ndQuarter  ForecastDividendPerShare3rdQuarter  ForecastDividendPerShareFiscalYearEnd  ForecastDividendPerShareAnnual  ForecastDistributionsPerUnit(REIT)  ForecastTotalDividendPaidAnnual  ForecastPayoutRatioAnnual  NextYearForecastDividendPerShare1stQuarter  NextYearForecastDividendPerShare2ndQuarter  NextYearForecastDividendPerShare3rdQuarter  NextYearForecastDividendPerShareFiscalYearEnd  NextYearForecastDividendPerShareAnnual  NextYearForecastDistributionsPerUnit(REIT)  NextYearForecastPayoutRatioAnnual  ForecastNetSales2ndQuarter  ForecastOperatingProfit2ndQuarter  ForecastOrdinaryProfit2ndQuarter  ForecastProfit2ndQuarter  ForecastEarningsPerShare2ndQuarter  NextYearForecastNetSales2ndQuarter  NextYearForecastOperatingProfit2ndQuarter  NextYearForecastOrdinaryProfit2ndQuarter  NextYearForecastProfit2ndQuarter  NextYearForecastEarningsPerShare2ndQuarter  ForecastNetSales  ForecastOperatingProfit  ForecastOrdinaryProfit  ForecastProfit  ForecastEarningsPerShare  NextYearForecastNetSales  NextYearForecastOperatingProfit  NextYearForecastOrdinaryProfit  NextYearForecastProfit  NextYearForecastEarningsPerShare  MaterialChangesInSubsidiaries  SignificantChangesInTheScopeOfConsolidation  ChangesBasedOnRevisionsOfAccountingStandard  ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard  ChangesInAccountingEstimates  RetrospectiveRestatement  NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock  NumberOfTreasuryStockAtTheEndOfFiscalYear  AverageNumberOfShares  NonConsolidatedNetSales  NonConsolidatedOperatingProfit  NonConsolidatedOrdinaryProfit  NonConsolidatedProfit  NonConsolidatedEarningsPerShare  NonConsolidatedTotalAssets  NonConsolidatedEquity  NonConsolidatedEquityToAssetRatio  NonConsolidatedBookValuePerShare  ForecastNonConsolidatedNetSales2ndQuarter  ForecastNonConsolidatedOperatingProfit2ndQuarter  ForecastNonConsolidatedOrdinaryProfit2ndQuarter  ForecastNonConsolidatedProfit2ndQuarter  ForecastNonConsolidatedEarningsPerShare2ndQuarter  NextYearForecastNonConsolidatedNetSales2ndQuarter  NextYearForecastNonConsolidatedOperatingProfit2ndQuarter  NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter  NextYearForecastNonConsolidatedProfit2ndQuarter  NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter  ForecastNonConsolidatedNetSales  ForecastNonConsolidatedOperatingProfit  ForecastNonConsolidatedOrdinaryProfit  ForecastNonConsolidatedProfit  ForecastNonConsolidatedEarningsPerShare  NextYearForecastNonConsolidatedNetSales  NextYearForecastNonConsolidatedOperatingProfit  NextYearForecastNonConsolidatedOrdinaryProfit  NextYearForecastNonConsolidatedProfit  NextYearForecastNonConsolidatedEarningsPerShare  FiscalYear   Period
Date                      Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
2008-11-04 00:00:00+09:00 18200    20081104099402  EarnForecastRevision                  FY             2008-04-01           2009-03-31                 2008-04-01               2009-03-31                     NaT                   NaT       NaN              NaN             NaN     NaN               NaN                      NaN          NaN     NaN                 NaN                NaN                               NaN                               NaN                               NaN                 NaN                               NaN                               NaN                               NaN                                  NaN                           NaN                         NaN                            NaN                      NaN                                 NaN                                 NaN                                 NaN                                    NaN                             NaN                                 NaN                              NaN                        NaN                                         NaN                                         NaN                                         NaN                                            NaN                                     NaN                                         NaN                                NaN                1.789000e+11                       -860000000.0                       260000000.0              -440000000.0                               -1.59                                 NaN                                        NaN                                       NaN                               NaN                                         NaN               NaN                      NaN                     NaN             NaN                       NaN                       NaN                              NaN                             NaN                     NaN                               NaN                          False                                        False                                        False                                                     False                         False                     False                                                                           NaN                                        NaN                    NaN                      NaN                             NaN                            NaN                    NaN                              NaN                         NaN                    NaN                                NaN                               NaN                               1.700000e+11                                     -1.890000e+09                                     -650000000.0                             -850000000.0                                              -3.06                                                NaN                                                       NaN                                                      NaN                                              NaN                                                        NaN                              NaN                                     NaN                                    NaN                            NaN                                      NaN                                      NaN                                             NaN                                            NaN                                    NaN                                              NaN        2008  2008.75
                          19250    20081031097034  EarnForecastRevision                  FY             2008-04-01           2009-03-31                 2008-04-01               2009-03-31                     NaT                   NaT       NaN              NaN             NaN     NaN               NaN                      NaN          NaN     NaN                 NaN                NaN                               NaN                               NaN                               NaN                 NaN                               NaN                               NaN                               NaN                                  NaN                           NaN                         NaN                            NaN                      NaN                                 NaN                                 NaN                                 NaN                                    NaN                             NaN                                 NaN                              NaN                        NaN                                         NaN                                         NaN                                         NaN                                            NaN                                     NaN                                         NaN                                NaN                         NaN                                NaN                               NaN                       NaN                                 NaN                                 NaN                                        NaN                                       NaN                               NaN                                         NaN      1.665000e+12             7.000000e+10            6.000000e+10    2.650000e+10                     45.75                       NaN                              NaN                             NaN                     NaN                               NaN                          False                                        False                                        False                                                     False                         False                     False                                                                           NaN                                        NaN                    NaN                      NaN                             NaN                            NaN                    NaN                              NaN                         NaN                    NaN                                NaN                               NaN                                        NaN                                               NaN                                              NaN                                      NaN                                                NaN                                                NaN                                                       NaN                                                      NaN                                              NaN                                                        NaN                              NaN                                     NaN                                    NaN                            NaN                                      NaN                                      NaN                                             NaN                                            NaN                                    NaN                                              NaN        2008  2008.75
                          37690    20081027086562  EarnForecastRevision                  FY             2007-10-01           2008-09-30                 2007-10-01               2008-09-30                     NaT                   NaT       NaN              NaN             NaN     NaN               NaN                      NaN          NaN     NaN                 NaN                NaN                               NaN                               NaN                               NaN                 NaN                               NaN                               NaN                               NaN                                  NaN                           NaN                         NaN                            NaN                      NaN                                 NaN                                 NaN                                 NaN                                    NaN                             NaN                                 NaN                              NaN                        NaN                                         NaN                                         NaN                                         NaN                                            NaN                                     NaN                                         NaN                                NaN                         NaN                                NaN                               NaN                       NaN                                 NaN                                 NaN                                        NaN                                       NaN                               NaN                                         NaN      2.207000e+09             7.760000e+08            7.560000e+08    4.400000e+08                   5155.05                       NaN                              NaN                             NaN                     NaN                               NaN                          False                                        False                                        False                                                     False                         False                     False                                                                           NaN                                        NaN                    NaN                      NaN                             NaN                            NaN                    NaN                              NaN                         NaN                    NaN                                NaN                               NaN                                        NaN                                               NaN                                              NaN                                      NaN                                                NaN                                                NaN                                                       NaN                                                      NaN                                              NaN                                                        NaN                     2.010000e+09                             669000000.0                            658000000.0                    383000000.0                                  4497.07                                      NaN                                             NaN                                            NaN                                    NaN                                              NaN        2007  2007.75
```

</details>

### `fins_statements_valid.parquet`

- 行数: 22,348 / サイズ: 3.1MB / 列数: 106
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00+09:00 .. 2026-07-31 00:00:00+09:00（1903 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `DisclosureNumber` | int64 | 0.0% | 22348 | 20160331448091 |
| `TypeOfDocument` | category | 0.0% | 22 | EarnForecastRevision |
| `TypeOfCurrentPeriod` | category | 0.0% | 4 | FY |
| `CurrentPeriodStartDate` | datetime64[ns] | 0.0% | 176 | 2015-04-01 00:00:00 |
| `CurrentPeriodEndDate` | datetime64[ns] | 0.0% | 251 | 2016-03-31 00:00:00 |
| `CurrentFiscalYearStartDate` | datetime64[ns] | 0.0% | 176 | 2015-04-01 00:00:00 |
| `CurrentFiscalYearEndDate` | datetime64[ns] | 0.0% | 174 | 2016-03-31 00:00:00 |
| `NextFiscalYearStartDate` | datetime64[ns] | 75.9% | 162 | 2016-02-21 00:00:00 |
| `NextFiscalYearEndDate` | datetime64[ns] | 75.9% | 163 | 2017-02-20 00:00:00 |
| `NetSales` | float64 | 11.6% | 19212 | 132140000000.0 |
| `OperatingProfit` | float64 | 19.1% | 16026 | 4350000000.0 |
| `OrdinaryProfit` | float64 | 35.8% | 12936 | 4581000000.0 |
| `Profit` | float64 | 11.5% | 17030 | 2391000000.0 |
| `EarningsPerShare` | float64 | 11.6% | 14754 | 15.76 |
| `DilutedEarningsPerShare` | float64 | 53.0% | 8559 |  |
| `TotalAssets` | float64 | 11.5% | 19369 | 363929000000.0 |
| `Equity` | float64 | 11.5% | 19281 | 243532000000.0 |
| `EquityToAssetRatio` | float64 | 11.7% | 950 | 0.581 |
| `BookValuePerShare` | float64 | 62.2% | 8179 | 8293.63 |
| `CashFlowsFromOperatingActivities` | float64 | 49.2% | 10673 | 5005000000.0 |
| `CashFlowsFromInvestingActivities` | float64 | 49.2% | 10475 | -9100000000.0 |
| `CashFlowsFromFinancingActivities` | float64 | 49.2% | 10473 | -224000000.0 |
| `CashAndEquivalents` | float64 | 46.8% | 11410 | 30384000000.0 |
| `ResultDividendPerShare1stQuarter` | float64 | 99.1% | 36 |  |
| `ResultDividendPerShare2ndQuarter` | float64 | 36.6% | 313 | 95.0 |
| `ResultDividendPerShare3rdQuarter` | float64 | 99.5% | 36 |  |
| `ResultDividendPerShareFiscalYearEnd` | float64 | 75.9% | 399 | 100.0 |
| `ResultDividendPerShareAnnual` | float64 | 76.7% | 452 | 195.0 |
| `DistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `ResultTotalDividendPaidAnnual` | float64 | 76.6% | 4402 | 7167000000.0 |
| `ResultPayoutRatioAnnual` | float64 | 77.5% | 961 | 0.29 |
| `ForecastDividendPerShare1stQuarter` | float64 | 99.9% | 2 |  |
| `ForecastDividendPerShare2ndQuarter` | float64 | 78.9% | 284 | 15.0 |
| `ForecastDividendPerShare3rdQuarter` | float64 | 99.7% | 32 |  |
| `ForecastDividendPerShareFiscalYearEnd` | float64 | 38.1% | 445 | 15.0 |
| `ForecastDividendPerShareAnnual` | float64 | 39.9% | 508 | 30.0 |
| `ForecastDistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `ForecastTotalDividendPaidAnnual` | float64 | 100.0% | 0 |  |
| `ForecastPayoutRatioAnnual` | float64 | 100.0% | 0 |  |
| `NextYearForecastDividendPerShare1stQuarter` | float64 | 99.9% | 20 |  |
| `NextYearForecastDividendPerShare2ndQuarter` | float64 | 79.7% | 273 | 97.5 |
| `NextYearForecastDividendPerShare3rdQuarter` | float64 | 99.9% | 20 |  |
| `NextYearForecastDividendPerShareFiscalYearEnd` | float64 | 79.4% | 316 | 97.5 |
| `NextYearForecastDividendPerShareAnnual` | float64 | 79.2% | 359 | 195.0 |
| `NextYearForecastDistributionsPerUnit(REIT)` | float64 | 100.0% | 0 |  |
| `NextYearForecastPayoutRatioAnnual` | float64 | 80.8% | 792 | 0.234 |
| `ForecastNetSales2ndQuarter` | float64 | 85.8% | 1735 | 283000000000.0 |
| `ForecastOperatingProfit2ndQuarter` | float64 | 86.3% | 888 | 11700000000.0 |
| `ForecastOrdinaryProfit2ndQuarter` | float64 | 87.7% | 863 | 12100000000.0 |
| `ForecastProfit2ndQuarter` | float64 | 85.4% | 886 | 5700000000.0 |
| `ForecastEarningsPerShare2ndQuarter` | float64 | 85.4% | 2892 | 37.55 |
| `NextYearForecastNetSales2ndQuarter` | float64 | 88.6% | 1431 | 280500000000.0 |
| `NextYearForecastOperatingProfit2ndQuarter` | float64 | 88.9% | 723 | 22600000000.0 |
| `NextYearForecastOrdinaryProfit2ndQuarter` | float64 | 90.0% | 706 | 22980000000.0 |
| `NextYearForecastProfit2ndQuarter` | float64 | 88.1% | 685 | 14800000000.0 |
| `NextYearForecastEarningsPerShare2ndQuarter` | float64 | 88.2% | 2347 | 402.66 |
| `ForecastNetSales` | float64 | 35.9% | 3421 | 1250000000000.0 |
| `ForecastOperatingProfit` | float64 | 39.4% | 1700 | 7000000000.0 |
| `ForecastOrdinaryProfit` | float64 | 50.5% | 1543 | 1000000000.0 |
| `ForecastProfit` | float64 | 32.4% | 1668 | -50000000000.0 |
| `ForecastEarningsPerShare` | float64 | 33.0% | 9527 | -53.35 |
| `NextYearForecastNetSales` | float64 | 79.9% | 1922 | 574200000000.0 |
| `NextYearForecastOperatingProfit` | float64 | 81.0% | 1043 | 46200000000.0 |
| `NextYearForecastOrdinaryProfit` | float64 | 84.8% | 987 | 46880000000.0 |
| `NextYearForecastProfit` | float64 | 78.8% | 1017 | 30600000000.0 |
| `NextYearForecastEarningsPerShare` | float64 | 78.9% | 4283 | 832.51 |
| `MaterialChangesInSubsidiaries` | bool | 0.0% | 2 | False |
| `SignificantChangesInTheScopeOfConsolidation` | bool | 0.0% | 2 | False |
| `ChangesBasedOnRevisionsOfAccountingStandard` | bool | 0.0% | 2 | False |
| `ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard` | bool | 0.0% | 2 | False |
| `ChangesInAccountingEstimates` | bool | 0.0% | 2 | False |
| `RetrospectiveRestatement` | bool | 0.0% | 2 | False |
| `NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock` | float64 | 11.7% | 3368 | 153000000.0 |
| `NumberOfTreasuryStockAtTheEndOfFiscalYear` | float64 | 12.2% | 17667 | 1232736.0 |
| `AverageNumberOfShares` | float64 | 11.7% | 19078 | 151767371.0 |
| `NonConsolidatedNetSales` | float64 | 80.0% | 4303 | 540216000000.0 |
| `NonConsolidatedOperatingProfit` | float64 | 80.8% | 4036 | 40466000000.0 |
| `NonConsolidatedOrdinaryProfit` | float64 | 80.0% | 4234 | 41391000000.0 |
| `NonConsolidatedProfit` | float64 | 80.0% | 4215 | 24796000000.0 |
| `NonConsolidatedEarningsPerShare` | float64 | 80.0% | 4097 | 674.57 |
| `NonConsolidatedTotalAssets` | float64 | 80.1% | 4320 | 351748000000.0 |
| `NonConsolidatedEquity` | float64 | 80.1% | 4309 | 306382000000.0 |
| `NonConsolidatedEquityToAssetRatio` | float64 | 80.1% | 928 | 0.871 |
| `NonConsolidatedBookValuePerShare` | float64 | 80.8% | 4144 | 8355.18 |
| `ForecastNonConsolidatedNetSales2ndQuarter` | float64 | 99.4% | 118 |  |
| `ForecastNonConsolidatedOperatingProfit2ndQuarter` | float64 | 99.5% | 93 |  |
| `ForecastNonConsolidatedOrdinaryProfit2ndQuarter` | float64 | 99.4% | 120 |  |
| `ForecastNonConsolidatedProfit2ndQuarter` | float64 | 99.4% | 115 |  |
| `ForecastNonConsolidatedEarningsPerShare2ndQuarter` | float64 | 99.4% | 131 |  |
| `NextYearForecastNonConsolidatedNetSales2ndQuarter` | float64 | 98.1% | 334 | 277000000000.0 |
| `NextYearForecastNonConsolidatedOperatingProfit2ndQuarter` | float64 | 98.9% | 169 |  |
| `NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter` | float64 | 98.0% | 268 | 23200000000.0 |
| `NextYearForecastNonConsolidatedProfit2ndQuarter` | float64 | 98.0% | 241 | 15100000000.0 |
| `NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter` | float64 | 98.0% | 429 | 410.81 |
| `ForecastNonConsolidatedNetSales` | float64 | 97.8% | 433 |  |
| `ForecastNonConsolidatedOperatingProfit` | float64 | 98.3% | 288 |  |
| `ForecastNonConsolidatedOrdinaryProfit` | float64 | 97.6% | 375 |  |
| `ForecastNonConsolidatedProfit` | float64 | 97.6% | 380 |  |
| `ForecastNonConsolidatedEarningsPerShare` | float64 | 97.7% | 512 |  |
| `NextYearForecastNonConsolidatedNetSales` | float64 | 97.0% | 550 | 566500000000.0 |
| `NextYearForecastNonConsolidatedOperatingProfit` | float64 | 98.0% | 284 |  |
| `NextYearForecastNonConsolidatedOrdinaryProfit` | float64 | 96.8% | 400 | 47200000000.0 |
| `NextYearForecastNonConsolidatedProfit` | float64 | 96.8% | 356 | 31000000000.0 |
| `NextYearForecastNonConsolidatedEarningsPerShare` | float64 | 96.9% | 669 | 843.36 |
| `FiscalYear` | int64 | 0.0% | 13 | 2015 |
| `Period` | float64 | 0.0% | 48 | 2015.75 |

<details><summary>先頭3行の値</summary>

```text
                                 DisclosureNumber                         TypeOfDocument TypeOfCurrentPeriod CurrentPeriodStartDate CurrentPeriodEndDate CurrentFiscalYearStartDate CurrentFiscalYearEndDate NextFiscalYearStartDate NextFiscalYearEndDate      NetSales  OperatingProfit  OrdinaryProfit        Profit  EarningsPerShare  DilutedEarningsPerShare   TotalAssets        Equity  EquityToAssetRatio  BookValuePerShare  CashFlowsFromOperatingActivities  CashFlowsFromInvestingActivities  CashFlowsFromFinancingActivities  CashAndEquivalents  ResultDividendPerShare1stQuarter  ResultDividendPerShare2ndQuarter  ResultDividendPerShare3rdQuarter  ResultDividendPerShareFiscalYearEnd  ResultDividendPerShareAnnual  DistributionsPerUnit(REIT)  ResultTotalDividendPaidAnnual  ResultPayoutRatioAnnual  ForecastDividendPerShare1stQuarter  ForecastDividendPerShare2ndQuarter  ForecastDividendPerShare3rdQuarter  ForecastDividendPerShareFiscalYearEnd  ForecastDividendPerShareAnnual  ForecastDistributionsPerUnit(REIT)  ForecastTotalDividendPaidAnnual  ForecastPayoutRatioAnnual  NextYearForecastDividendPerShare1stQuarter  NextYearForecastDividendPerShare2ndQuarter  NextYearForecastDividendPerShare3rdQuarter  NextYearForecastDividendPerShareFiscalYearEnd  NextYearForecastDividendPerShareAnnual  NextYearForecastDistributionsPerUnit(REIT)  NextYearForecastPayoutRatioAnnual  ForecastNetSales2ndQuarter  ForecastOperatingProfit2ndQuarter  ForecastOrdinaryProfit2ndQuarter  ForecastProfit2ndQuarter  ForecastEarningsPerShare2ndQuarter  NextYearForecastNetSales2ndQuarter  NextYearForecastOperatingProfit2ndQuarter  NextYearForecastOrdinaryProfit2ndQuarter  NextYearForecastProfit2ndQuarter  NextYearForecastEarningsPerShare2ndQuarter  ForecastNetSales  ForecastOperatingProfit  ForecastOrdinaryProfit  ForecastProfit  ForecastEarningsPerShare  NextYearForecastNetSales  NextYearForecastOperatingProfit  NextYearForecastOrdinaryProfit  NextYearForecastProfit  NextYearForecastEarningsPerShare  MaterialChangesInSubsidiaries  SignificantChangesInTheScopeOfConsolidation  ChangesBasedOnRevisionsOfAccountingStandard  ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard  ChangesInAccountingEstimates  RetrospectiveRestatement  NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock  NumberOfTreasuryStockAtTheEndOfFiscalYear  AverageNumberOfShares  NonConsolidatedNetSales  NonConsolidatedOperatingProfit  NonConsolidatedOrdinaryProfit  NonConsolidatedProfit  NonConsolidatedEarningsPerShare  NonConsolidatedTotalAssets  NonConsolidatedEquity  NonConsolidatedEquityToAssetRatio  NonConsolidatedBookValuePerShare  ForecastNonConsolidatedNetSales2ndQuarter  ForecastNonConsolidatedOperatingProfit2ndQuarter  ForecastNonConsolidatedOrdinaryProfit2ndQuarter  ForecastNonConsolidatedProfit2ndQuarter  ForecastNonConsolidatedEarningsPerShare2ndQuarter  NextYearForecastNonConsolidatedNetSales2ndQuarter  NextYearForecastNonConsolidatedOperatingProfit2ndQuarter  NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter  NextYearForecastNonConsolidatedProfit2ndQuarter  NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter  ForecastNonConsolidatedNetSales  ForecastNonConsolidatedOperatingProfit  ForecastNonConsolidatedOrdinaryProfit  ForecastNonConsolidatedProfit  ForecastNonConsolidatedEarningsPerShare  NextYearForecastNonConsolidatedNetSales  NextYearForecastNonConsolidatedOperatingProfit  NextYearForecastNonConsolidatedOrdinaryProfit  NextYearForecastNonConsolidatedProfit  NextYearForecastNonConsolidatedEarningsPerShare  FiscalYear   Period
Date                      Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
2016-04-01 00:00:00+09:00 91070    20160331448091                   EarnForecastRevision                  FY             2015-04-01           2016-03-31                 2015-04-01               2016-03-31                     NaT                   NaT           NaN              NaN             NaN           NaN               NaN                      NaN           NaN           NaN                 NaN                NaN                               NaN                               NaN                               NaN                 NaN                               NaN                               NaN                               NaN                                  NaN                           NaN                         NaN                            NaN                      NaN                                 NaN                                 NaN                                 NaN                                    NaN                             NaN                                 NaN                              NaN                        NaN                                         NaN                                         NaN                                         NaN                                            NaN                                     NaN                                         NaN                                NaN                         NaN                                NaN                               NaN                       NaN                                 NaN                                 NaN                                        NaN                                       NaN                               NaN                                         NaN      1.250000e+12             7.000000e+09            1.000000e+09   -5.000000e+10                    -53.35                       NaN                              NaN                             NaN                     NaN                               NaN                          False                                        False                                        False                                                     False                         False                     False                                                                           NaN                                        NaN                    NaN                      NaN                             NaN                            NaN                    NaN                              NaN                         NaN                    NaN                                NaN                               NaN                                        NaN                                               NaN                                              NaN                                      NaN                                                NaN                                                NaN                                                       NaN                                                      NaN                                              NaN                                                        NaN                              NaN                                     NaN                                    NaN                            NaN                                      NaN                                      NaN                                             NaN                                            NaN                                    NaN                                              NaN        2015  2015.75
2016-04-04 00:00:00+09:00 28090    20160329445600  1QFinancialStatements_Consolidated_JP                  1Q             2015-12-01           2016-02-29                 2015-12-01               2016-11-30                     NaT                   NaT  1.321400e+11     4.350000e+09    4.581000e+09  2.391000e+09             15.76                      NaN  3.639290e+11  2.435320e+11               0.581                NaN                      5.005000e+09                     -9.100000e+09                     -2.240000e+08        3.038400e+10                               NaN                               NaN                               NaN                                  NaN                           NaN                         NaN                            NaN                      NaN                                 NaN                                15.0                                 NaN                                   15.0                            30.0                                 NaN                              NaN                        NaN                                         NaN                                         NaN                                         NaN                                            NaN                                     NaN                                         NaN                                NaN                2.830000e+11                       1.170000e+10                      1.210000e+10              5.700000e+09                               37.55                                 NaN                                        NaN                                       NaN                               NaN                                         NaN      5.750000e+11             2.800000e+10            2.910000e+10    1.500000e+10                     98.83                       NaN                              NaN                             NaN                     NaN                               NaN                          False                                        False                                         True                                                      True                          True                     False                                                                   153000000.0                                  1232736.0            151767371.0                      NaN                             NaN                            NaN                    NaN                              NaN                         NaN                    NaN                                NaN                               NaN                                        NaN                                               NaN                                              NaN                                      NaN                                                NaN                                                NaN                                                       NaN                                                      NaN                                              NaN                                                        NaN                              NaN                                     NaN                                    NaN                            NaN                                      NaN                                      NaN                                             NaN                                            NaN                                    NaN                                              NaN        2015  2015.00
                          82270    20160322439631  FYFinancialStatements_Consolidated_JP                  FY             2015-02-21           2016-02-20                 2015-02-21               2016-02-20              2016-02-21            2017-02-20  5.460580e+11     3.991300e+10    4.070900e+10  2.474700e+10            673.25                      NaN  3.512830e+11  3.048430e+11               0.868            8293.63                      2.372000e+10                      3.350400e+10                     -9.118000e+09        7.194300e+10                               NaN                              95.0                               NaN                                100.0                         195.0                         NaN                   7.167000e+09                     0.29                                 NaN                                 NaN                                 NaN                                    NaN                             NaN                                 NaN                              NaN                        NaN                                         NaN                                        97.5                                         NaN                                           97.5                                   195.0                                         NaN                              0.234                         NaN                                NaN                               NaN                       NaN                                 NaN                        2.805000e+11                               2.260000e+10                              2.298000e+10                      1.480000e+10                                      402.66               NaN                      NaN                     NaN             NaN                       NaN              5.742000e+11                     4.620000e+10                    4.688000e+10            3.060000e+10                            832.51                          False                                        False                                         True                                                     False                         False                     False                                                                    36913299.0                                   156975.0             36757774.0             5.402160e+11                    4.046600e+10                   4.139100e+10           2.479600e+10                           674.57                3.517480e+11           3.063820e+11                              0.871                           8355.18                                        NaN                                               NaN                                              NaN                                      NaN                                                NaN                                       2.770000e+11                                                       NaN                                             2.320000e+10                                     1.510000e+10                                                     410.81                              NaN                                     NaN                                    NaN                            NaN                                      NaN                             5.665000e+11                                             NaN                                   4.720000e+10                           3.100000e+10                                           843.36        2015  2015.75
```

</details>

### `listed_info_train.parquet`

- 行数: 5,274,692 / サイズ: 152.7MB / 列数: 11
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は先頭 5 行ベース（大容量のため）。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `CompanyName` | string | 0.0% | 5 | 極洋 |
| `CompanyNameEnglish` | string | 0.0% | 5 | KYOKUYO CO.,LTD. |
| `Sector17Code` | string | 0.0% | 2 | 1 |
| `Sector17CodeName` | string | 0.0% | 2 | 食品 |
| `Sector33Code` | string | 0.0% | 2 | 0050 |
| `Sector33CodeName` | string | 0.0% | 2 | 水産・農林業 |
| `ScaleCategory` | string | 0.0% | 2 | TOPIX Small 2 |
| `MarketCode` | string | 0.0% | 2 | 0101 |
| `MarketCodeName` | string | 0.0% | 2 | 東証一部 |
| `MarginCode` | string | 0.0% | 1 | 2 |
| `MarginCodeName` | string | 0.0% | 1 | 貸借 |

<details><summary>先頭3行の値</summary>

```text
                                       CompanyName          CompanyNameEnglish Sector17Code Sector17CodeName Sector33Code Sector33CodeName  ScaleCategory MarketCode MarketCodeName MarginCode MarginCodeName
Date       Code                                                                                                                                                                                              
2008-11-04 13010                                極洋            KYOKUYO CO.,LTD.            1               食品         0050           水産・農林業  TOPIX Small 2       0101           東証一部          2             貸借
           13050     大和証券投資信託委託株式会社　　ダイワ上場投信−トピックス             Daiwa ETF-TOPIX           99              その他         9999              その他              -       0109            その他          2             貸借
           13060  野村アセットマネジメント株式会社　　ＴＯＰＩＸ連動型上場投資信託  TOPIX Exchange Traded Fund           99              その他         9999              その他              -       0109            その他          2             貸借
```

</details>

### `listed_info_valid.parquet`

- 行数: 10,415,399 / サイズ: 304.3MB / 列数: 11
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は先頭 5 行ベース（大容量のため）。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `CompanyName` | string | 0.0% | 5 | 極洋 |
| `CompanyNameEnglish` | string | 0.0% | 5 | KYOKUYO CO.,LTD. |
| `Sector17Code` | string | 0.0% | 2 | 1 |
| `Sector17CodeName` | string | 0.0% | 2 | 食品 |
| `Sector33Code` | string | 0.0% | 2 | 0050 |
| `Sector33CodeName` | string | 0.0% | 2 | 水産・農林業 |
| `ScaleCategory` | string | 0.0% | 2 | TOPIX Small 2 |
| `MarketCode` | string | 0.0% | 2 | 0101 |
| `MarketCodeName` | string | 0.0% | 2 | 東証一部 |
| `MarginCode` | string | 0.0% | 1 | 2 |
| `MarginCodeName` | string | 0.0% | 1 | 貸借 |

<details><summary>先頭3行の値</summary>

```text
                                       CompanyName          CompanyNameEnglish Sector17Code Sector17CodeName Sector33Code Sector33CodeName  ScaleCategory MarketCode MarketCodeName MarginCode MarginCodeName
Date       Code                                                                                                                                                                                              
2016-04-01 13010                                極洋            KYOKUYO CO.,LTD.            1               食品         0050           水産・農林業  TOPIX Small 2       0101           東証一部          2             貸借
           13050     大和証券投資信託委託株式会社　　ダイワ上場投信−トピックス             Daiwa ETF-TOPIX           99              その他         9999              その他              -       0109            その他          2             貸借
           13060  野村アセットマネジメント株式会社　　ＴＯＰＩＸ連動型上場投資信託  TOPIX Exchange Traded Fund           99              その他         9999              その他              -       0109            その他          2             貸借
```

</details>

### `prices_daily_quotes_train.parquet`

- 行数: 809,636 / サイズ: 64.5MB / 列数: 40
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は先頭 5 行ベース（大容量のため）。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Open` | float64 | 0.0% | 5 | 252.0 |
| `High` | float64 | 0.0% | 5 | 259.0 |
| `Low` | float64 | 0.0% | 5 | 248.0 |
| `Close` | float64 | 0.0% | 5 | 259.0 |
| `UpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `LowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `Volume` | float64 | 0.0% | 5 | 1926700.0 |
| `TurnoverValue` | float64 | 0.0% | 5 | 489873400.0 |
| `AdjustmentFactor` | float64 | 0.0% | 1 | 1.0 |
| `AdjustmentOpen` | float64 | 0.0% | 5 | 252.0 |
| `AdjustmentHigh` | float64 | 0.0% | 5 | 259.0 |
| `AdjustmentLow` | float64 | 0.0% | 5 | 248.0 |
| `AdjustmentClose` | float64 | 0.0% | 5 | 259.0 |
| `AdjustmentVolume` | float64 | 0.0% | 5 | 1926700.0 |
| `MorningOpen` | float64 | 0.0% | 5 | 252.0 |
| `MorningHigh` | float64 | 0.0% | 5 | 255.0 |
| `MorningLow` | float64 | 0.0% | 5 | 248.0 |
| `MorningClose` | float64 | 0.0% | 5 | 250.0 |
| `MorningUpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `MorningLowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `MorningVolume` | float64 | 0.0% | 5 | 727800.0 |
| `MorningTurnoverValue` | float64 | 0.0% | 5 | 183438100.0 |
| `MorningAdjustmentOpen` | float64 | 0.0% | 5 | 252.0 |
| `MorningAdjustmentHigh` | float64 | 0.0% | 5 | 255.0 |
| `MorningAdjustmentLow` | float64 | 0.0% | 5 | 248.0 |
| `MorningAdjustmentClose` | float64 | 0.0% | 5 | 250.0 |
| `MorningAdjustmentVolume` | float64 | 0.0% | 5 | 727800.0 |
| `AfternoonOpen` | float64 | 0.0% | 5 | 254.0 |
| `AfternoonHigh` | float64 | 0.0% | 5 | 259.0 |
| `AfternoonLow` | float64 | 0.0% | 5 | 253.0 |
| `AfternoonClose` | float64 | 0.0% | 5 | 259.0 |
| `AfternoonUpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `AfternoonLowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `AfternoonVolume` | float64 | 0.0% | 5 | 1198900.0 |
| `AfternoonTurnoverValue` | float64 | 0.0% | 5 | 306435300.0 |
| `AfternoonAdjustmentOpen` | float64 | 0.0% | 5 | 254.0 |
| `AfternoonAdjustmentHigh` | float64 | 0.0% | 5 | 259.0 |
| `AfternoonAdjustmentLow` | float64 | 0.0% | 5 | 253.0 |
| `AfternoonAdjustmentClose` | float64 | 0.0% | 5 | 259.0 |
| `AfternoonAdjustmentVolume` | float64 | 0.0% | 5 | 1198900.0 |

<details><summary>先頭3行の値</summary>

```text
                      Open      High       Low     Close  UpperLimit  LowerLimit     Volume  TurnoverValue  AdjustmentFactor  AdjustmentOpen  AdjustmentHigh  AdjustmentLow  AdjustmentClose  AdjustmentVolume  MorningOpen  MorningHigh  MorningLow  MorningClose  MorningUpperLimit  MorningLowerLimit  MorningVolume  MorningTurnoverValue  MorningAdjustmentOpen  MorningAdjustmentHigh  MorningAdjustmentLow  MorningAdjustmentClose  MorningAdjustmentVolume  AfternoonOpen  AfternoonHigh  AfternoonLow  AfternoonClose  AfternoonUpperLimit  AfternoonLowerLimit  AfternoonVolume  AfternoonTurnoverValue  AfternoonAdjustmentOpen  AfternoonAdjustmentHigh  AfternoonAdjustmentLow  AfternoonAdjustmentClose  AfternoonAdjustmentVolume
Date       Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
2008-11-04 13320     252.0     259.0     248.0     259.0         0.0         0.0  1926700.0   4.898734e+08               1.0           252.0           259.0          248.0            259.0         1926700.0        252.0        255.0       248.0         250.0                0.0                0.0       727800.0          1.834381e+08                  252.0                  255.0                 248.0                   250.0                 727800.0          254.0          259.0         253.0           259.0                  0.0                  0.0        1198900.0            3.064353e+08                    254.0                    259.0                   253.0                     259.0                  1198900.0
           14140    1715.0    1772.0    1688.0    1759.0         0.0         0.0   137400.0   2.382756e+08               1.0           214.4           221.5          211.0            219.9         1099200.0       1715.0       1766.0      1688.0        1715.0                0.0                0.0        68400.0          1.176018e+08                  214.4                  220.8                 211.0                   214.4                 547200.0         1720.0         1772.0        1714.0          1759.0                  0.0                  0.0          69000.0            1.206738e+08                    215.0                    221.5                   214.3                     219.9                   552000.0
           16050  586000.0  586000.0  565000.0  580000.0         0.0         0.0     7190.0   4.162937e+09               1.0          1465.0          1465.0         1412.5           1450.0         2876000.0     586000.0     586000.0    565000.0      566000.0                0.0                0.0         2591.0          1.490420e+09                 1465.0                 1465.0                1412.5                  1415.0                1036400.0       570000.0       586000.0      569000.0        580000.0                  0.0                  0.0           4599.0            2.672517e+09                   1425.0                   1465.0                  1422.5                    1450.0                  1839600.0
```

</details>

### `prices_daily_quotes_valid.parquet`

- 行数: 1,227,148 / サイズ: 103.5MB / 列数: 40
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は先頭 5 行ベース（大容量のため）。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Open` | float64 | 0.0% | 5 | 542.0 |
| `High` | float64 | 0.0% | 5 | 545.0 |
| `Low` | float64 | 0.0% | 5 | 530.0 |
| `Close` | float64 | 0.0% | 5 | 539.0 |
| `UpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `LowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `Volume` | float64 | 0.0% | 5 | 3689900.0 |
| `TurnoverValue` | float64 | 0.0% | 5 | 1986110900.0 |
| `AdjustmentFactor` | float64 | 0.0% | 1 | 1.0 |
| `AdjustmentOpen` | float64 | 0.0% | 5 | 542.0 |
| `AdjustmentHigh` | float64 | 0.0% | 5 | 545.0 |
| `AdjustmentLow` | float64 | 0.0% | 5 | 530.0 |
| `AdjustmentClose` | float64 | 0.0% | 5 | 539.0 |
| `AdjustmentVolume` | float64 | 0.0% | 5 | 3689900.0 |
| `MorningOpen` | float64 | 0.0% | 5 | 542.0 |
| `MorningHigh` | float64 | 0.0% | 5 | 545.0 |
| `MorningLow` | float64 | 0.0% | 5 | 530.0 |
| `MorningClose` | float64 | 0.0% | 5 | 534.0 |
| `MorningUpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `MorningLowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `MorningVolume` | float64 | 0.0% | 5 | 1548400.0 |
| `MorningTurnoverValue` | float64 | 0.0% | 5 | 831578200.0 |
| `MorningAdjustmentOpen` | float64 | 0.0% | 5 | 542.0 |
| `MorningAdjustmentHigh` | float64 | 0.0% | 5 | 545.0 |
| `MorningAdjustmentLow` | float64 | 0.0% | 5 | 530.0 |
| `MorningAdjustmentClose` | float64 | 0.0% | 5 | 534.0 |
| `MorningAdjustmentVolume` | float64 | 0.0% | 5 | 1548400.0 |
| `AfternoonOpen` | float64 | 0.0% | 5 | 533.0 |
| `AfternoonHigh` | float64 | 0.0% | 5 | 542.0 |
| `AfternoonLow` | float64 | 0.0% | 5 | 533.0 |
| `AfternoonClose` | float64 | 0.0% | 5 | 539.0 |
| `AfternoonUpperLimit` | float64 | 0.0% | 1 | 0.0 |
| `AfternoonLowerLimit` | float64 | 0.0% | 1 | 0.0 |
| `AfternoonVolume` | float64 | 0.0% | 5 | 2141500.0 |
| `AfternoonTurnoverValue` | float64 | 0.0% | 5 | 1154532700.0 |
| `AfternoonAdjustmentOpen` | float64 | 0.0% | 5 | 533.0 |
| `AfternoonAdjustmentHigh` | float64 | 0.0% | 5 | 542.0 |
| `AfternoonAdjustmentLow` | float64 | 0.0% | 5 | 533.0 |
| `AfternoonAdjustmentClose` | float64 | 0.0% | 5 | 539.0 |
| `AfternoonAdjustmentVolume` | float64 | 0.0% | 5 | 2141500.0 |

<details><summary>先頭3行の値</summary>

```text
                    Open    High     Low   Close  UpperLimit  LowerLimit     Volume  TurnoverValue  AdjustmentFactor  AdjustmentOpen  AdjustmentHigh  AdjustmentLow  AdjustmentClose  AdjustmentVolume  MorningOpen  MorningHigh  MorningLow  MorningClose  MorningUpperLimit  MorningLowerLimit  MorningVolume  MorningTurnoverValue  MorningAdjustmentOpen  MorningAdjustmentHigh  MorningAdjustmentLow  MorningAdjustmentClose  MorningAdjustmentVolume  AfternoonOpen  AfternoonHigh  AfternoonLow  AfternoonClose  AfternoonUpperLimit  AfternoonLowerLimit  AfternoonVolume  AfternoonTurnoverValue  AfternoonAdjustmentOpen  AfternoonAdjustmentHigh  AfternoonAdjustmentLow  AfternoonAdjustmentClose  AfternoonAdjustmentVolume
Date       Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
2016-04-01 13320   542.0   545.0   530.0   539.0         0.0         0.0  3689900.0   1.986111e+09               1.0           542.0           545.0          530.0            539.0         3689900.0        542.0        545.0       530.0         534.0                0.0                0.0      1548400.0           831578200.0                  542.0                  545.0                 530.0                   534.0                1548400.0          533.0          542.0         533.0           539.0                  0.0                  0.0        2141500.0            1.154533e+09                    533.0                    542.0                   533.0                     539.0                  2141500.0
           13330  2080.0  2080.0  2026.0  2029.0         0.0         0.0   438000.0   8.961062e+08               1.0           693.3           693.3          675.3            676.3         1314000.0       2080.0       2080.0      2041.0        2049.0                0.0                0.0       211900.0           435300300.0                  693.3                  693.3                 680.3                   683.0                 635700.0         2040.0         2051.0        2026.0          2029.0                  0.0                  0.0         226100.0            4.608059e+08                    680.0                    683.7                   675.3                     676.3                   678300.0
           14140  4310.0  4320.0  4135.0  4135.0         0.0         0.0    97100.0   4.085405e+08               1.0           538.8           540.0          516.9            516.9          776800.0       4310.0       4320.0      4175.0        4205.0                0.0                0.0        49000.0           208502000.0                  538.8                  540.0                 521.9                   525.6                 392000.0         4185.0         4190.0        4135.0          4135.0                  0.0                  0.0          48100.0            2.000385e+08                    523.1                    523.8                   516.9                     516.9                   384800.0
```

</details>

### `raw_return_1day_train.parquet`

- 行数: 809,636 / サイズ: 5.0MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.1% | 245675 | -0.015625 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2008-11-04 13320 -0.015625
           14140 -0.004180
           16050  0.029877
```

</details>

### `raw_return_1day_valid.parquet`

- 行数: 1,227,148 / サイズ: 9.2MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.0% | 457648 | -0.03900709219858156 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2016-04-01 13320 -0.039007
           13330 -0.033997
           14140  0.025114
```

</details>

### `raw_target_1day_train.parquet`

- 行数: 809,636 / サイズ: 5.0MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.1% | 245726 | -0.037735849056603765 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2008-11-04 13320 -0.037736
           14140 -0.012135
           16050 -0.050000
```

</details>

### `raw_target_1day_valid.parquet`

- 行数: 1,227,148 / サイズ: 9.2MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.1% | 457337 | 0.053505535055350606 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2016-04-01 13320  0.053506
           13330  0.045541
           14140 -0.008440
```

</details>

### `target_1day_train.parquet`

- 行数: 809,636 / サイズ: 6.7MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.7% | 804043 | -0.048959691680344655 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2008-11-04 13320 -0.048960
           14140 -0.019851
           16050 -0.064317
```

</details>

### `target_1day_valid.parquet`

- 行数: 1,227,148 / サイズ: 10.5MB / 列数: 1
- 索引: `['Date', 'Code']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.3% | 1223593 | 0.05359673330657059 |

<details><summary>先頭3行の値</summary>

```text
                    Return
Date       Code           
2016-04-01 13320  0.053597
           13330  0.045608
           14140 -0.008340
```

</details>

### `topix_return_1day_train.parquet`

- 行数: 1,814 / サイズ: 34.2KB / 列数: 1
- 索引: `['Date']`
- Date 範囲: 2008-11-04 00:00:00 .. 2016-03-31 00:00:00（1814 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.0% | 1814 | 0.001610936526305548 |

<details><summary>先頭3行の値</summary>

```text
              Return
Date                
2008-11-04  0.001611
2008-11-05  0.044578
2008-11-06  0.011087
```

</details>

### `topix_return_1day_valid.parquet`

- 行数: 2,523 / サイズ: 47.6KB / 列数: 1
- 索引: `['Date']`
- Date 範囲: 2016-04-01 00:00:00 .. 2026-07-31 00:00:00（2523 日）
- 統計は全行ベース。

| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |
|---|---|---|---|---|
| `Return` | float64 | 0.0% | 2523 | -0.011814211361235526 |

<details><summary>先頭3行の値</summary>

```text
              Return
Date                
2016-04-01 -0.011814
2016-04-04 -0.034224
2016-04-05 -0.000117
```

</details>
