"""財務ファンダメンタル戦略（ロングホライズンの例）。

仮説
----
決算短信で開示される「稼ぐ力」と「財務の頑健さ」は、年単位で持続する。
利益率やキャッシュ創出力は四半期ごとに乱高下せず、平均回帰も遅い。
したがってこれらを根拠にした銘柄の優劣は、数ヶ月スケールでしか入れ替わらない。

戦略の枠組み
------------
1. クオリティ（稼ぐ力・頑健さ）を3指標で測る
   - 営業キャッシュフロー / 総資産  … 会計上の利益ではなく現金の創出力
   - 当期純利益 / 自己資本 (ROE)    … 株主資本に対する収益性
   - 自己資本比率                    … 財務レバレッジの低さ
2. バリュー（割安さ）を2指標で測る
   - 1株当たり純資産 / 株価 (B/P)
   - 会社予想EPS / 株価 (予想E/P)
3. すべて日次の断面でZ標準化し、単純合算する

ロングホライズンであることの意味
--------------------------------
このコンペは日次リバランスが前提だが、**シグナルがゆっくりしか動かなければ、
実際の売買はほとんど発生しない**。本戦略の日次回転率は約0.033で、
これは年間で建玉の約8.3倍しか売買しない＝平均保有期間およそ2.5ヶ月に相当する。

  年率コスト = 片道0.1% × 日次回転率 × 252営業日
             = 0.001 × 0.033 × 252 = 0.83%

一方、1日リバーサル（回転率1.34）だと年率コストは33.7%になる。
「毎日リバランスしなくてよい」＝「毎日リバランスさせられる仕組みの上でも、
遅いシグナルなら実質的に長期保有になる」ということ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SHARES_COLUMN = (
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"
)
FINS_COLUMNS = [
    "Equity",
    "TotalAssets",
    "Profit",
    "CashFlowsFromOperatingActivities",
    "EquityToAssetRatio",
    "ForecastEarningsPerShare",
    SHARES_COLUMN,
]


def load_history(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    train = pd.read_parquet(f"{name}_train.parquet", columns=columns)
    valid = pd.read_parquet(f"{name}_valid.parquet", columns=columns)
    return pd.concat([train, valid]).sort_index()


def cross_section_z(values: pd.Series) -> pd.Series:
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0)


def fill_with_daily_median(values: pd.Series) -> pd.Series:
    filled = values.fillna(values.groupby(level="Date").transform("median"))
    return filled.fillna(0.0)


def asof_join_fins(panel_index: pd.MultiIndex, columns: list[str]) -> pd.DataFrame:
    """決算短信を開示日基準の backward as-of で日次パネルへ展開する。

    ここが決算データを扱うときの肝。
    - Date は tz-aware(Asia/Tokyo) なので、価格系列に合わせて tz を外す
    - 未来方向の bfill は絶対に使わない（開示前の値を使うことになる）
    - 列ごとに独立して直近の開示値を引き継ぐ（欠損した項目は前回開示分を使う）
    """
    fins = load_history("fins_statements", columns=columns).reset_index()
    fins["Date"] = fins["Date"].dt.tz_localize(None)
    fins["Code"] = fins["Code"].astype(str)
    fins = fins.sort_values("Date")
    # 開示ごとに欠損する項目があるため、銘柄内で過去方向にのみ引き継ぐ。
    fins[columns] = fins.groupby("Code", sort=False)[columns].ffill()

    base = panel_index.to_frame(index=False)
    base["Code"] = base["Code"].astype(str)
    merged = pd.merge_asof(base, fins, on="Date", by="Code", direction="backward")
    return pd.DataFrame(
        {name: merged[name].to_numpy() for name in columns},
        index=panel_index,
    )


def predict() -> pd.DataFrame:
    prices = load_history("prices_daily_quotes", columns=["Close"])
    panel_index = prices.index
    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index

    fins = asof_join_fins(panel_index, FINS_COLUMNS)
    price = prices["Close"]
    shares = fins[SHARES_COLUMN].where(fins[SHARES_COLUMN] > 0)
    equity = fins["Equity"]

    # --- クオリティ: 稼ぐ力と財務の頑健さ ------------------------------------
    quality = (
        cross_section_z(fins["CashFlowsFromOperatingActivities"] / fins["TotalAssets"])
        + cross_section_z(fins["Profit"] / equity)
        + cross_section_z(fins["EquityToAssetRatio"])
    )

    # --- バリュー: 株価に対する純資産と予想利益 -------------------------------
    book_per_share = equity / shares
    value = (
        cross_section_z(book_per_share / price)
        + cross_section_z(fins["ForecastEarningsPerShare"] / price)
    )

    signal = fill_with_daily_median(quality) + fill_with_daily_median(value)
    return signal.rename("Return").to_frame().loc[valid_index]
