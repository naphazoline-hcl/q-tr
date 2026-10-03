"""マルチファクター戦略（複数のリスクプレミアムを合成する例）。

仮説
----
超過リターンの源泉は一つではない。規模・流動性・クオリティ・バリューは、
それぞれ別のリスク（換金しにくさ／事業の脆さ／期待の織り込みすぎ）に対する
対価であり、効く局面も異なる。互いに相関の低いものを足し合わせれば、
どれか一つに賭けるより予測が安定する。

戦略の枠組み
------------
1. 3つのブロックを作り、それぞれをZ標準化して尺度を揃える
   - 規模・流動性ブロック : −log(時価総額) + Amihud非流動性
   - クオリティブロック   : 営業CF/総資産 + ROE + 自己資本比率
   - バリューブロック     : B/P + 予想E/P
2. ブロック単位で等ウェイト合算する

なぜ「ブロック」にまとめるのか
------------------------------
規模と非流動性の断面相関は約0.89ある。**別々のファクターのつもりでも、
実質的に同じものを二重に賭けている**ことがある。個別指標を等ウェイトで並べると
相関の高い指標群に配分が偏るため、先に相関を確認してブロックへまとめ、
ブロック間で等ウェイトにする。

合成の効果と限界
----------------
合成すると予測精度（RankIC）は単独ファクターより上がる。ただしコスト控除後の
スコアは回転率とのトレードオフになるため、「ICが上がる＝スコアが上がる」ではない。
実測の比較は README.md を参照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ILLIQ_WINDOW = 60
ILLIQ_MIN_PERIODS = 40
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
SCALE_RANK = {
    "TOPIX Core30": 5.0,
    "TOPIX Large70": 4.0,
    "TOPIX Mid400": 3.0,
    "TOPIX Small 1": 2.0,
    "TOPIX Small 2": 1.0,
}


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
    fins = load_history("fins_statements", columns=columns).reset_index()
    fins["Date"] = fins["Date"].dt.tz_localize(None)
    fins["Code"] = fins["Code"].astype(str)
    fins = fins.sort_values("Date")
    fins[columns] = fins.groupby("Code", sort=False)[columns].ffill()

    base = panel_index.to_frame(index=False)
    base["Code"] = base["Code"].astype(str)
    merged = pd.merge_asof(base, fins, on="Date", by="Code", direction="backward")
    return pd.DataFrame(
        {name: merged[name].to_numpy() for name in columns},
        index=panel_index,
    )


def build_blocks(panel_index: pd.MultiIndex) -> pd.DataFrame:
    """3つのファクターブロックを (Date, Code) パネルとして返す。"""
    prices = load_history("prices_daily_quotes", columns=["Close", "TurnoverValue"])
    raw_return = load_history("raw_return_1day")["Return"].reindex(panel_index)
    fins = asof_join_fins(panel_index, FINS_COLUMNS)
    price = prices["Close"]

    # 規模・流動性
    shares = fins[SHARES_COLUMN].where(fins[SHARES_COLUMN] > 0)
    market_cap = price * shares
    size = cross_section_z(-np.log(market_cap.where(market_cap > 0)))
    listed = load_history("listed_info", columns=["ScaleCategory"])
    scale = listed["ScaleCategory"].reindex(panel_index).map(SCALE_RANK)
    size = fill_with_daily_median(size.fillna(cross_section_z(-scale.astype("float64"))))

    turnover = prices["TurnoverValue"].replace(0.0, np.nan)
    illiq = (raw_return.abs() / turnover).groupby(level="Code", sort=False).transform(
        lambda values: values.rolling(
            window=ILLIQ_WINDOW, min_periods=ILLIQ_MIN_PERIODS
        ).mean()
    )
    # 非流動性は対数を取らない。流動性プレミアムは極端に売買しにくい一部の銘柄に
    # 集中しており、生のままZ標準化した方が裾が立つ（規模は逆に対数化する）。
    illiq = fill_with_daily_median(cross_section_z(illiq))

    # クオリティ
    equity = fins["Equity"]
    quality = (
        cross_section_z(fins["CashFlowsFromOperatingActivities"] / fins["TotalAssets"])
        + cross_section_z(fins["Profit"] / equity)
        + cross_section_z(fins["EquityToAssetRatio"])
    )

    # バリュー
    value = (
        cross_section_z((equity / shares) / price)
        + cross_section_z(fins["ForecastEarningsPerShare"] / price)
    )

    return pd.DataFrame(
        {
            "size_liquidity": cross_section_z(size + illiq),
            "quality": cross_section_z(fill_with_daily_median(quality)),
            "value": cross_section_z(fill_with_daily_median(value)),
        },
        index=panel_index,
    )


def predict() -> pd.DataFrame:
    panel_index = load_history("prices_daily_quotes", columns=["Close"]).index
    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index

    blocks = build_blocks(panel_index)
    signal = blocks.sum(axis=1).pipe(fill_with_daily_median)
    return signal.rename("Return").to_frame().loc[valid_index]
