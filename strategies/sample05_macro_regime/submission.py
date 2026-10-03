"""マクロ局面によるファクター配分切替（外部データの使い方の例）。

まず押さえるべき構造
--------------------
このコンペの採点は「その日の断面でシグナルを順位付けし、5分位に切る」。
つまり**同じ日の中で単調変換しても、できあがるポートフォリオは変わらない**。
消費者態度指数のようなマクロ変数は、ある日のすべての銘柄で同じ値を取る。
だから、マクロ変数を足しても掛けても、それ単独では順位が1ミリも動かない。

  マクロ変数が効くのは、**複数の断面シグナルの相対ウェイトを動かすとき**だけ。

これが本サンプルの設計の出発点である。

仮説
----
フライト・トゥ・クオリティ。家計の景況感（消費者態度指数）が悪化している局面では、
投資家は業績の下振れを警戒し、財務が頑健で現金を稼げる企業にプレミアムを払う。
逆に景況感が良い局面では、リスクを取って割安株（バリュー）を拾いにいく。
したがって、クオリティとバリューの「どちらをどれだけ重視するか」は、
景況感の水準によって変えるべきである。

戦略の枠組み
------------
1. 断面シグナルを2本用意する（どちらも sample04 と同じ定義）
   - クオリティ : 営業CF/総資産 + ROE + 自己資本比率
   - バリュー   : B/P + 予想E/P
2. 消費者態度指数を、過去3年の平均・標準偏差で標準化する（水準の高低を測る）
3. 景況感が低いほどクオリティへ、高いほどバリューへ配分を寄せる

     w      = clip(-DI_z / 2, -1, +1)
     signal = (1 + w) × クオリティ + (1 - w) × バリュー

   w = 0 なら等ウェイト、w = +1 ならクオリティのみ、w = -1 ならバリューのみ。

先読みをしない結合
------------------
`consumer_attitude_index_*.parquet` の `Date` は観測月ではなく
**その値が利用可能になった日時**（Asia/Tokyo）である。日次パネルへは
tz を外して `merge_asof(direction="backward")` で結合し、未来方向の bfill はしない。
また3年ローリングの標準化にはTrain期間の履歴が必要なので、Trainのマクロ系列も
連結してから計算し、最後にValidの行だけを返す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_WINDOW = 750       # 約3年。景況感の「平常値」を測る窓
REGIME_MIN_PERIODS = 250  # 約1年たてば標準化を始める
REGIME_CLIP = 2.0

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


def build_regime_weight(trading_days: pd.DatetimeIndex) -> pd.Series:
    """消費者態度指数から、クオリティ寄せ／バリュー寄せの配分係数を作る。"""
    macro = load_history(
        "consumer_attitude_index", columns=["ConsumerAttitudeIndex"]
    ).reset_index()
    # Date は観測月ではなく「利用可能日時(JST)」で、配布側が**同日先読みを避けるため**
    # 23:59:59 を割り当てている。営業日(00:00)と直接 backward as-of すれば、
    # ある日の23:59:59に公表された値はその日には使われず、翌営業日から使われる。
    #
    # ここで normalize() して日付に丸めてはいけない。丸めると 23:59:59 → 00:00:00 となり、
    # 公表当日の日中から使えることになって、配布側の先読み防止が無効になる。
    macro["Date"] = macro["Date"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    macro = macro.drop_duplicates(subset="Date", keep="last").sort_values("Date")

    base = pd.DataFrame({"Date": trading_days})
    merged = pd.merge_asof(base, macro, on="Date", direction="backward")
    index_series = pd.Series(
        merged["ConsumerAttitudeIndex"].to_numpy(), index=trading_days
    )

    # 過去3年の平均・標準偏差で標準化する（rollingは過去方向のみ）。
    rolling = index_series.rolling(REGIME_WINDOW, min_periods=REGIME_MIN_PERIODS)
    regime_z = ((index_series - rolling.mean()) / rolling.std()).clip(
        -REGIME_CLIP, REGIME_CLIP
    )
    # 景況感が低い（zが負）ほどクオリティ寄り。履歴が足りない期間は等ウェイト。
    return (-regime_z / 2.0).clip(-1.0, 1.0).fillna(0.0)


def predict() -> pd.DataFrame:
    prices = load_history("prices_daily_quotes", columns=["Close"])
    panel_index = prices.index
    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index

    fins = asof_join_fins(panel_index, FINS_COLUMNS)
    price = prices["Close"]
    shares = fins[SHARES_COLUMN].where(fins[SHARES_COLUMN] > 0)
    equity = fins["Equity"]

    quality = fill_with_daily_median(
        cross_section_z(fins["CashFlowsFromOperatingActivities"] / fins["TotalAssets"])
        + cross_section_z(fins["Profit"] / equity)
        + cross_section_z(fins["EquityToAssetRatio"])
    )
    value = fill_with_daily_median(
        cross_section_z((equity / shares) / price)
        + cross_section_z(fins["ForecastEarningsPerShare"] / price)
    )

    trading_days = panel_index.get_level_values("Date").unique().sort_values()
    weight_by_day = build_regime_weight(pd.DatetimeIndex(trading_days))
    weight = pd.Series(
        weight_by_day.reindex(panel_index.get_level_values("Date")).to_numpy(),
        index=panel_index,
    )

    signal = (1.0 + weight) * quality + (1.0 - weight) * value
    return signal.rename("Return").to_frame().loc[valid_index]
