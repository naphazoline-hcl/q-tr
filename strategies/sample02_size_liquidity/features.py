"""sample02 で共有する、未来を見ない特徴量定義。

`train.py`（係数の学習）と `submission.py`（推論）の両方から呼ぶ。
同じ関数を使うことで、学習時と推論時で特徴量の定義がずれるのを防ぐ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = ("size", "illiq")

ILLIQ_WINDOW = 60
ILLIQ_MIN_PERIODS = 40

SHARES_COLUMN = (
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"
)

# ScaleCategory は時価総額の粗い代理。決算データが無い銘柄（ETF等）の穴埋めに使う。
SCALE_RANK = {
    "TOPIX Core30": 5.0,
    "TOPIX Large70": 4.0,
    "TOPIX Mid400": 3.0,
    "TOPIX Small 1": 2.0,
    "TOPIX Small 2": 1.0,
}


def load_history(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """TrainとValidを縦に連結して、Valid冒頭のrolling履歴を確保する。"""
    train = pd.read_parquet(f"{name}_train.parquet", columns=columns)
    valid = pd.read_parquet(f"{name}_valid.parquet", columns=columns)
    return pd.concat([train, valid]).sort_index()


def cross_section_z(values: pd.Series) -> pd.Series:
    """同じ日付の銘柄だけで標準化する。別の日付は集計に混ぜない。"""
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0)


def fill_with_daily_median(values: pd.Series) -> pd.Series:
    """欠損はその日の断面中央値で埋める（0埋めすると順位の端に飛ぶため）。"""
    filled = values.fillna(values.groupby(level="Date").transform("median"))
    return filled.fillna(0.0)


def asof_join_fins(panel_index: pd.MultiIndex, column: str) -> pd.Series:
    """決算短信の疎データを、開示日基準の backward as-of で日次パネルへ展開する。"""
    fins = load_history("fins_statements", columns=[column]).reset_index()
    # fins の Date は tz-aware(Asia/Tokyo)。価格系列と揃えるため tz を外す。
    fins["Date"] = fins["Date"].dt.tz_localize(None)
    fins = fins.dropna(subset=[column]).sort_values("Date")
    fins["Code"] = fins["Code"].astype(str)

    base = panel_index.to_frame(index=False)
    base["Code"] = base["Code"].astype(str)
    merged = pd.merge_asof(
        base,
        fins,
        on="Date",
        by="Code",
        direction="backward",  # その時点で開示済みの値だけを使う
    )
    return pd.Series(merged[column].to_numpy(), index=panel_index)


def build_features() -> pd.DataFrame:
    """各行の時点までに確定した値だけから、2つの特徴量を作る。

    どちらも日次の断面でZ標準化してある。単位も分布も揃うので、
    あとは重みを掛けて足すだけで合成できる。
    """
    prices = load_history("prices_daily_quotes", columns=["Close", "TurnoverValue"])
    raw_return = load_history("raw_return_1day")
    panel_index = prices.index

    # --- 規模: 時価総額 = 終値 × 発行済株式数 -------------------------------
    #
    # 対数を取るのは、他の特徴量と足し合わせるため。時価総額の分布は右に大きく歪んでおり
    # （断面の歪度が約 −6）、そのままZ標準化すると上位数銘柄が分散を独占して、
    # 8割の銘柄が |Z| < 0.5 に潰れてしまう。対数化すると歪度は −0.4 程度になる。
    # なお採点は断面ランクしか見ないため、この特徴量を単独で使う場合は
    # 対数を取っても取らなくてもスコアは変わらない。
    shares = asof_join_fins(panel_index, SHARES_COLUMN)
    market_cap = prices["Close"] * shares
    size = cross_section_z(-np.log(market_cap.where(market_cap > 0)))

    # 決算データが無い銘柄は ScaleCategory の粗い順位で代替する。
    listed = load_history("listed_info", columns=["ScaleCategory"])
    scale = listed["ScaleCategory"].reindex(panel_index).map(SCALE_RANK)
    size = fill_with_daily_median(size.fillna(cross_section_z(-scale.astype("float64"))))

    # --- 非流動性: Amihud ILLIQ ---------------------------------------------
    #
    # こちらは対数を取らない。分布が右に歪んでいるのは時価総額と同じ（歪度は約 +6）だが、
    # 流動性プレミアムは「極端に売買しにくい一部の銘柄」に集中しており、
    # 生のままZ標準化すると7割の銘柄が |Z| < 0.5 に収まって、
    # その裾の銘柄だけが強く立つ指標になる。対数化するとこれが均されて薄まる。
    # Train 期間・Valid 期間のどちらで測っても、対数を取らない方が良かった。
    turnover = prices["TurnoverValue"].replace(0.0, np.nan)
    daily_illiq = raw_return["Return"].reindex(panel_index).abs() / turnover
    illiq = daily_illiq.groupby(level="Code", sort=False).transform(
        lambda values: values.rolling(
            window=ILLIQ_WINDOW,
            min_periods=ILLIQ_MIN_PERIODS,
        ).mean()
    )
    illiq = fill_with_daily_median(cross_section_z(illiq))

    return pd.DataFrame({"size": size, "illiq": illiq}, index=panel_index)
