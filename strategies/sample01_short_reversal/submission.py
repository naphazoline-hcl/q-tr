"""短期リバーサル戦略。

仮説
----
1日で大きく売られた銘柄は、売り手の換金需要や需給の一時的な偏りで下げすぎている。
その偏りは翌日以降に解消するので、下げた銘柄を買い、上げた銘柄を売る。

予測対象が市場残差リターンなので、シグナルも市場の動きを抜いた残差で測る。

    残差リターン = 実際のリターン − β × 市場リターン

戦略の枠組み
------------
1. 前営業日の始値から当日の始値までのリターン（`raw_return_1day`）を読む
2. β（`beta_1day`）と市場リターン（`topix_return_1day`）で市場の動きを差し引く
3. 符号を反転する。下げた銘柄ほどスコアが高くなる

これだけである。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_history(name: str) -> pd.DataFrame:
    """TrainとValidを縦に連結する。Valid初日にも履歴が必要な戦略のための定型。"""
    train = pd.read_parquet(f"{name}_train.parquet")
    valid = pd.read_parquet(f"{name}_valid.parquet")
    return pd.concat([train, valid]).sort_index()


def fill_with_daily_median(values: pd.Series) -> pd.Series:
    """欠損はその日の断面中央値で埋める。"""
    filled = values.fillna(values.groupby(level="Date").transform("median"))
    return filled.fillna(0.0)


def predict() -> pd.DataFrame:
    raw_return = load_history("raw_return_1day")
    beta = load_history("beta_1day")
    market = load_history("topix_return_1day")
    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index

    panel_index = raw_return.index
    beta_series = beta["Return"].reindex(panel_index)

    # 市場リターンは日付だけのSeries。各行の日付に合わせて広げる。
    dates = panel_index.get_level_values("Date")
    market_series = pd.Series(
        market["Return"].reindex(dates).to_numpy(),
        index=panel_index,
    )

    residual = raw_return["Return"] - beta_series * market_series
    signal = fill_with_daily_median(-residual)

    return signal.rename("Return").to_frame().loc[valid_index]
