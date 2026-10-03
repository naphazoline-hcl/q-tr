"""robust_multifactor で共有する特徴量定義（未来を見ない）。

`train.py`（モデル学習）と `submission.py`（推論）の両方から呼び、
学習時と推論時で定義がずれないようにする。

時点管理のルール
----------------
- 決算短信は開示日基準の backward as-of join のみ（未来方向の ffill/bfill 禁止）
- rolling は過去方向のみ、`shift` は正の方向のみ
- 公開特徴量（為替・消費者態度指数）は本戦略では使わない
- Train と Valid を縦に連結し、Valid 冒頭の rolling 履歴を確保する

メモリ対策（採点環境の上限対策で、シグナルの数式は変えない）
------------------------------------------------------------
- parquet は pyarrow の predicate pushdown で Date / Code を読む前に絞る
- listed_info は全上場約4,400銘柄×13列あるため、必要5列×ユニバース498銘柄だけ読む
- 特徴量と順位は float32 で保持し、順位化は DataFrame のコピーを作らず in-place
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SHARES_COLUMN = (
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"
)

FINS_COLUMNS = [
    SHARES_COLUMN,
    "Equity",
    "TotalAssets",
    "Profit",
    "NetSales",
    "OperatingProfit",
    "CashFlowsFromOperatingActivities",
    "EquityToAssetRatio",
    "EarningsPerShare",
    "ForecastEarningsPerShare",
    "ForecastDividendPerShareAnnual",
    "ResultDividendPerShareAnnual",
    "ForecastNetSales",
    "ForecastOperatingProfit",
    "ForecastProfit",
]

LISTED_COLUMNS = [
    "ScaleCategory",
    "Sector17Code",
    "Sector33Code",
    "MarketCode",
    "MarginCode",
]

SCALE_RANK = {
    "TOPIX Core30": 5.0,
    "TOPIX Large70": 4.0,
    "TOPIX Mid400": 3.0,
    "TOPIX Small 1": 2.0,
    "TOPIX Small 2": 1.0,
}

SECTOR_COLUMNS = ["sector17", "sector33", "market", "margin"]

# 合成（ブロック）に使うファクターと符号。+1 は値が大きいほどロング。
FACTOR_SIGNS = {
    # 規模・流動性（小型・売買されにくい銘柄をロング）
    "logsize": -1.0,
    "illiq60": +1.0,
    "logturn60": -1.0,
    # バリュー
    "bp": +1.0,
    "sp": +1.0,
    "ep_f": +1.0,
    "ep_a": +1.0,
    "cfy": +1.0,
    "div_y": +1.0,
    # クオリティ
    "roe": +1.0,
    "cfo_ta": +1.0,
    "eq_ratio": +1.0,
    # 低リスク
    "beta": -1.0,
    "vol60": -1.0,
}

FACTOR_BLOCKS = {
    "size": ["logsize", "illiq60", "logturn60"],
    "value": ["bp", "sp", "ep_f", "ep_a", "cfy", "div_y"],
    "quality": ["roe", "cfo_ta", "eq_ratio"],
    "lowrisk": ["beta", "vol60"],
}


def _read_splits(
    name: str,
    splits: tuple[str, ...],
    columns: list[str] | None = None,
    start: pd.Timestamp | None = None,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """parquet を読む。start / codes は pyarrow のフィルタとして読込前に適用する。"""
    filters = []
    if start is not None:
        filters.append(("Date", ">=", start))
    if codes is not None:
        filters.append(("Code", "in", codes))
    frames = [
        pd.read_parquet(f"{name}_{split}.parquet", columns=columns, filters=filters or None)
        for split in splits
    ]
    return pd.concat(frames).sort_index()


def _by_code(values: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    return values.groupby(level="Code", sort=False)


def _rolling(values: pd.Series, window: int, min_periods: int | None = None, how: str = "mean") -> pd.Series:
    if min_periods is None:
        min_periods = max(2, int(window * 0.75))
    return _by_code(values).transform(
        lambda x: x.rolling(window, min_periods=min_periods).agg(how)
    )


def _asof_fins(panel_index: pd.MultiIndex, columns: list[str], adj_ratio: pd.Series) -> pd.DataFrame:
    """決算短信（疎データ）を開示日基準の backward as-of で日次パネルへ展開する。

    `adj_ratio` は `AdjustmentClose / Close`。1株当たり財務値は開示時点の
    株式数基準なので、この比率を開示日にあわせて持ち、調整後価格と同じ基準へ
    そろえる（`_rfactor`）。未来方向の bfill はしない。
    """
    fins = _read_splits("fins_statements", ("train", "valid"), columns=columns).reset_index()
    fins["Date"] = fins["Date"].dt.tz_localize(None)
    fins["Code"] = fins["Code"].astype(str)
    fins["_fdate"] = fins["Date"]
    fins = fins.sort_values("Date")
    fins[columns] = fins.groupby("Code", sort=False)[columns].ffill()

    ratio = adj_ratio.rename("_rfactor").reset_index()
    ratio["Code"] = ratio["Code"].astype(str)
    ratio = ratio.dropna(subset=["_rfactor"]).sort_values("Date")
    fins = pd.merge_asof(fins, ratio, on="Date", by="Code", direction="backward")
    fins["_rfactor"] = fins.groupby("Code", sort=False)["_rfactor"].ffill()

    base = panel_index.to_frame(index=False)
    base["Code"] = base["Code"].astype(str)
    merged = pd.merge_asof(base, fins, on="Date", by="Code", direction="backward")
    out = pd.DataFrame(
        {c: merged[c].to_numpy(dtype=np.float32) for c in columns + ["_rfactor"]},
        index=panel_index,
    )
    out["_fdate"] = pd.Series(
        pd.to_datetime(merged["_fdate"].to_numpy()), index=panel_index
    )
    return out


def build_features(
    splits: tuple[str, ...] = ("train", "valid"),
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """(Date, Code) パネルの生特徴量を返す。ランク化は呼び出し側で行う。"""
    raw = _read_splits("raw_return_1day", splits, columns=["Return"], start=start)["Return"]
    panel_index = raw.index
    codes = sorted(raw.index.get_level_values("Code").unique().tolist())

    px = _read_splits(
        "prices_daily_quotes",
        splits,
        columns=[
            "Close", "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose",
            "MorningAdjustmentOpen", "MorningAdjustmentClose",
            "AfternoonAdjustmentOpen", "AfternoonAdjustmentClose",
            "TurnoverValue", "Volume",
        ],
        start=start,
    )
    if not px.index.equals(panel_index):
        raise RuntimeError("prices と raw_return の index が一致しません")

    ao = px["AdjustmentOpen"]
    ah = px["AdjustmentHigh"]
    al = px["AdjustmentLow"]
    ac = px["AdjustmentClose"]
    mo, mc = px["MorningAdjustmentOpen"], px["MorningAdjustmentClose"]
    no, nc = px["AfternoonAdjustmentOpen"], px["AfternoonAdjustmentClose"]
    turn = px["TurnoverValue"]

    f: dict[str, pd.Series] = {}
    f["r1"] = raw
    rcc = ac / _by_code(ac).shift(1) - 1.0
    roc = ac / ao - 1.0
    rco = ao / _by_code(ac).shift(1) - 1.0
    f["rev1"] = rcc
    f["rev_oc"] = roc
    f["rev_co"] = rco
    f["mret"] = mc / mo - 1.0
    f["aret"] = nc / no - 1.0
    for window in (3, 5, 10, 20, 60, 120, 250):
        f[f"rcc{window}"] = _rolling(rcc, window, max(2, int(window * 0.7)), "sum")
    f["mom"] = f["rcc250"] - f["rcc20"]
    f["oc20"] = _rolling(roc, 20, 12, "sum")
    f["co20"] = _rolling(rco, 20, 12, "sum")
    f["vol20"] = _rolling(raw, 20, 15, "std")
    f["vol60"] = _rolling(raw, 60, 45, "std")
    f["dvol60"] = _rolling(raw.clip(upper=0.0), 60, 45, "std")
    f["range20"] = _rolling((ah - al) / ac, 20, 15, "mean")
    f["range1"] = (ah - al) / ac
    f["clv1"] = ((ac - al) / (ah - al).replace(0, np.nan)).clip(0, 1)

    log_turn = np.log(turn.where(turn > 0))
    f["logturn20"] = _rolling(log_turn, 20, 15, "mean")
    f["logturn60"] = _rolling(log_turn, 60, 45, "mean")
    illiq = raw.abs() / turn.replace(0, np.nan)
    f["illiq20"] = _rolling(illiq, 20, 15, "mean")
    f["illiq60"] = _rolling(illiq, 60, 45, "mean")
    traded = px["Volume"].where(px["Volume"] > 0)
    f["volratio"] = traded / _rolling(traded, 20, 15, "mean")

    adj_ratio = (px["AdjustmentClose"] / px["Close"]).rename("_adj_ratio")
    del px
    fin = _asof_fins(panel_index, FINS_COLUMNS, adj_ratio)

    # 1株当たり財務値は開示時点の株式数基準なので、開示日の調整比率 `_rfactor` を
    # 掛けて調整後価格と同じ基準へそろえる。株数は割って同じ基準にする。
    # （探索Notebookの注意どおり、基準を混ぜると分割・併合倍率だけ指標がずれる）
    r_factor = fin["_rfactor"]
    price = ac  # 調整後終値（生成時点基準）
    shares = (fin[SHARES_COLUMN] / r_factor).where(fin[SHARES_COLUMN] > 0)
    equity = fin["Equity"]
    market_cap = price * shares
    f["logsize"] = np.log(market_cap.where(market_cap > 0))
    f["bp"] = (equity / shares) / price
    f["ep_f"] = fin["ForecastEarningsPerShare"] * r_factor / price
    f["ep_a"] = fin["EarningsPerShare"] * r_factor / price
    f["sp"] = fin["NetSales"] / shares / price
    f["cfy"] = fin["CashFlowsFromOperatingActivities"] / market_cap
    f["div_y"] = fin["ForecastDividendPerShareAnnual"] * r_factor / price
    f["roe"] = fin["Profit"] / equity
    f["cfo_ta"] = fin["CashFlowsFromOperatingActivities"] / fin["TotalAssets"]
    f["eq_ratio"] = fin["EquityToAssetRatio"]
    f["op_margin"] = fin["OperatingProfit"] / fin["NetSales"]
    f["prof_margin"] = fin["Profit"] / fin["NetSales"]
    # フローは期間累計値なので、前年同時期（約250営業日前）との比で伸び率にする。
    # 1株当たり値（EPS）は基準をそろえてから比べる。
    f["sales_yoy"] = fin["NetSales"] / _by_code(fin["NetSales"]).shift(250) - 1.0
    f["profit_yoy"] = fin["Profit"] / _by_code(fin["Profit"]).shift(250) - 1.0
    eps_fwd = fin["ForecastEarningsPerShare"] * r_factor
    f["eps_fwd_chg"] = eps_fwd - _by_code(eps_fwd).shift(125)
    f["eps_fwd_yoy"] = eps_fwd / _by_code(eps_fwd).shift(250) - 1.0
    f["cf_yoy"] = (
        fin["CashFlowsFromOperatingActivities"]
        / _by_code(fin["CashFlowsFromOperatingActivities"]).shift(250)
        - 1.0
    )
    days = (panel_index.get_level_values("Date") - pd.DatetimeIndex(fin["_fdate"])).days
    f["disc_days"] = pd.Series(days, index=panel_index).astype(float)

    listed = _read_splits(
        "listed_info", splits, columns=LISTED_COLUMNS, start=start, codes=codes
    )
    listed.index = listed.index.set_levels(listed.index.levels[1].astype(str), level=1)
    listed = listed.reindex(panel_index)
    f["scale"] = listed["ScaleCategory"].map(SCALE_RANK)
    f["sector17"] = pd.to_numeric(listed["Sector17Code"], errors="coerce")
    f["sector33"] = pd.to_numeric(listed["Sector33Code"], errors="coerce")
    f["market"] = pd.to_numeric(listed["MarketCode"], errors="coerce")
    f["margin"] = pd.to_numeric(listed["MarginCode"], errors="coerce")

    beta = _read_splits("beta_1day", splits, columns=["Return"], start=start)["Return"]
    f["beta"] = beta.reindex(panel_index)

    del fin, listed, beta
    # float64 のまま全列を DataFrame 化してから astype するとピークが倍近くなるため、
    # 列ごとに float32 化してから組み立てる（丸め結果は全列 astype と同一）。
    for key in f:
        f[key] = f[key].astype(np.float32)
    features = pd.DataFrame(f, index=panel_index)
    del f
    return features


def xsec_rank(values: pd.Series) -> pd.Series:
    """同じ日付の断面だけで 0〜1 に順位化する（別日付を混ぜない）。"""
    return (values.groupby(level="Date").rank(pct=True) - 0.5).astype(np.float32)


def xsec_z(values: pd.Series) -> pd.Series:
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0).astype(np.float32)


def rank_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """モデル用に、営業日ごとの断面順位へ変換する（in-place、float32）。

    呼び出し側の DataFrame をそのまま書き換える。ランク化後の値だけが必要な
    場合に使い、生の特徴量が必要な処理は先に済ませること。
    """
    for column in features.columns:
        if column in SECTOR_COLUMNS:
            features[column] = features[column].astype(np.float32)
        else:
            features[column] = xsec_rank(features[column])
    return features


def signed_block(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    """ブロック内の符号付き順位を等ウェイト平均する（欠損は無視）。"""
    ranks = pd.DataFrame(
        {name: FACTOR_SIGNS[name] * xsec_rank(features[name]) for name in columns}
    )
    return ranks.mean(axis=1, skipna=True).astype(np.float32)
