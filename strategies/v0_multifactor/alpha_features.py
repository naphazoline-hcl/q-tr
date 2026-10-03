"""alpha_features v2: 金融仮説に基づく特徴量ライブラリ（未来を見ない）。

v1（rf_features.py, 51列）の列名・意味を維持したまま、末尾に新規列を追加する。
`build_features(splits, start, end)` は (Date, Code) index の float32 パネルを返す。

時点管理のルール（採点の禁止事項に対応）
----------------------------------------
- rolling は過去方向のみ、shift は正の方向のみ、未来値での穴埋め（bfill 等）はしない
- 決算短信は開示日基準の backward as-of（Date は tz-aware なので naive JST にそろえる）
- マクロ（ECB 為替・消費者態度指数）は Date（日本時間の利用可能日時）で backward as-of。
  ObservationDate では結合しない。パネル行の時刻は 00:00 とみなすため、
  当日 23:00 や 23:59:59 に利用可能になった値は翌営業日の行から使われる（保守的）
- 順位化は同一 Date の断面のみ（size_in_sector33）。時系列 z は過去方向の窓のみ
- end を渡すと、全入力を Date <= end で読み込み時に絞る（因果性検査で使う）

財務の基準そろえ
----------------
AdjustmentClose は配布時点の株式数基準、決算短信の1株当たり値は開示時点の基準。
R = AdjustmentClose / Close を開示日に as-of で持ち（`_rfactor`）、
1株当たり値 x R、株式数 / R と換算してから価格と比べる（v1 と同じ方式）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SHARES_COLUMN = (
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"
)

# v1 で as-of 展開していた列（意味を変えないため、この列は v1 と同じ前方埋めをする）
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

# v2 で追加で読む列（期間の識別・同日開示の順序・翌期予想）
FINS_KEY_COLUMNS = [
    "DisclosureNumber",
    "CurrentPeriodStartDate",
    "CurrentPeriodEndDate",
    "CurrentFiscalYearStartDate",
    "NextYearForecastEarningsPerShare",
    "NextYearForecastOperatingProfit",
]

# TTM 化するフロー項目（列名 -> 出力の短縮名）
TTM_ITEMS = {
    "NetSales": "sales_ttm",
    "OperatingProfit": "op_ttm",
    "Profit": "profit_ttm",
    "CashFlowsFromOperatingActivities": "cfo_ttm",
}

# 開示行で計算して日次へ as-of 展開する派生列
FINS_DERIVED = list(TTM_ITEMS.values()) + [
    "eqr_chg", "opm_chg", "eps_rev", "op_rev", "progress",
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

V1_COLUMNS = [
    "r1", "rev1", "rev_oc", "rev_co", "mret", "aret",
    "rcc3", "rcc5", "rcc10", "rcc20", "rcc60", "rcc120", "rcc250",
    "mom", "oc20", "co20", "vol20", "vol60", "dvol60", "range20", "range1", "clv1",
    "logturn20", "logturn60", "illiq20", "illiq60", "volratio",
    "logsize", "bp", "ep_f", "ep_a", "sp", "cfy", "div_y", "roe", "cfo_ta", "eq_ratio",
    "op_margin", "prof_margin", "sales_yoy", "profit_yoy", "eps_fwd_chg", "eps_fwd_yoy",
    "cf_yoy", "disc_days", "scale", "sector17", "sector33", "market", "margin", "beta",
]

# 新規列（v1 の後ろにこの順で並ぶ）
FEATURE_GROUPS = {
    "price_reversal": [
        "hi250", "lo250", "res20", "res60", "res250", "lrev120_20", "lrev250_60",
        "gap20", "ac1_60", "skew20", "kurt20", "updays60",
    ],
    "risk_liquidity": [
        "vol5_20", "uvol60", "dvol_uvol60", "turn_trend", "illiq5", "volume_z20",
        "highlow60", "valid_ratio20", "size_in_sector33",
    ],
    "value": [
        "ep_ttm", "sp_ttm", "cfy_ttm", "roe_ttm", "op_margin_ttm", "accrual", "ep_gap",
    ],
    "quality_growth": [
        "equity_ratio_chg", "op_margin_yoy_chg", "sales_yoy_acc", "div_chg",
    ],
    "revision": [
        "eps_revision", "opprofit_revision", "progress_1q", "disclosed_5d", "disc_days_bd",
    ],
    "regime": [
        "usdjpy_chg20", "usdjpy_chg60", "consumer_level_z", "consumer_diff",
        "topix_cum20", "topix_vol60",
    ],
}

NEW_COLUMNS = [name for names in FEATURE_GROUPS.values() for name in names]
ALL_COLUMNS = V1_COLUMNS + NEW_COLUMNS

# v1 互換（signed_block が参照する）
FACTOR_SIGNS = {
    "logsize": -1.0, "illiq60": +1.0, "logturn60": -1.0,
    "bp": +1.0, "sp": +1.0, "ep_f": +1.0, "ep_a": +1.0, "cfy": +1.0, "div_y": +1.0,
    "roe": +1.0, "cfo_ta": +1.0, "eq_ratio": +1.0,
    "beta": -1.0, "vol60": -1.0,
}

FACTOR_BLOCKS = {
    "size": ["logsize", "illiq60", "logturn60"],
    "value": ["bp", "sp", "ep_f", "ep_a", "cfy", "div_y"],
    "quality": ["roe", "cfo_ta", "eq_ratio"],
    "lowrisk": ["beta", "vol60"],
}


# ---------------------------------------------------------------- IO helpers


def _to_ts(value) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def _read_splits(
    name: str,
    splits: tuple[str, ...],
    columns: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """parquet を読む。start / end / codes は pyarrow のフィルタとして読込前に適用する。"""
    filters = []
    if start is not None:
        filters.append(("Date", ">=", start))
    if end is not None:
        filters.append(("Date", "<=", end))
    if codes is not None:
        filters.append(("Code", "in", codes))
    frames = [
        pd.read_parquet(f"{name}_{split}.parquet", columns=columns, filters=filters or None)
        for split in splits
    ]
    return pd.concat(frames).sort_index()


def _naive_jst(values: pd.Series) -> pd.Series:
    """tz-aware の日時を日本時間の naive datetime64[ns] にそろえる（merge_asof 用）。"""
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return values.astype("datetime64[ns]")


def _by_code(values: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    return values.groupby(level="Code", sort=False)


def _rolling(values: pd.Series, window: int, min_periods: int | None = None, how: str = "mean") -> pd.Series:
    if min_periods is None:
        min_periods = max(2, int(window * 0.75))
    return _by_code(values).transform(
        lambda x: x.rolling(window, min_periods=min_periods).agg(how)
    )


def _rolling_autocorr(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    """銘柄ごとに corr(x_t, x_{t-1}) を過去 window 日で計算する。"""
    return _by_code(values).transform(
        lambda x: x.rolling(window, min_periods=min_periods).corr(x.shift(1))
    )


def _f32(values) -> pd.Series:
    return values.astype(np.float32)


# ------------------------------------------------------------ feature blocks
# 以下は骨格。各ブロックは dict f に float32 の Series を追加する。


def _price_block(f: dict, px: pd.DataFrame, raw: pd.Series, beta: pd.Series, topix_t: np.ndarray) -> None:
    """A. 価格・リターン（v1 の価格列 + 新規 price_reversal / ボラ系）。

    v1 列は v1 と同じ式・同じ min_periods。差分系の列は float64 のまま計算してから
    float32 にする（v1 と丸め結果を一致させるため）。
    """
    ao, ah, al, ac = (px[c] for c in ("AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose"))
    mo, mc = px["MorningAdjustmentOpen"], px["MorningAdjustmentClose"]
    no, nc = px["AfternoonAdjustmentOpen"], px["AfternoonAdjustmentClose"]
    prev_ac = _by_code(ac).shift(1)

    f["r1"] = _f32(raw)
    rcc = ac / prev_ac - 1.0
    roc = ac / ao - 1.0
    rco = ao / prev_ac - 1.0
    f["rev1"], f["rev_oc"], f["rev_co"] = _f32(rcc), _f32(roc), _f32(rco)
    f["mret"] = _f32(mc / mo - 1.0)
    f["aret"] = _f32(nc / no - 1.0)
    cum = {}
    for window in (3, 5, 10, 20, 60, 120, 250):
        cum[window] = _rolling(rcc, window, max(2, int(window * 0.7)), "sum")
        f[f"rcc{window}"] = _f32(cum[window])
    f["mom"] = _f32(cum[250] - cum[20])
    f["oc20"] = _f32(_rolling(roc, 20, 12, "sum"))
    f["co20"] = _f32(_rolling(rco, 20, 12, "sum"))
    vol20 = _rolling(raw, 20, 15, "std")
    dvol60 = _rolling(raw.clip(upper=0.0), 60, 45, "std")
    f["vol20"] = _f32(vol20)
    f["vol60"] = _f32(_rolling(raw, 60, 45, "std"))
    f["dvol60"] = _f32(dvol60)
    f["range20"] = _f32(_rolling((ah - al) / ac, 20, 15, "mean"))
    f["range1"] = _f32((ah - al) / ac)
    f["clv1"] = _f32(((ac - al) / (ah - al).replace(0, np.nan)).clip(0, 1))

    # --- new: price_reversal
    # 52週高値・安値からの距離（アンカー効果・高値更新モメンタム）
    f["hi250"] = _f32(ac / _rolling(ac, 250, 175, "max") - 1.0)
    f["lo250"] = _f32(ac / _rolling(ac, 250, 175, "min") - 1.0)
    # 市場中立化した過去リターン（beta と topix は配布値で、どちらも t 時点で確定済み）
    resid = raw - beta * topix_t
    for window in (20, 60, 250):
        f[f"res{window}"] = _f32(_rolling(resid, window, max(2, int(window * 0.7)), "sum"))
    # 長期リバーサル（直近の短期部分を除いた過去リターン）
    f["lrev120_20"] = _f32(cum[120] - cum[20])
    f["lrev250_60"] = _f32(cum[250] - cum[60])
    # rev_co（前日終値→当日始値）がギャップと同義なので、その20日平均を追加
    f["gap20"] = _f32(_rolling(rco, 20, 12, "mean"))
    f["ac1_60"] = _f32(_rolling_autocorr(raw, 60, 45))
    f["skew20"] = _f32(_rolling(raw, 20, 15, "skew"))
    f["kurt20"] = _f32(_rolling(raw, 20, 15, "kurt"))
    up = (raw > 0).astype(np.float64).where(raw.notna())
    f["updays60"] = _f32(_rolling(up, 60, 45, "mean"))

    # --- new: volatility shape (risk_liquidity)
    vol5 = _rolling(raw, 5, 4, "std")
    f["vol5_20"] = _f32(vol5 / vol20.replace(0.0, np.nan))
    uvol60 = _rolling(raw.clip(lower=0.0), 60, 45, "std")
    f["uvol60"] = _f32(uvol60)
    f["dvol_uvol60"] = _f32(dvol60 / uvol60.replace(0.0, np.nan))


def _liquidity_block(f: dict, px: pd.DataFrame, raw: pd.Series) -> None:
    """B. 流動性・需給（v1 の流動性列 + 新規）。"""
    ah, al, ac = px["AdjustmentHigh"], px["AdjustmentLow"], px["AdjustmentClose"]
    turn = px["TurnoverValue"]

    log_turn = np.log(turn.where(turn > 0))
    logturn20 = _rolling(log_turn, 20, 15, "mean")
    logturn60 = _rolling(log_turn, 60, 45, "mean")
    f["logturn20"], f["logturn60"] = _f32(logturn20), _f32(logturn60)
    illiq = raw.abs() / turn.replace(0, np.nan)
    f["illiq20"] = _f32(_rolling(illiq, 20, 15, "mean"))
    f["illiq60"] = _f32(_rolling(illiq, 60, 45, "mean"))
    traded = px["Volume"].where(px["Volume"] > 0)
    f["volratio"] = _f32(traded / _rolling(traded, 20, 15, "mean"))

    # --- new
    f["turn_trend"] = _f32(logturn20 - logturn60)
    f["illiq5"] = _f32(_rolling(illiq, 5, 4, "mean"))
    # 出来高の急増（対数出来高の20日 z。窓は t を含む過去20日）
    log_vol = np.log(traded)
    mean20 = _rolling(log_vol, 20, 15, "mean")
    std20 = _rolling(log_vol, 20, 15, "std")
    f["volume_z20"] = _f32((log_vol - mean20) / std20.replace(0.0, np.nan))
    f["highlow60"] = _f32(_rolling((ah - al) / ac, 60, 45, "mean"))
    # 売買停止・欠損の代理（価格が NaN でない行の比率）
    f["valid_ratio20"] = _f32(_rolling(ac.notna().astype(np.float64), 20, 1, "mean"))


def _carry(codes: pd.Series, values: np.ndarray, event: np.ndarray) -> np.ndarray:
    """event 行の値（NaN を含む）を、同一銘柄の次の event 行まで前方へ持ち越す。"""
    pos = pd.Series(np.where(event, np.arange(len(values), dtype=np.float64), np.nan))
    src = pos.groupby(codes.to_numpy(), sort=False).ffill().to_numpy()
    out = np.full(len(values), np.nan)
    ok = ~np.isnan(src)
    out[ok] = values[src[ok].astype(np.int64)]
    return out


def _lookup(keys: pd.DataFrame, src: pd.DataFrame, cols: list[str], fy_shift: int, length: int | None = None) -> pd.DataFrame:
    """各開示行について、開示日以前で最新の「会計年度キー - fy_shift・同じ期間長」の行を引く。

    期間長 length を指定すると通期（12か月）など固定の期間を引く。backward as-of なので
    その時点で未開示の行は使わない（前年同期・前年通期の参照に使う）。
    """
    q = keys[["Date", "Code", "_fyk", "_len"]].copy()
    q["_fyk"] = q["_fyk"] - fy_shift
    if length is not None:
        q["_len"] = np.int64(length)
    hit = pd.merge_asof(
        q, src[["Date", "Code", "_fyk", "_len"] + cols],
        on="Date", by=["Code", "_fyk", "_len"], direction="backward",
    )
    return hit[cols].reset_index(drop=True)


def _add_ttm(fins: pd.DataFrame) -> None:
    """四半期累計値から TTM を作る: 今回累計 + 前年通期 - 前年同期累計（通期行は通期値）。

    前年同期または前年通期が無い行（データ開始直後・決算期変更など）は、
    その時点で最新の通期実績で代用する。
    """
    keys = fins[["Date", "Code", "_fyk", "_len"]]
    quarterly = fins["_len"].isin([3, 6, 9]).to_numpy()
    annual_row = (fins["_len"] == 12).to_numpy()
    for item, out in TTM_ITEMS.items():
        src = fins.loc[fins[item].notna()]
        same = _lookup(keys, src, [item], 12)[item].to_numpy()
        prev_fy = _lookup(keys, src, [item], 12, length=12)[item].to_numpy()
        cur = fins[item].to_numpy()
        ttm = np.where(annual_row, cur, np.where(quarterly, cur + prev_fy - same, np.nan))
        annual_ff = fins[item].where(annual_row).groupby(fins["Code"], sort=False).ffill().to_numpy()
        fallback = np.isnan(ttm) & ~np.isnan(cur)
        ttm[fallback] = annual_ff[fallback]
        fins[out] = ttm


def _add_yoy_changes(fins: pd.DataFrame) -> None:
    """自己資本比率・営業利益率（累計ベース）の前年同期差。"""
    keys = fins[["Date", "Code", "_fyk", "_len"]]
    cols = ["EquityToAssetRatio", "OperatingProfit", "NetSales"]
    prev = _lookup(keys, fins.loc[fins["NetSales"].notna()], cols, 12)
    fins["eqr_chg"] = fins["EquityToAssetRatio"].to_numpy() - prev["EquityToAssetRatio"].to_numpy()
    opm = fins["OperatingProfit"] / fins["NetSales"].where(fins["NetSales"] > 0)
    opm_prev = prev["OperatingProfit"] / prev["NetSales"].where(prev["NetSales"] > 0)
    fins["opm_chg"] = opm.to_numpy() - opm_prev.to_numpy()


def _revision(fins: pd.DataFrame, cur_col: str, next_col: str, scale: pd.Series | None) -> np.ndarray:
    """同じ対象年度の予想について、1つ前の開示からの改訂率 (new - prev) / |prev|。

    通期決算の行は翌期予想（NextYear*）を「翌年度が対象」として扱い、それ以外は当期予想。
    対象年度が変わった最初の予想（期初ガイダンス）は改訂ではないので NaN。
    予想が無い開示行は直前の予想行の値を持ち越す（as-of 展開で最新の改訂が残る）。
    """
    use_next = (fins["_len"] == 12) & fins["NetSales"].notna() & fins[next_col].notna()
    guide = fins[cur_col].where(~use_next, fins[next_col])
    if scale is not None:
        guide = guide * scale
    has = guide.notna().to_numpy()
    sub = pd.DataFrame({
        "Code": fins["Code"],
        "tk": fins["_fyk"] + np.where(use_next, 12, 0),
        "g": guide,
    }).loc[has]
    prev = sub.groupby(["Code", "tk"], sort=False)["g"].shift(1)
    rev = np.full(len(fins), np.nan)
    rev[has] = ((sub["g"] - prev) / prev.abs().replace(0.0, np.nan)).to_numpy()
    return _carry(fins["Code"], rev, has)


def _month_key(values: pd.Series) -> pd.Series:
    return (values.dt.year * 12 + values.dt.month).fillna(0).astype(np.int64)


def _load_fins(end: pd.Timestamp | None, adj_ratio: pd.Series) -> pd.DataFrame:
    """決算短信を開示行のまま読み、TTM・前年同期差・改訂率などを開示行で計算する。

    同日に複数の開示がある銘柄の順序を DisclosureNumber で固定する（as-of の結果を
    入力範囲に依存させないため）。v1 列は v1 と同じく銘柄内で前方埋めする。
    """
    fins = _read_splits(
        "fins_statements", ("train", "valid"), columns=FINS_COLUMNS + FINS_KEY_COLUMNS
    ).reset_index()
    fins["Date"] = _naive_jst(fins["Date"])
    fins["Code"] = fins["Code"].astype(str)
    if end is not None:
        fins = fins.loc[fins["Date"] <= end]
    fins = fins.sort_values(["Date", "DisclosureNumber"], kind="stable").reset_index(drop=True)
    fins["_fdate"] = fins["Date"]
    fins["_fyk"] = _month_key(fins["CurrentFiscalYearStartDate"])
    fins["_len"] = _month_key(fins["CurrentPeriodEndDate"]) - _month_key(fins["CurrentPeriodStartDate"]) + 1

    # 開示日の調整比率 R = AdjustmentClose / Close（v1 の _rfactor と同じ）
    ratio = adj_ratio.rename("_rfactor").reset_index()
    ratio["Code"] = ratio["Code"].astype(str)
    ratio["Date"] = ratio["Date"].astype("datetime64[ns]")
    ratio = ratio.dropna(subset=["_rfactor"]).sort_values("Date")
    fins = pd.merge_asof(fins, ratio, on="Date", by="Code", direction="backward")
    fins["_rfactor"] = fins.groupby("Code", sort=False)["_rfactor"].ffill()

    # 前方埋めの前（実績行の生の値）で派生列を作る
    _add_ttm(fins)
    _add_yoy_changes(fins)
    fins["eps_rev"] = _revision(
        fins, "ForecastEarningsPerShare", "NextYearForecastEarningsPerShare", fins["_rfactor"]
    )
    fins["op_rev"] = _revision(fins, "ForecastOperatingProfit", "NextYearForecastOperatingProfit", None)
    len_act = fins["_len"].where(fins["NetSales"].notna()).groupby(fins["Code"], sort=False).ffill()

    ff_cols = FINS_COLUMNS + list(TTM_ITEMS.values()) + ["eqr_chg", "opm_chg"]
    fins[ff_cols] = fins.groupby("Code", sort=False)[ff_cols].ffill()
    # 1Q 進捗率: 最新の実績が 1Q（3か月累計）のときだけ、最新の通期予想利益に対する比率
    fcst = fins["ForecastProfit"]
    fins["progress"] = (fins["Profit"] / fcst.where(fcst > 0)).where(len_act == 3)
    return fins


def _asof_fins(panel_index: pd.MultiIndex, adj_ratio: pd.Series, end: pd.Timestamp | None) -> pd.DataFrame:
    """開示行の値を開示日基準の backward as-of で日次パネルへ展開する（v1 と同じく float32）。"""
    fins = _load_fins(end, adj_ratio)
    out_cols = FINS_COLUMNS + FINS_DERIVED + ["_rfactor"]
    base = panel_index.to_frame(index=False)
    base["Code"] = base["Code"].astype(str)
    base["Date"] = base["Date"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        base, fins[["Date", "Code", "_fdate"] + out_cols], on="Date", by="Code", direction="backward"
    )
    out = pd.DataFrame(
        {c: merged[c].to_numpy(dtype=np.float32) for c in out_cols}, index=panel_index
    )
    out["_fdate"] = pd.Series(pd.to_datetime(merged["_fdate"].to_numpy()), index=panel_index)
    return out


def _safe_div(num, den):
    return num / den.where(den > 0)


def _fins_block(f: dict, fin: pd.DataFrame, ac: pd.Series, panel_index: pd.MultiIndex) -> None:
    """C/D/E. v1 の財務列（式は v1 のまま）+ TTM バリュー・クオリティ成長・改訂イベント。"""
    r_factor = fin["_rfactor"]
    price = ac
    shares = (fin[SHARES_COLUMN] / r_factor).where(fin[SHARES_COLUMN] > 0)
    equity = fin["Equity"]
    market_cap = price * shares
    cfo = fin["CashFlowsFromOperatingActivities"]
    f["logsize"] = _f32(np.log(market_cap.where(market_cap > 0)))
    f["bp"] = _f32((equity / shares) / price)
    ep_f = fin["ForecastEarningsPerShare"] * r_factor / price
    ep_a = fin["EarningsPerShare"] * r_factor / price
    f["ep_f"], f["ep_a"] = _f32(ep_f), _f32(ep_a)
    f["sp"] = _f32(fin["NetSales"] / shares / price)
    f["cfy"] = _f32(cfo / market_cap)
    f["div_y"] = _f32(fin["ForecastDividendPerShareAnnual"] * r_factor / price)
    f["roe"] = _f32(fin["Profit"] / equity)
    f["cfo_ta"] = _f32(cfo / fin["TotalAssets"])
    f["eq_ratio"] = _f32(fin["EquityToAssetRatio"])
    f["op_margin"] = _f32(fin["OperatingProfit"] / fin["NetSales"])
    f["prof_margin"] = _f32(fin["Profit"] / fin["NetSales"])
    sales_yoy = fin["NetSales"] / _by_code(fin["NetSales"]).shift(250) - 1.0
    f["sales_yoy"] = _f32(sales_yoy)
    f["profit_yoy"] = _f32(fin["Profit"] / _by_code(fin["Profit"]).shift(250) - 1.0)
    eps_fwd = fin["ForecastEarningsPerShare"] * r_factor
    f["eps_fwd_chg"] = _f32(eps_fwd - _by_code(eps_fwd).shift(125))
    f["eps_fwd_yoy"] = _f32(eps_fwd / _by_code(eps_fwd).shift(250) - 1.0)
    f["cf_yoy"] = _f32(cfo / _by_code(cfo).shift(250) - 1.0)
    dates = panel_index.get_level_values("Date")
    fdate = pd.DatetimeIndex(fin["_fdate"])
    f["disc_days"] = _f32(pd.Series((dates - fdate).days, index=panel_index).astype(float))

    # --- new: value（TTM 基準。時価総額は v1 と同じ基準そろえ済みの株数 x 調整後終値）
    f["ep_ttm"] = _f32(fin["profit_ttm"] / market_cap)
    f["sp_ttm"] = _f32(fin["sales_ttm"] / market_cap)
    f["cfy_ttm"] = _f32(fin["cfo_ttm"] / market_cap)
    f["roe_ttm"] = _f32(_safe_div(fin["profit_ttm"], equity))
    f["op_margin_ttm"] = _f32(_safe_div(fin["op_ttm"], fin["sales_ttm"]))
    # 発生主義利益の割合（大きいほど利益の質が低い）
    f["accrual"] = _f32(_safe_div(fin["profit_ttm"] - fin["cfo_ttm"], fin["TotalAssets"]))
    f["ep_gap"] = _f32(ep_f - ep_a)

    # --- new: quality_growth
    f["equity_ratio_chg"] = _f32(fin["eqr_chg"])
    f["op_margin_yoy_chg"] = _f32(fin["opm_chg"])
    f["sales_yoy_acc"] = _f32(sales_yoy - _by_code(sales_yoy).shift(250))
    dps = fin["ForecastDividendPerShareAnnual"] * r_factor
    f["div_chg"] = _f32(dps / _by_code(dps).shift(250).where(lambda x: x > 0) - 1.0)

    # --- new: revision / events
    f["eps_revision"] = _f32(fin["eps_rev"])
    f["opprofit_revision"] = _f32(fin["op_rev"])
    f["progress_1q"] = _f32(fin["progress"])
    # 開示からの経過営業日（パネルの取引日カレンダーで数える。開示日が休日なら翌営業日起点）
    calendar = pd.DatetimeIndex(np.unique(dates.to_numpy()))
    pos_t = calendar.searchsorted(dates)
    pos_f = calendar.searchsorted(fdate, side="left")
    unknown = fdate.isna() | (fdate < calendar[0])
    bd = np.where(unknown, np.nan, (pos_t - pos_f).astype(np.float64))
    f["disc_days_bd"] = pd.Series(bd, index=panel_index, dtype=np.float32)
    f["disclosed_5d"] = pd.Series(np.where(unknown, np.nan, (bd < 5).astype(np.float64)), index=panel_index, dtype=np.float32)


def _listed_block(f: dict, panel_index, splits, start, end, codes) -> None:
    """属性（v1、同日付結合）と業種内サイズ順位（同一 Date・同一 sector33 の断面順位）。"""
    listed = _read_splits(
        "listed_info", splits, columns=LISTED_COLUMNS, start=start, end=end, codes=codes
    )
    listed.index = listed.index.set_levels(listed.index.levels[1].astype(str), level=1)
    listed = listed.reindex(panel_index)
    f["scale"] = _f32(listed["ScaleCategory"].map(SCALE_RANK))
    for name, column in (("sector17", "Sector17Code"), ("sector33", "Sector33Code"),
                         ("market", "MarketCode"), ("margin", "MarginCode")):
        f[name] = _f32(pd.to_numeric(listed[column], errors="coerce"))
    del listed
    dates = panel_index.get_level_values("Date")
    by_day_sector = f["logsize"].groupby([dates, f["sector33"].to_numpy()])
    f["size_in_sector33"] = _f32(by_day_sector.rank(pct=True))


def _macro_asof(day: pd.DataFrame, table: pd.DataFrame, end: pd.Timestamp | None, columns: list[str]) -> pd.DataFrame:
    """マクロ表を Date（日本時間の利用可能日時）で取引日へ backward as-of する。"""
    table = table.copy()
    table["Date"] = _naive_jst(table["Date"])
    if end is not None:
        table = table.loc[table["Date"] <= end]
    table = table.sort_values("Date", kind="stable")
    return pd.merge_asof(day, table[["Date"] + columns], on="Date", direction="backward")


def _macro_block(f: dict, panel_index: pd.MultiIndex, splits, end, topix: pd.Series) -> None:
    """F. マクロ（Date で backward as-of）と市場状態。全銘柄に同じ値を配る。

    パネル行は取引日の 00:00 とみなすので、当日夜に利用可能になった値は翌営業日から使う。
    変化率・窓統計は取引日カレンダー上で過去方向にのみ計算する。
    """
    dates = panel_index.get_level_values("Date")
    calendar = pd.DatetimeIndex(np.unique(dates.to_numpy()))
    day = pd.DataFrame({"Date": calendar.astype("datetime64[ns]")})
    daily: dict[str, np.ndarray] = {}

    fx = _read_splits("ecb_fx_rates", splits, columns=["Rate"]).reset_index()
    fx = fx.loc[fx["Pair"].astype(str) == "USD/JPY", ["Date", "Rate"]]
    usd = _macro_asof(day, fx, end, ["Rate"])["Rate"]
    daily["usdjpy_chg20"] = (usd / usd.shift(20) - 1.0).to_numpy()
    daily["usdjpy_chg60"] = (usd / usd.shift(60) - 1.0).to_numpy()

    ci = _read_splits("consumer_attitude_index", splits, columns=["ConsumerAttitudeIndex"]).reset_index()
    ci["Date"] = _naive_jst(ci["Date"])
    if end is not None:
        ci = ci.loc[ci["Date"] <= end]
    ci = ci.sort_values("Date", kind="stable").reset_index(drop=True)
    level = ci["ConsumerAttitudeIndex"]
    window = level.rolling(60, min_periods=24)  # 月次 60 本 = 過去5年
    ci["level_z"] = (level - window.mean()) / window.std()
    ci["diff"] = level.diff()
    cons = _macro_asof(day, ci, end, ["level_z", "diff"])
    daily["consumer_level_z"] = cons["level_z"].to_numpy()
    daily["consumer_diff"] = cons["diff"].to_numpy()

    tpx = topix.reindex(calendar)
    daily["topix_cum20"] = tpx.rolling(20, min_periods=15).sum().to_numpy()
    daily["topix_vol60"] = tpx.rolling(60, min_periods=45).std().to_numpy()

    pos = calendar.searchsorted(dates)
    for name, values in daily.items():
        f[name] = pd.Series(np.asarray(values, dtype=np.float64)[pos], index=panel_index, dtype=np.float32)


PRICE_COLUMNS = [
    "Close", "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose",
    "MorningAdjustmentOpen", "MorningAdjustmentClose",
    "AfternoonAdjustmentOpen", "AfternoonAdjustmentClose",
    "TurnoverValue", "Volume",
]


def _assemble(f: dict, panel_index: pd.MultiIndex) -> pd.DataFrame:
    """列を ALL_COLUMNS の順に 1 つの float32 配列へ詰める（dict を空けながら詰めてピークを抑える）。"""
    missing = [c for c in ALL_COLUMNS if c not in f]
    extra = [c for c in f if c not in ALL_COLUMNS]
    if missing or extra:
        raise RuntimeError(f"列定義と実装が不一致: missing={missing} extra={extra}")
    values = np.empty((len(panel_index), len(ALL_COLUMNS)), dtype=np.float32, order="F")
    for j, name in enumerate(ALL_COLUMNS):
        column = f.pop(name)
        if len(column) != len(panel_index):
            raise RuntimeError(f"{name}: 行数がパネルと一致しません")
        values[:, j] = column.to_numpy(dtype=np.float32)
    return pd.DataFrame(values, index=panel_index, columns=ALL_COLUMNS, copy=False)


def build_features(
    splits: tuple[str, ...] = ("train", "valid"),
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """(Date, Code) パネルの生特徴量（float32）を返す。ランク化は呼び出し側で行う。

    end を渡すと、全入力を Date <= end で読み込み時に絞るので、返り値も Date <= end のみ。
    """
    start, end = _to_ts(start), _to_ts(end)
    raw = _read_splits("raw_return_1day", splits, columns=["Return"], start=start, end=end)["Return"]
    panel_index = raw.index
    codes = sorted(panel_index.get_level_values("Code").unique().tolist())
    px = _read_splits("prices_daily_quotes", splits, columns=PRICE_COLUMNS, start=start, end=end)
    if not px.index.equals(panel_index):
        raise RuntimeError("prices と raw_return の index が一致しません")
    beta = _read_splits("beta_1day", splits, columns=["Return"], start=start, end=end)["Return"]
    beta = beta.reindex(panel_index)
    topix = _read_splits("topix_return_1day", splits, columns=["Return"], start=start, end=end)["Return"]
    topix_t = topix.reindex(panel_index.get_level_values("Date")).to_numpy()

    f: dict[str, pd.Series] = {}
    _price_block(f, px, raw, beta, topix_t)
    _liquidity_block(f, px, raw)
    ac = px["AdjustmentClose"].copy()
    adj_ratio = (px["AdjustmentClose"] / px["Close"]).rename("_adj_ratio")
    del px
    fin = _asof_fins(panel_index, adj_ratio, end)
    del adj_ratio
    _fins_block(f, fin, ac, panel_index)
    del fin, ac
    _listed_block(f, panel_index, splits, start, end, codes)
    f["beta"] = _f32(beta)
    del beta
    _macro_block(f, panel_index, splits, end, topix)
    return _assemble(f, panel_index)


# ------------------------------------------------------------ v1 互換 API


def xsec_rank(values: pd.Series) -> pd.Series:
    """同じ日付の断面だけで 0〜1 に順位化する（別日付を混ぜない）。"""
    return (values.groupby(level="Date").rank(pct=True) - 0.5).astype(np.float32)


def xsec_z(values: pd.Series) -> pd.Series:
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0).astype(np.float32)


def rank_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """モデル用に営業日ごとの断面順位へ変換する（in-place、float32。SECTOR_COLUMNS はコードのまま）。"""
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
