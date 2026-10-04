"""regime_v4.py — P5 B: causal 3-regime block mixing for alpha_v2 (default OFF: params["regime_mix"]).

Inputs are alpha_features' same-day macro / market columns (one value per Date):
  macro 1  consumer_level_z, consumer_diff  -> level and 63-day mean of the monthly change (trend)
  macro 2  usdjpy_chg60                     -> sharp yen appreciation flag
  market 1 topix_cum60 = cum20[t] + cum20[t-20] + cum20[t-40]  (positive shifts only; when the
           older windows are missing the available parts are rescaled to 60 days)
  market 2 topix_vol60 vs a quantile fitted on the training window only
Rule per Date (values known on that Date + training-window thresholds):
  risk_off  : vol_high and (consumer_trend < 0 or yen_strong or cum60 < cum_floor)
  expansion : not vol_high and consumer_level_z > consumer_z_min and cum60 > 0
  neutral   : otherwise (also when inputs are missing)
Then a causal debounce (a new regime must hold confirm_days in a row) and an EWMA of the one-hot
memberships (blend_span) so block weights glide between regimes instead of jumping (turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("risk_off", "neutral", "expansion")
INPUT_COLUMNS = ["consumer_level_z", "consumer_diff", "usdjpy_chg60", "topix_cum20", "topix_vol60"]

DEFAULT_REGIME_V4 = {
    "vol_quantile": 0.6,
    "consumer_trend_days": 63,
    "consumer_z_min": 0.0,
    "fx_strong_yen": -0.05,
    "cum_floor": -0.05,
    "confirm_days": 5,
    "blend_span": 20,
}
# Empty dict -> params["block_weights"]. Untested defaults (B is OFF by default).
DEFAULT_REGIME_WEIGHTS = {
    "risk_off": {"size": 1.0, "value": 0.4, "quality": 0.7, "lowrisk": 0.6},
    "neutral": {},
    "expansion": {"size": 1.0, "value": 0.6, "quality": 0.4, "lowrisk": 0.2},
}


def daily_inputs(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Date ごとの局面入力（同日値のみ。窓はすべて過去方向）。"""
    missing = [c for c in INPUT_COLUMNS if c not in features.columns]
    if missing:
        raise KeyError(f"regime_v4: columns not in features: {missing}")
    daily = features[INPUT_COLUMNS].groupby(level="Date").first().sort_index().astype(np.float64)
    cum20 = daily["topix_cum20"]
    parts = pd.concat([cum20, cum20.shift(20), cum20.shift(40)], axis=1)
    daily["topix_cum60"] = parts.sum(axis=1, min_count=1) * 3.0 / parts.notna().sum(axis=1).replace(0, np.nan)
    days = int(cfg["consumer_trend_days"])
    daily["consumer_trend"] = daily["consumer_diff"].rolling(days, min_periods=1).mean()
    return daily


def fit_thresholds(features: pd.DataFrame, cfg: dict) -> dict:
    """topix_vol60 のしきい値を学習期間の日次値だけで決める（推論時は固定値として使う）。"""
    if "topix_vol60" not in features.columns or len(features) == 0:
        return {"vol_threshold": float("nan")}
    daily = features["topix_vol60"].groupby(level="Date").first().dropna()
    value = float(daily.quantile(float(cfg["vol_quantile"]))) if len(daily) else float("nan")
    return {"vol_threshold": value}


def classify(daily: pd.DataFrame, thresholds: dict, cfg: dict) -> np.ndarray:
    """0=risk_off / 1=neutral / 2=expansion（欠損比較は False -> neutral）。"""
    threshold = float(thresholds.get("vol_threshold", float("nan")))
    with np.errstate(invalid="ignore"):
        vol_high = daily["topix_vol60"].to_numpy() > threshold
        vol_low = daily["topix_vol60"].to_numpy() <= threshold
        falling = daily["consumer_trend"].to_numpy() < 0.0
        yen_strong = daily["usdjpy_chg60"].to_numpy() < float(cfg["fx_strong_yen"])
        cum60 = daily["topix_cum60"].to_numpy()
        risk_off = vol_high & (falling | yen_strong | (cum60 < float(cfg["cum_floor"])))
        expansion = vol_low & (daily["consumer_level_z"].to_numpy() > float(cfg["consumer_z_min"])) & (cum60 > 0.0)
    return np.where(risk_off, 0, np.where(expansion, 2, 1)).astype(np.int8)


def debounce(raw: np.ndarray, confirm_days: int) -> np.ndarray:
    """新しい局面は confirm_days 日連続で出たときだけ採用（過去方向の逐次処理）。"""
    out = np.empty_like(raw)
    if len(raw) == 0:
        return out
    state, candidate, streak = raw[0], None, 0
    for i, value in enumerate(raw):
        if value == state:
            candidate, streak = None, 0
        else:
            streak = streak + 1 if value == candidate else 1
            candidate = value
            if streak >= max(1, int(confirm_days)):
                state, candidate, streak = value, None, 0
        out[i] = state
    return out


def memberships(features: pd.DataFrame, thresholds: dict, cfg: dict) -> pd.DataFrame:
    """Date x 3 局面の所属度（one-hot の EWMA。行和 1）。"""
    daily = daily_inputs(features, cfg)
    codes = debounce(classify(daily, thresholds, cfg), int(cfg["confirm_days"]))
    onehot = pd.DataFrame(np.eye(len(REGIMES))[codes], index=daily.index, columns=list(REGIMES))
    span = int(cfg["blend_span"])
    return onehot.ewm(span=span, min_periods=1).mean() if span > 1 else onehot


def block_weight_arrays(features: pd.DataFrame, thresholds: dict, cfg: dict,
                        regime_weights: dict, base_weights: dict, blocks: list[str]) -> dict[str, np.ndarray]:
    """行ごとのブロック重み w_b(t) = sum_r m_r(t) * W_r[b]（W_r に無いブロックは base_weights）。"""
    member = memberships(features, thresholds, cfg)
    pos = member.index.searchsorted(features.index.get_level_values("Date"))
    out = {}
    for block in blocks:
        per_regime = np.array([
            float((regime_weights.get(r) or {}).get(block, base_weights.get(block, 0.0))) for r in REGIMES
        ])
        out[block] = (member.to_numpy() @ per_regime)[pos]
    return out


def regime_share(features: pd.DataFrame, thresholds: dict, cfg: dict) -> dict[str, float]:
    """診断用: 各局面の平均所属度（ログ・REPORT 用）。"""
    member = memberships(features, thresholds, cfg)
    return {r: round(float(member[r].mean()), 4) for r in REGIMES} if len(member) else {}
