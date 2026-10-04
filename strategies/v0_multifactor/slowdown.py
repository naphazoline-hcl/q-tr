"""slowdown.py — P5 C: turnover_cap-driven slowing of the model term (labels never used here).

- SLOW_PROFILES bundle the per-code EWMA span of the model prediction (5 / 10 / 20) with the label
  design it is meant for (which horizons train the model, label_transform). "span10" == K1.
- quintile_turnover() reproduces the scorer's weights (rank(method="first") -> 5 buckets ->
  (q - 2) / n / 1.2) and its turnover sum|w_t - w_{t-1}| from the *signal alone*, so a turnover
  estimate needs no target data and is safe inside fit_model / walk-forward folds.
- choose_span(): with params["turnover_control"] == "auto", fit_model evaluates the candidate spans
  on the tail of the training window and keeps the fastest span whose estimated turnover is
  <= params["turnover_cap"] (if none qualifies, the slowest candidate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPAN_CHOICES = (5, 10, 20)

SLOW_PROFILES = {
    # span 5: shortest label only (fast signal, fast label)
    "span5": {"model_smoothing_span": 5, "horizons": ["k126"], "label_transform": "raw"},
    # span 10: K1 (both horizons)
    "span10": {"model_smoothing_span": 10, "horizons": ["k126", "k250"], "label_transform": "raw"},
    # span 20: longest label only (slowest)
    "span20": {"model_smoothing_span": 20, "horizons": ["k250"], "label_transform": "raw"},
}


def apply_profile(config: dict) -> dict:
    """params["slow_profile"]（None なら何もしない）を model_smoothing_span / label_transform に展開。

    profile_horizons は fit_model が「どのラベル列で学習するか」を絞るのに使う。
    """
    name = config.get("slow_profile")
    if not name:
        return config
    if name not in SLOW_PROFILES:
        raise ValueError(f"unknown slow_profile: {name!r} (choose from {sorted(SLOW_PROFILES)})")
    profile = SLOW_PROFILES[name]
    out = dict(config)
    out["model_smoothing_span"] = int(profile["model_smoothing_span"])
    out["label_transform"] = profile["label_transform"]
    out["profile_horizons"] = list(profile["horizons"])
    return out


def ewm_by_code(values: pd.Series, span: int) -> pd.Series:
    """銘柄ごとの EWMA（過去方向のみ。alpha_v2.smooth_by_code と同じ式）。"""
    if int(span or 1) <= 1:
        return values
    return values.groupby(level="Code", sort=False).transform(lambda x: x.ewm(span=int(span), min_periods=1).mean())


def quintile_weights(signal: pd.Series) -> pd.Series:
    """採点と同じ重み (q - 2) / n / 1.2（NaN は 0 扱い、rank(method="first") 後に 5 等分）。"""
    filled = signal.sort_index().fillna(0.0)
    rank = filled.groupby(level="Date").rank(method="first")
    count = rank.groupby(level="Date").transform("count")
    # pd.qcut(ranks 1..n, 5) edges are 1 + (n - 1) * k / 5 with right-closed bins (ranks are distinct).
    bucket = sum((rank > 1.0 + (count - 1.0) * k / 5.0).astype(np.float64) for k in range(1, 5))
    return (bucket - 2.0) / count / 1.2


def quintile_turnover(signal: pd.Series) -> float:
    """日次回転率 mean_t sum_i |w_it - w_i,t-1|（採点の delta と同じ。初日は fillna(weight)）。"""
    weight = quintile_weights(signal)
    delta = weight.groupby(level="Code").diff().abs().fillna(weight.abs())
    return float(delta.groupby(level="Date").sum().mean()) if len(delta) else float("nan")


def tail_rows(index: pd.Index, days: int) -> np.ndarray:
    """学習期間の末尾 days 営業日の行マスク（推定コストを抑える）。"""
    dates = index.get_level_values("Date")
    unique = np.unique(dates.to_numpy())
    if len(unique) <= days:
        return np.ones(len(index), dtype=bool)
    return np.asarray(dates >= unique[-days], dtype=bool)


def choose_span(estimates: dict[int, float], cap: float) -> int:
    """推定回転率 <= cap を満たす最速（最小）の span。無ければ最遅の候補。"""
    ok = [span for span, value in sorted(estimates.items()) if np.isfinite(value) and value <= cap]
    return int(ok[0]) if ok else int(max(estimates))
