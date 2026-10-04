"""blend.py — P8 A: learned composition weights for alpha_v2 (default OFF, params["blend_learning"]).

The manual weights of the signal

    signal = sum_b w_b z(block_b) + w_model z(p) + w_sector z(p - sector mean),  p = EWMA(combine(models, v))

are re-estimated inside fit_model from the training window only (labels <= train_end):

- "blocks"  : w_b over the blocks whose manual weight is non-zero. Blocks have no fitted parameters, so the
              whole training window is an honest sample. Learned weights are rescaled to the manual L1 total
              (the block-vs-model balance is kept) and mixed with the manual ones by `shrink`.
- "ensemble": v over the active ensemble members (lgbm / ridge / linear) on an inner holdout (the last
              `holdout_frac` of label dates). Members are refit on the purged inner-train rows first, so the
              holdout predictions are out-of-sample (in-sample LGBM would always win). rank keeps its weight.
- "model"   : w_model / w_sector as ratios to the block composite, on the same inner holdout.

Weights solve the max-correlation combination of the terms with the same-Date demeaned label
(beta = (C + l2 * mean(diag C) I)^-1 c, non-negative by active-set elimination). The fitted weights are
constants stored in model["blend"] (meta_v2.json "blend"); prediction only overrides params with them.
This module never reads files and never sees dates after the training window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TARGETS = ("blocks", "ensemble", "model")

DEFAULT_BLEND = {
    "enabled": False,
    "targets": ["blocks"],  # subset of TARGETS
    "shrink": 0.5,  # 0 = manual weights, 1 = learned weights (after the L1 rescale)
    "l2": 0.1,  # ridge on the term covariance, relative to its mean diagonal
    "nonneg": True,  # no sign flips against the manual design
    "holdout_frac": 0.3,  # ensemble / model: last share of label dates used as the inner holdout
    "min_holdout_dates": 60,
    "inner_seeds": [0],  # seeds of the inner-train LGBM (honest holdout predictions)
    "inner_n_estimators": None,  # None -> model_params.n_estimators
    "max_model_ratio": 4.0,  # learned model_weight / sector_weight <= this multiple of the manual value
}


def settings(config: dict) -> dict:
    out = dict(DEFAULT_BLEND)
    out.update(config.get("blend_learning") or {})
    unknown = set(out["targets"]) - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown blend_learning targets: {sorted(unknown)} (choose from {TARGETS})")
    return out


def is_on(config: dict) -> bool:
    return bool((config.get("blend_learning") or {}).get("enabled", False))


def apply(config: dict, learned: dict | None) -> dict:
    """学習済みの重みで params を上書きした新しい dict。OFF / 未学習なら config をそのまま返す（同一オブジェクト）。"""
    if not learned or not is_on(config):
        return config
    out = dict(config)
    for key in ("block_weights", "ensemble_weights"):
        if learned.get(key):
            out[key] = dict(config.get(key) or {}, **learned[key])
    for key in ("model_weight", "sector_weight"):
        if learned.get(key) is not None:
            out[key] = float(learned[key])
    return out


def target_vector(labels: pd.DataFrame, clip: float) -> np.ndarray:
    """ホライズン平均の「clip 後・同日 demean 済み」ラベル（どのホライズンも無い行は NaN）。"""
    parts = []
    for horizon in labels.columns:
        y = labels[horizon].astype(np.float64).clip(-clip, clip)
        parts.append((y - y.groupby(level="Date").transform("mean")).to_numpy(np.float64))
    if not parts:
        return np.full(len(labels), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmean(np.column_stack(parts), axis=1)


def solve(terms: dict[str, np.ndarray], y: np.ndarray, l2: float, nonneg: bool) -> dict[str, float]:
    """max-correlation の合成係数。欠損の項は中立 0、ラベル欠損の行は使わない。行が足りなければ {}。"""
    names = list(terms)
    mask = np.isfinite(y)
    if not names or mask.sum() < 1000:
        return {}
    x = np.column_stack([np.nan_to_num(np.asarray(terms[n], dtype=np.float64)[mask], nan=0.0) for n in names])
    target = y[mask]
    cov = x.T @ x / len(target)
    cxy = x.T @ target / len(target)
    ridge = float(l2) * (float(np.mean(np.diag(cov))) or 1.0)
    beta = np.zeros(len(names))
    active = list(range(len(names)))
    while active:
        sub = np.linalg.solve(cov[np.ix_(active, active)] + ridge * np.eye(len(active)), cxy[active])
        if not nonneg or (sub >= 0.0).all():
            beta[active] = sub
            break
        active.pop(int(np.argmin(sub)))
    return {n: float(b) for n, b in zip(names, beta)}


def rescale_mix(manual: dict[str, float], learned: dict[str, float], shrink: float) -> dict[str, float]:
    """learned を manual と同じ L1 合計に揃えてから (1 - shrink) * manual + shrink * learned。退化時は manual。"""
    total_manual = sum(abs(v) for v in manual.values())
    total_learned = sum(abs(learned.get(k, 0.0)) for k in manual)
    if total_manual <= 0.0 or total_learned <= 0.0:
        return dict(manual)
    scale, s = total_manual / total_learned, float(shrink)
    return {k: (1.0 - s) * v + s * scale * learned.get(k, 0.0) for k, v in manual.items()}


def ratio_mix(manual: float, learned: float | None, shrink: float, cap: float) -> float:
    if learned is None or not np.isfinite(learned):
        return float(manual)
    limit = float(cap) * manual if manual > 0.0 else float(cap)
    learned = min(max(float(learned), 0.0), limit)
    return (1.0 - float(shrink)) * float(manual) + float(shrink) * learned


def inner_cut(index: pd.Index, y: np.ndarray, holdout_frac: float, min_dates: int) -> pd.Timestamp | None:
    """内側ホールドアウトの開始日（ラベルのある日付の後ろ holdout_frac）。日数が足りなければ None。"""
    dates = index.get_level_values("Date")
    labelled = np.unique(np.asarray(dates)[np.isfinite(y)])
    if len(labelled) < 2 * int(min_dates):
        return None
    position = min(int(len(labelled) * (1.0 - float(holdout_frac))), len(labelled) - int(min_dates))
    return pd.Timestamp(labelled[position])


def before_cut(index: pd.Index, cut: pd.Timestamp, k: int) -> np.ndarray:
    """ラベル窓 t..t+k-1（学習期間の取引日カレンダー）が cut より前で終わる行（内側学習の purge）。"""
    dates = index.get_level_values("Date")
    calendar = pd.DatetimeIndex(np.unique(np.asarray(dates)))
    return calendar.searchsorted(dates) + (int(k) - 1) < calendar.searchsorted(cut)
