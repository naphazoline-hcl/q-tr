"""ensemble.py — P5 A: model diversification for alpha_v2.

- ridge: sklearn Ridge on the factor-block z scores (pooled panel regression over the whole
  training window: one coefficient vector per horizon, learned through time, not per date).
- rank : LightGBM ranking / quantile model on the same ranked inputs as the main LGBM.
         objective = lambdarank | rank_xendcg (per-Date query, label = per-Date quantile bucket)
                   | quantile (median regression on the clipped label). Default OFF (weight 0).
- combine(): every model prediction is z-scored within the same Date, then summed with
  params["ensemble_weights"]. If only "lgbm" is active the LGBM prediction is returned as is
  (bit-identical to K1).

This module never reads files. Labels are passed in by the caller (walk-forward / train_v2).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

MODEL_KEYS = ("lgbm", "ridge", "rank")
DEFAULT_ENSEMBLE_WEIGHTS = {"lgbm": 1.0, "ridge": 0.0, "rank": 0.0}

DEFAULT_RIDGE = {
    "alpha": 10.0,
    "label_demean": True,  # regress the same-Date demeaned label (the score is cross-sectional)
    "blocks": None,  # None -> the signal's own block definitions (alpha_v2.resolve_blocks)
}

DEFAULT_RANK = {
    "objective": "lambdarank",  # lambdarank | rank_xendcg | quantile
    "relevance_levels": 5,  # per-Date label quintile -> integer relevance 0..4 (ranking objectives)
    "two_sided": True,  # ranking objectives: also fit the reversed relevance; score = long - short
    "quantile_alpha": 0.5,
    "date_stride": 2,  # train on every n-th Date (CPU time; groups stay whole days)
    "horizons": None,  # None -> every label column given to fit_model
    "seeds": [0],
    "lgbm_params": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 300,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "max_bin": 127,
        "lambdarank_truncation_level": 30,
        "n_jobs": -1,
        "verbose": -1,
    },
}


def xsec_z(values: pd.Series) -> pd.Series:
    """Same-Date cross-sectional z clipped to +-3 (same formula as alpha_v2.xsec_z)."""
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0).astype(np.float32)


def ensemble_weights(config: dict) -> dict[str, float]:
    weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)
    weights.update({k: float(v) for k, v in (config.get("ensemble_weights") or {}).items()})
    unknown = set(weights) - set(MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown ensemble_weights keys: {sorted(unknown)} (choose from {MODEL_KEYS})")
    return weights


def merged_section(config: dict, key: str, default: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in default.items()}
    for k, v in (config.get(key) or {}).items():
        out[k] = dict(out[k], **v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def combine(predictions: dict[str, pd.Series | None], weights: dict[str, float]) -> pd.Series | None:
    """sum_m w_m z(pred_m)。NaN の行はその模型の寄与を 0 にする。lgbm 単独なら素通し（K1 と同一）。"""
    active = {k: w for k, w in weights.items() if w != 0.0 and predictions.get(k) is not None}
    if not active:
        return None
    if list(active) == ["lgbm"]:
        return predictions["lgbm"]
    index = predictions[next(iter(active))].index
    total = np.zeros(len(index), dtype=np.float64)
    for key, weight in active.items():
        total += weight * np.nan_to_num(xsec_z(predictions[key]).to_numpy(np.float64), nan=0.0)
    return pd.Series(total, index=index, dtype=np.float64)


# ------------------------------------------------------------------- ridge


def block_frame(block_z: dict[str, np.ndarray], index: pd.Index) -> pd.DataFrame:
    """ブロック z（同日断面 z 済み）を Ridge の説明変数に。欠損は中立 0。"""
    data = {name: np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0) for name, z in block_z.items()}
    return pd.DataFrame(data, index=index)


def fit_ridge(blocks: pd.DataFrame, labels: pd.DataFrame, cfg: dict, clip: float,
              sample_weight: np.ndarray | None = None) -> dict:
    """ホライズンごとに 1 本の Ridge（全学習期間のパネルをプールして時系列方向に係数を学習）。

    sample_weight（P7 B、blocks.index と同じ行順、None = 従来どおり等ウェイト）を渡すと重み付き最小二乗。
    重みは同一 Date 内で一定（時間減衰）なので、label_demean の同日平均は重みの有無で変わらない。
    """
    out = {"columns": list(blocks.columns), "coef": {}, "intercept": {}, "rows": {}}
    for horizon in labels.columns:
        y = labels[horizon].reindex(blocks.index).astype(np.float64).clip(-clip, clip)
        if cfg["label_demean"]:
            y = y - y.groupby(level="Date").transform("mean")
        mask = y.notna().to_numpy()
        if mask.sum() < 1000:
            continue
        model = Ridge(alpha=float(cfg["alpha"]), fit_intercept=not cfg["label_demean"])
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)[mask]
        model.fit(blocks.to_numpy()[mask], y.to_numpy()[mask], sample_weight=weight)
        out["coef"][horizon] = [float(c) for c in model.coef_]
        out["intercept"][horizon] = float(model.intercept_) if not cfg["label_demean"] else 0.0
        out["rows"][horizon] = int(mask.sum())
    return out


def predict_ridge(blocks: pd.DataFrame, ridge: dict | None) -> pd.Series | None:
    """ホライズンごとに X @ coef -> 同日断面 z -> ホライズン平均。係数が無ければ None。"""
    if not ridge or not ridge.get("coef"):
        return None
    matrix = blocks.reindex(columns=ridge["columns"]).fillna(0.0).to_numpy(np.float64)
    parts = []
    for horizon, coef in ridge["coef"].items():
        raw = matrix @ np.asarray(coef, dtype=np.float64) + float(ridge["intercept"].get(horizon, 0.0))
        parts.append(xsec_z(pd.Series(raw, index=blocks.index)).to_numpy(np.float64))
    return pd.Series(np.mean(parts, axis=0), index=blocks.index, dtype=np.float64)


# -------------------------------------------------------- rank / quantile LGBM


def relevance(label: pd.Series, levels: int) -> pd.Series:
    """同日断面の分位バケット 0..levels-1（ランキング目的関数の整数ラベル）。"""
    pct = label.groupby(level="Date").rank(pct=True, method="first")
    return np.floor(pct * levels).clip(0, levels - 1)


def _date_mask(index: pd.Index, stride: int) -> np.ndarray:
    """n 日おきの Date だけ残す（日単位で間引くのでクエリ=1日は欠けない）。"""
    dates = index.get_level_values("Date")
    codes, uniques = pd.factorize(dates, sort=True)
    return (codes % max(1, int(stride))) == 0


def _group_sizes(index: pd.Index) -> np.ndarray:
    dates = index.get_level_values("Date").to_numpy()
    if len(dates) > 1 and (dates[1:] < dates[:-1]).any():
        raise ValueError("rank model needs rows sorted by Date (contiguous per-Date queries)")
    _, counts = np.unique(dates, return_counts=True)
    return counts


def fit_rank(ranked: pd.DataFrame, labels: pd.DataFrame, cfg: dict, categorical: list[str], clip: float,
             sample_weight: np.ndarray | None = None) -> dict:
    """{"k250_s0_long": model, "k250_s0_short": model, ...}。quantile は "_long" だけ。

    sample_weight（P7 B、ranked.index と同じ行順、None = 従来どおり）は各 fit の sample_weight へ渡す
    （LightGBM は行重みで勾配・ヘッセを掛ける。ranking でも行単位で効く）。
    """
    objective = str(cfg["objective"])
    if objective not in ("lambdarank", "rank_xendcg", "quantile"):
        raise ValueError(f"unknown rank objective: {objective!r} (lambdarank | rank_xendcg | quantile)")
    params = dict(cfg["lgbm_params"])
    params.pop("random_state", None)
    if objective != "lambdarank":
        params.pop("lambdarank_truncation_level", None)
    horizons = cfg.get("horizons") or list(labels.columns)
    keep_dates = _date_mask(ranked.index, int(cfg["date_stride"]))
    boosters: dict[str, object] = {}
    for horizon in [h for h in horizons if h in labels.columns]:
        y = labels[horizon].reindex(ranked.index).astype(np.float64).clip(-clip, clip)
        mask = keep_dates & y.notna().to_numpy()
        if mask.sum() < 1000:
            continue
        x, y = ranked.loc[mask], y.loc[mask]
        # LightGBM 4.1.0 + numpy 2 calls np.array(..., copy=False): labels / groups must already
        # have the dtype LightGBM wants (float32 / int32), otherwise it raises "Unable to avoid copy".
        groups = _group_sizes(x.index).astype(np.int32)
        # Same numpy 2 rule as labels: weights must already be contiguous float32 (or None = unweighted).
        weight = None if sample_weight is None else \
            np.ascontiguousarray(np.asarray(sample_weight)[mask], dtype=np.float32)
        for seed in cfg["seeds"]:
            if objective == "quantile":
                model = lgb.LGBMRegressor(objective="quantile", alpha=float(cfg["quantile_alpha"]),
                                          random_state=int(seed), **params)
                model.fit(x, np.ascontiguousarray(y.to_numpy(), dtype=np.float32), sample_weight=weight,
                          categorical_feature=categorical)
                boosters[f"{horizon}_s{seed}_long"] = model
                continue
            rel = relevance(y, int(cfg["relevance_levels"])).to_numpy()
            sides = {"long": rel}
            if cfg["two_sided"]:
                sides["short"] = int(cfg["relevance_levels"]) - 1 - rel
            for side, target in sides.items():
                model = lgb.LGBMRanker(objective=objective, random_state=int(seed), **params)
                model.fit(x, np.ascontiguousarray(target, dtype=np.float32), sample_weight=weight, group=groups,
                          categorical_feature=categorical)
                boosters[f"{horizon}_s{seed}_{side}"] = model
    return boosters


def combine_rank(raw: dict[str, np.ndarray], index: pd.Index) -> pd.Series | None:
    """raw = {model 名: 予測配列}。seed 平均 -> z(long) - z(short) -> 同日 z -> ホライズン平均。"""
    if not raw:
        return None
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for name, values in raw.items():
        head, side = name.rsplit("_", 1)
        horizon = head.rsplit("_s", 1)[0]
        grouped.setdefault(horizon, {}).setdefault(side, []).append(np.asarray(values, dtype=np.float64))
    parts = []
    for sides in grouped.values():
        score = xsec_z(pd.Series(np.mean(sides["long"], axis=0), index=index)).to_numpy(np.float64)
        if "short" in sides:
            score = score - xsec_z(pd.Series(np.mean(sides["short"], axis=0), index=index)).to_numpy(np.float64)
        parts.append(xsec_z(pd.Series(score, index=index)).to_numpy(np.float64))
    return pd.Series(np.mean(parts, axis=0), index=index, dtype=np.float64)
