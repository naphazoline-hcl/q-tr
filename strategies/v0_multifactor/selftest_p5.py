#!/usr/bin/env python3
"""selftest_p5.py — P5 の切替（ensemble / regime / turnover）を合成データで検査する（実データ不要）。

    python strategies/v0_multifactor/selftest_p5.py [--reference path/to/old/alpha_v2.py]

検査内容（合成データの Sharpe 等は性能の指標ではない。動作と因果性の確認だけ）:
1. 既定（K1）設定で fit_model -> predict_signal が動く。--reference に旧 alpha_v2.py を渡すと
   同じ入力で旧版と新版のシグナルが一致するかを max|diff| で出す（K1 不変の確認）。
2. ensemble（ridge / rank: lambdarank, rank_xendcg, quantile）・regime_mix（v4 / legacy）・
   turnover_control auto・slow_profile がそれぞれ設定だけで ON/OFF でき、シグナルが変わる。
3. 因果性: 予測期間の後半を切り落としても、前半のシグナルが変わらない（全部 ON の構成で検査）。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import alpha_v2  # noqa: E402
import slowdown  # noqa: E402
from alpha_features import ALL_COLUMNS, SECTOR_COLUMNS  # noqa: E402

FAST = {"seeds": [0], "model_params": {"n_estimators": 20}, "rank_model": {"lgbm_params": {"n_estimators": 15}}}
# Every case starts from K1 regardless of walkforward_config.json (which may ship a candidate such as E1).
K1 = {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.0}, "regime_mix": False,
      "turnover_control": "off", "slow_profile": None}
CASES = {
    "K1_default": {},
    "ridge_0.3": {"ensemble_weights": {"ridge": 0.3}},
    "rank_lambdarank": {"ensemble_weights": {"rank": 0.3}},
    "rank_xendcg": {"ensemble_weights": {"rank": 0.3}, "rank_model": {"objective": "rank_xendcg"}},
    "rank_quantile": {"ensemble_weights": {"rank": 0.3}, "rank_model": {"objective": "quantile"}},
    "regime_v4": {"regime_mix": True},
    "regime_legacy": {"regime_mix": True, "regime_mode": "legacy"},
    "turnover_auto": {"turnover_control": "auto"},
    "profile_span5": {"slow_profile": "span5"},
    "profile_span20": {"slow_profile": "span20"},
    "all_on": {"ensemble_weights": {"ridge": 0.3, "rank": 0.2}, "regime_mix": True,
               "turnover_control": "auto"},
}


def synthetic_panel(n_dates: int = 400, n_codes: int = 80, seed: int = 0):
    """銘柄ごとに持続する特徴量 + ノイズ、日付ごとに同値の局面列、特徴量に弱く依存するラベル。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    codes = [str(1300 + i) for i in range(n_codes)]
    index = pd.MultiIndex.from_product([dates, codes], names=["Date", "Code"])
    n = len(index)
    base = rng.normal(size=(n_codes, len(ALL_COLUMNS)))
    values = np.tile(base, (n_dates, 1)) + 0.3 * rng.normal(size=(n, len(ALL_COLUMNS)))
    features = pd.DataFrame(values.astype(np.float32), index=index, columns=ALL_COLUMNS)
    for column in SECTOR_COLUMNS:
        features[column] = np.tile(rng.integers(1, 6, n_codes), n_dates).astype(np.float32)
    t = np.arange(n_dates, dtype=np.float64)
    cycles = {  # per-Date macro / market paths that visit every regime in both train and predict windows
        "consumer_level_z": np.sin(t / 30.0), "consumer_diff": 0.1 * np.cos(t / 30.0),
        "usdjpy_chg20": 0.03 * np.sin(t / 40.0), "usdjpy_chg60": 0.06 * np.sin(t / 50.0),
        "topix_cum20": 0.05 * np.sin(t / 25.0), "topix_vol60": 0.01 + 0.004 * np.sin(t / 45.0),
    }
    for column, path in cycles.items():
        features[column] = np.repeat(path, n_codes).astype(np.float32)
    alpha = -0.004 * features["logsize"] + 0.002 * features["bp"] + 0.002 * features["roe"]
    labels = {}
    for k in (126, 250):
        label = (alpha + 0.01 * rng.normal(size=n)).astype(np.float32)
        cut = dates[-(k - 1)] if k - 1 < n_dates else dates[0]
        labels[f"k{k}"] = label.where(label.index.get_level_values("Date") < cut)
    return features, pd.DataFrame(labels)


def run_case(module, train, labels, predict, params) -> tuple[pd.Series, dict, float]:
    started = time.time()
    model = module.fit_model(train, labels, params)
    signal = module.predict_signal(predict, model, params)
    return signal, model, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="P5 switches self-test on synthetic data")
    parser.add_argument("--reference", default=None, help="旧 alpha_v2.py（K1 一致の検査用・任意）")
    args = parser.parse_args()

    features, labels = synthetic_panel()
    dates = features.index.get_level_values("Date")
    split = dates.unique()[280]
    train, predict = features.loc[dates < split], features.loc[dates >= split]
    labels = labels.loc[train.index]
    failures = []
    baseline = None
    print(f"synthetic: train={len(train):,} predict={len(predict):,} rows")
    print(f"{'case':<18}{'sec':>6}{'finite':>8}{'max|d K1|':>11}{'turnover':>10}  extra")
    for name, override in CASES.items():
        params = alpha_v2._deep_merge(alpha_v2._deep_merge(FAST, K1), override)
        signal, model, seconds = run_case(alpha_v2, train, labels, predict, params)
        finite = float(np.isfinite(signal.to_numpy()).mean())
        turnover = slowdown.quintile_turnover(slowdown.ewm_by_code(signal.astype(np.float64), 5))
        baseline = signal if baseline is None else baseline
        diff = float(np.nanmax(np.abs(signal.to_numpy() - baseline.to_numpy())))
        extra = {k: model[k] for k in ("turnover_estimates", "model_smoothing_span") if k in model}
        if "rank" in model:
            extra["rank_models"] = len(model["rank"])
        if "ridge" in model:
            extra["ridge_coef"] = {h: [round(c, 5) for c in v] for h, v in model["ridge"]["coef"].items()}
        print(f"{name:<18}{seconds:>6.1f}{finite:>8.3f}{diff:>11.4f}{turnover:>10.4f}  {extra}")
        if finite < 0.99 or (name != "K1_default" and diff == 0.0):
            failures.append(name)

    params = alpha_v2._deep_merge(alpha_v2._deep_merge(FAST, K1), CASES["all_on"])
    model = alpha_v2.fit_model(train, labels, params)
    full = alpha_v2.predict_signal(predict, model, params)
    pdates = predict.index.get_level_values("Date")
    half = pdates.unique()[len(pdates.unique()) // 2]
    head = alpha_v2.predict_signal(predict.loc[pdates < half], model, params)
    causal = float(np.nanmax(np.abs(full.loc[head.index].to_numpy() - head.to_numpy())))
    print(f"causality (all_on, truncated tail): max|diff| = {causal:.2e}")
    if causal > 1e-6:
        failures.append("causality")

    if args.reference:
        spec = importlib.util.spec_from_file_location("alpha_v2_reference", args.reference)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        old, _, _ = run_case(reference, train, labels, predict, dict(FAST))
        diff = float(np.nanmax(np.abs(old.to_numpy() - baseline.to_numpy())))
        print(f"K1 vs reference alpha_v2: max|diff| = {diff:.2e}")
        if diff > 1e-6:
            failures.append("reference")
    print("RESULT:", "OK" if not failures else f"NG {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
