#!/usr/bin/env python3
"""selftest_p7.py — P7（block_set v3 / sample_decay_halflife / feature_top_k）を合成データで検査する（実データ不要）。

    python strategies/v0_multifactor/selftest_p7.py [--reference path/to/v4_S5/alpha_v2.py]

検査内容（合成データの数値は性能の指標ではない。動作・一致・因果性の確認だけ）:
1. 既定 OFF: --reference に v4/S5 の alpha_v2.py を渡すと、旧版（同じフォルダの ensemble.py 等と一緒に
   読み込む）と新版のシグナルを OFF の 4 構成（S5 / K1 / S5+rank / S5+regime_mix）でビット比較する。
2. ON/OFF: block_set v3 / sample_decay_halflife 250 / feature_top_k 20 などが設定だけで切り替わり、
   シグナルが S5 から変わる（列数・カテゴリ列の保持・rank モデルの列・重みの要約も確認）。
3. 因果性: 全部 ON の構成で、予測期間の後半を切り落としても前半のシグナルが変わらない。
4. 一貫性: select_features + 選択列での学習（train_v2.py の経路）が fit_model の 2 パスと同じ列・シグナルになる。
   tools/sweep_improve2.py が import できれば、成分からの再合成（compose）が predict_signal と一致するかも見る。
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
from alpha_features import SECTOR_COLUMNS  # noqa: E402
from selftest_p5 import synthetic_panel  # noqa: E402

SIBLINGS = ("alpha_features", "ensemble", "regime_v4", "slowdown")
FAST = {"seeds": [0, 1], "model_params": {"n_estimators": 25}, "rank_model": {"lgbm_params": {"n_estimators": 15}}}
# v4/S5 pinned explicitly (independent of the walkforward_config.json shipped next to either module).
S5 = {"model_features": "v1", "block_set": "v1", "model_weight": 0.25, "model_smoothing_span": 20,
      "ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0}, "regime_mix": False,
      "turnover_control": "off", "slow_profile": None, "label_clip": 0.05,
      "block_weights": {"size": 1.0, "value": 0.5, "quality": 0.5, "lowrisk": 0.3},
      "sample_decay_halflife": None, "feature_top_k": None}
OFF_CASES = {
    "S5": {},
    "K1": {"ensemble_weights": {"ridge": 0.0}, "model_weight": 0.5, "model_smoothing_span": 10},
    "S5_rank0.2": {"ensemble_weights": {"rank": 0.2}},
    "S5_regime_v4": {"regime_mix": True},
}
ON_CASES = {
    "block_set_v3": {"block_set": "v3"},
    "v3_direct": {"block_set": "v3", "block_weights": {"momentum": 0.2, "revision": 0.2, "liquidity": 0.1}},
    "decay_250": {"sample_decay_halflife": 250},
    "top_k_20": {"feature_top_k": 20},
    "all_on": {"block_set": "v3", "sample_decay_halflife": 250, "feature_top_k": 20,
               "ensemble_weights": {"rank": 0.2}, "block_weights": {"momentum": 0.2}},
}


def spec(*overrides: dict) -> dict:
    out = alpha_v2._deep_merge(FAST, S5)
    for override in overrides:
        out = alpha_v2._deep_merge(out, override)
    return out


def load_reference(path: str):
    """旧 alpha_v2.py を、同じフォルダにある旧 ensemble.py 等と一緒に読み込む（新版のモジュールは汚さない）。"""
    ref_dir = str(Path(path).resolve().parent)
    saved = {name: sys.modules.pop(name) for name in SIBLINGS if name in sys.modules}
    sys.path.insert(0, ref_dir)
    try:
        module_spec = importlib.util.spec_from_file_location("alpha_v2_reference", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    finally:
        sys.path.remove(ref_dir)
        for name in SIBLINGS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module


def run(module, train, labels, predict, params) -> tuple[pd.Series, dict, float]:
    started = time.time()
    model = module.fit_model(train, labels, params)
    return module.predict_signal(predict, model, params), model, time.time() - started


def max_diff(a: pd.Series, b: pd.Series) -> float:
    a, b = a.to_numpy(np.float64), b.reindex(a.index).to_numpy(np.float64)
    if np.array_equal(a, b, equal_nan=True):
        return 0.0
    return float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else float("inf")


def check_model(name: str, params: dict, model: dict) -> list[str]:
    """ON 構成ごとの構造チェック（選択列の本数・カテゴリ列の保持・LGBM / rank の列・Ridge のブロック・重み）。"""
    problems = []
    columns = list(model["columns"])
    if params.get("feature_top_k") is not None:
        if len([c for c in columns if c not in SECTOR_COLUMNS]) != int(params["feature_top_k"]):
            problems.append("top_k count")
        if not set(SECTOR_COLUMNS) <= set(columns):
            problems.append("sector columns dropped")
        names = [alpha_v2._feature_names(b) for b in model["models"].values()]
        names += [alpha_v2._feature_names(m) for m in (model.get("rank") or {}).values()]
        if not names or any(list(n) != columns for n in names):
            problems.append("model columns != selected")
    if params.get("block_set") == "v3" and (model.get("ridge") or {}).get("columns") != list(alpha_v2.BLOCK_SETS["v3"]):
        problems.append("ridge blocks != v3")
    if params.get("sample_decay_halflife") is not None:
        if not 0.0 < (model.get("sample_decay") or {}).get("weight_min", 0.0) < 1.0:
            problems.append("decay weights")
    return problems


def check_consistency(train, labels, predict) -> list[str]:
    """train_v2.py の経路（select_features -> 選択列で学習）と sweep_improve2 の再合成の一致。"""
    failures = []
    params = spec({"feature_top_k": 20})
    selection = alpha_v2.select_features(train, labels, params)
    two_pass = alpha_v2.fit_model(train, labels, params)
    per_job = alpha_v2.fit_model(train, labels, alpha_v2._deep_merge(
        params, {"feature_top_k": None, "model_features": selection["columns"]}))
    same = selection["columns"] == two_pass["columns"] == per_job["columns"]
    diff = max_diff(alpha_v2.predict_signal(predict, per_job, params), alpha_v2.predict_signal(predict, two_pass, params))
    print(f"[4] train_v2 path (select_features -> fit on selected): same columns={same}  max|diff| = {diff:.2e}")
    if not same or diff > 1e-6:
        failures.append("train_v2_path")
    try:
        sys.path.insert(0, str(HERE.parent.parent / "tools"))
        import sweep_improve2 as sweep  # noqa: PLC0415 - optional (needs the repo's tools/)
    except Exception as exc:  # noqa: BLE001
        print(f"[4] sweep_improve2 compose: SKIP ({type(exc).__name__}: {exc})")
        return failures
    for name in ("block_set_v3", "v3_direct", "all_on"):
        params = spec(ON_CASES[name])
        config = alpha_v2.merged_params(params)
        model = alpha_v2.fit_model(train, labels, params)
        expected = slowdown.ewm_by_code(alpha_v2.predict_signal(predict, model, params), 5)
        composed = sweep.compose(alpha_v2, sweep.components(alpha_v2, predict, model, config),
                                 dict(config, smoothing_span=5))
        diff = max_diff(composed, expected)
        print(f"[4] sweep_improve2 compose vs predict_signal ({name}): max|diff| = {diff:.2e}")
        if diff > 1e-6:
            failures.append(f"compose:{name}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="P7 switches self-test on synthetic data")
    parser.add_argument("--reference", default=None, help="v4/S5 の alpha_v2.py（OFF のビット一致検査・任意）")
    args = parser.parse_args()
    features, labels = synthetic_panel()
    dates = features.index.get_level_values("Date")
    split = dates.unique()[280]
    train, predict = features.loc[dates < split], features.loc[dates >= split]
    labels = labels.loc[train.index]
    failures: list[str] = []
    print(f"synthetic: train={len(train):,} predict={len(predict):,} rows")

    # 1. default OFF == v4/S5 (bit-identical)
    if args.reference:
        reference = load_reference(args.reference)
        for name, override in OFF_CASES.items():
            new, _, _ = run(alpha_v2, train, labels, predict, spec(override))
            old, _, _ = run(reference, train, labels, predict, spec(override))
            diff = max_diff(new, old)
            print(f"[1] OFF {name:<14} new vs reference: max|diff| = {diff:.3e}  {'bit-identical' if diff == 0 else 'DIFF'}")
            if diff != 0.0:
                failures.append(f"reference:{name}")
    else:
        print("[1] OFF vs reference: SKIP（--reference 未指定）")

    # 2. ON/OFF switches
    base, base_model, _ = run(alpha_v2, train, labels, predict, spec())
    print(f"{'case':<16}{'sec':>6}{'finite':>8}{'max|d S5|':>11}{'turnover':>10}  extra")
    for name, override in ON_CASES.items():
        params = spec(override)
        signal, model, seconds = run(alpha_v2, train, labels, predict, params)
        finite = float(np.isfinite(signal.to_numpy()).mean())
        diff = max_diff(signal, base)
        turnover = slowdown.quintile_turnover(slowdown.ewm_by_code(signal.astype(np.float64), 5))
        problems = check_model(name, params, model)
        extra = {"columns": len(model["columns"]), "ridge_blocks": len((model.get("ridge") or {}).get("columns", []))}
        if "sample_decay" in model:
            extra["decay"] = model["sample_decay"]
        print(f"{name:<16}{seconds:>6.1f}{finite:>8.3f}{diff:>11.4f}{turnover:>10.4f}  {extra} {problems or ''}")
        if finite < 0.99 or diff == 0.0 or problems:
            failures.append(name)

    # 3. causality (all ON): truncating the prediction tail must not change the head
    params = spec(ON_CASES["all_on"])
    model = alpha_v2.fit_model(train, labels, params)
    full = alpha_v2.predict_signal(predict, model, params)
    pdates = predict.index.get_level_values("Date")
    half = pdates.unique()[len(pdates.unique()) // 2]
    head = alpha_v2.predict_signal(predict.loc[pdates < half], model, params)
    causal = max_diff(head, full)
    print(f"[3] causality (all_on, truncated tail): max|diff| = {causal:.2e}")
    if causal > 1e-6:
        failures.append("causality")

    failures += check_consistency(train, labels, predict)
    print("RESULT:", "OK" if not failures else f"NG {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
