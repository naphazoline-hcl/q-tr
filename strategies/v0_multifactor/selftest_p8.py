#!/usr/bin/env python3
"""selftest_p8.py — P8（学習型合成 blend_learning / 線形成分 linear）を合成データで検査する（実データ不要）。

    python strategies/v0_multifactor/selftest_p8.py [--reference <v5 の戦略フォルダ or その alpha_v2.py>]

1. 既定 OFF（v5 設定）で fit_model -> predict_signal が動き、キーを明示 OFF にしても完全一致する。
   --reference を渡すと、そのフォルダの旧モジュール一式（alpha_v2 / ensemble / ... ）を別プロセスで
   読み込み、同じ入力・同じ params で旧版と新版のシグナルがビット一致（np.array_equal）するかを見る。
2. linear / blend_learning（blocks / ensemble / model）が設定だけで ON/OFF でき、シグナルが変わる。
   学習結果（model["linear"] / model["blend"]）が JSON 化できる（meta_v2.json に入る形）。
3. 内側ホールドアウトの purge: 内側学習に使う行のラベル窓がすべて cut より前に終わる。
4. 因果性: 予測期間の後半を切り落としても前半のシグナルが変わらない（全部 ON で検査）。
walkforward_config.json の影響を受けないよう、両プロセスとも load_config を空にして params を明示する。
合成データの数値は性能の指標ではない（動作・一致・因果性の確認だけ）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FAST = {"seeds": [0], "model_params": {"n_estimators": 20}}
V5 = {  # v5 = v4/S5 + explicit blocks (size_pure). sector_rank_blocks is an assumption (irrelevant to the test).
    "model_features": "v1", "label_transform": "raw", "label_window": "strict", "horizon_combine": "z",
    "model_weight": 0.25, "model_smoothing_span": 20, "sector_neutral": True, "sector_weight": 0.5,
    "block_set": "v1", "sector_rank_blocks": ["value"], "regime_mix": False, "turnover_control": "off",
    "slow_profile": None, "ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
    "ridge": {"alpha": 10.0, "label_demean": True, "blocks": None},
    "blocks": {
        "size_pure": {"logsize": -1.0},
        "value": {"bp": 1.0, "sp": 1.0, "ep_f": 1.0, "ep_a": 1.0, "cfy": 1.0, "div_y": 1.0},
        "quality": {"roe": 1.0, "cfo_ta": 1.0, "eq_ratio": 1.0},
        "lowrisk": {"beta": -1.0, "vol60": -1.0},
    },
    "block_weights": {"size_pure": 2.5, "value": 0.5, "quality": 0.5, "lowrisk": 0.3},
}
ALL_TARGETS = ["blocks", "ensemble", "model"]
CASES = {
    "v5_default": {},
    "v5_explicit_off": {"ensemble_weights": {"linear": 0.0}, "blend_learning": {"enabled": False, "targets": ALL_TARGETS}},
    "linear_0.3": {"ensemble_weights": {"linear": 0.3}},
    "linear_only": {"ensemble_weights": {"lgbm": 0.0, "ridge": 0.0, "linear": 1.0}},
    "blend_blocks": {"blend_learning": {"enabled": True, "targets": ["blocks"]}},
    "blend_ensemble": {"ensemble_weights": {"linear": 0.3}, "blend_learning": {"enabled": True, "targets": ["ensemble"]}},
    "blend_model": {"blend_learning": {"enabled": True, "targets": ["model"]}},
    "all_on": {"ensemble_weights": {"linear": 0.3},
               "blend_learning": {"enabled": True, "targets": ALL_TARGETS, "shrink": 1.0}},
}


def load_strategy(directory: Path):
    """directory の alpha_v2（と同じフォルダの ensemble 等）を読み、walkforward_config.json を無効化する。"""
    sys.path.insert(0, str(directory))
    import alpha_v2  # noqa: PLC0415  (import after sys.path is set: reference or new folder)

    alpha_v2.load_config = lambda: {}
    return alpha_v2


def synthetic_panel(columns: list[str], sector_columns: list[str], n_dates: int = 500, n_codes: int = 60,
                    seed: int = 0):
    """selftest_p5 と同じ作り（銘柄ごとに持続する特徴量 + ノイズ、特徴量に弱く依存するラベル）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    codes = [str(1300 + i) for i in range(n_codes)]
    index = pd.MultiIndex.from_product([dates, codes], names=["Date", "Code"])
    n = len(index)
    base = rng.normal(size=(n_codes, len(columns)))
    values = np.tile(base, (n_dates, 1)) + 0.3 * rng.normal(size=(n, len(columns)))
    features = pd.DataFrame(values.astype(np.float32), index=index, columns=columns)
    for column in sector_columns:
        features[column] = np.tile(rng.integers(1, 6, n_codes), n_dates).astype(np.float32)
    alpha = -0.004 * features["logsize"] + 0.002 * features["bp"] + 0.002 * features["roe"] \
        + 0.001 * features["rcc120"]
    labels = {}
    for k in (126, 250):
        label = (alpha + 0.01 * rng.normal(size=n)).astype(np.float32)
        labels[f"k{k}"] = label.where(label.index.get_level_values("Date") < dates[-(k - 1)])
    return features, pd.DataFrame(labels)


def split_panel(alpha_v2):
    import alpha_features  # noqa: PLC0415

    features, labels = synthetic_panel(list(alpha_features.ALL_COLUMNS), list(alpha_features.SECTOR_COLUMNS))
    dates = features.index.get_level_values("Date")
    cut = dates.unique()[380]
    train, predict = features.loc[dates < cut], features.loc[dates >= cut]
    return train, labels.loc[train.index].dropna(how="all"), predict


def run_case(alpha_v2, train, labels, predict, override: dict):
    params = alpha_v2._deep_merge(alpha_v2._deep_merge(FAST, V5), override)
    model = alpha_v2.fit_model(train.loc[labels.index], labels, params)
    return alpha_v2.predict_signal(predict, model, params), model, params


def emit_reference(directory: Path, out: Path) -> int:
    alpha_v2 = load_strategy(directory)
    train, labels, predict = split_panel(alpha_v2)
    signal, _, _ = run_case(alpha_v2, train, labels, predict, {})
    np.save(out, signal.to_numpy())
    return 0


def check_inner_purge(alpha_v2, train, labels) -> float:
    """内側学習行の max(位置 + k - 1) と cut の位置の差（負なら purge OK）。"""
    blend = alpha_v2.blend
    y = blend.target_vector(labels.reindex(train.loc[labels.index].index), 0.05)
    index = train.loc[labels.index].index
    cut = blend.inner_cut(index, y, 0.3, 60)
    calendar = pd.DatetimeIndex(np.unique(np.asarray(index.get_level_values("Date"))))
    worst = -np.inf
    for horizon in labels.columns:
        k = int(horizon[1:])
        rows = blend.before_cut(index, cut, k) & labels[horizon].reindex(index).notna().to_numpy()
        if rows.any():
            last = calendar.searchsorted(index.get_level_values("Date")[rows]).max() + k - 1
            worst = max(worst, float(last - calendar.searchsorted(cut)))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description="P8 self-test on synthetic data")
    parser.add_argument("--reference", default=None, help="v5 の戦略フォルダ（または旧 alpha_v2.py）")
    parser.add_argument("--emit-reference", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.emit_reference:
        return emit_reference(Path(args.emit_reference), Path(args.out))

    alpha_v2 = load_strategy(HERE)
    train, labels, predict = split_panel(alpha_v2)
    print(f"synthetic: train={len(train):,} predict={len(predict):,} rows")
    failures, baseline = [], None
    for name, override in CASES.items():
        started = time.time()
        signal, model, params = run_case(alpha_v2, train, labels, predict, override)
        values = signal.to_numpy()
        baseline = values if baseline is None else baseline
        same = bool(np.array_equal(values, baseline, equal_nan=True))
        diff = float(np.nanmax(np.abs(values - baseline)))
        finite = float(np.isfinite(values).mean())
        extra = {k: model[k] for k in ("blend",) if k in model}
        if "linear" in model:
            extra["linear_rows"] = model["linear"]["rows"]
        json.dumps({k: model.get(k) for k in ("linear", "blend")})  # must be meta_v2.json-serializable
        print(f"{name:<18}{time.time() - started:>6.1f}s finite={finite:.3f} max|d v5|={diff:.4g} {extra}")
        expect_same = name in ("v5_default", "v5_explicit_off")
        if finite < 0.99 or same != expect_same:
            failures.append(name)

    worst = check_inner_purge(alpha_v2, train, labels)
    print(f"inner purge: max(window end) - cut = {worst:+.0f} trading days (must be < 0)")
    if not worst < 0:
        failures.append("inner_purge")

    signal, model, params = run_case(alpha_v2, train, labels, predict, CASES["all_on"])
    pdates = predict.index.get_level_values("Date")
    half = pdates.unique()[len(pdates.unique()) // 2]
    head = alpha_v2.predict_signal(predict.loc[pdates < half], model, params)
    causal = float(np.nanmax(np.abs(signal.loc[head.index].to_numpy() - head.to_numpy())))
    print(f"causality (all_on, truncated tail): max|diff| = {causal:.2e}")
    if causal > 1e-6:
        failures.append("causality")

    if args.reference:
        directory = Path(args.reference).resolve()
        directory = directory.parent if directory.is_file() else directory
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reference.npy"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--emit-reference", str(directory),
                            "--out", str(out)], check=True)
            old = np.load(out)
        same = bool(np.array_equal(old, baseline, equal_nan=True))
        print(f"v5 (OFF) vs reference {directory}: bit-identical={same} max|diff|={np.nanmax(np.abs(old - baseline)):.2e}")
        if not same:
            failures.append("reference")
    print("RESULT:", "OK" if not failures else f"NG {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
