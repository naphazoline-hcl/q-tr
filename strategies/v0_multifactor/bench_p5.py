#!/usr/bin/env python3
"""bench_p5.py — P5 モデル追加の学習・推論時間を、実データと同じ形の合成パネルで測る（実データ不要）。

    python strategies/v0_multifactor/bench_p5.py [--train-dates 1500] [--predict-dates 2930] [--codes 498]

既定の行数は Train 全期間の学習（train_v2: 2008-11〜2016-03、strict ラベル）と Valid 推論
（2014-06〜2026-07）の規模の概算。列はモデル入力 "v1"（51 列: 断面順位 47 + 業種等コード 4）。
測るのは LightGBM の fit / predict と Ridge だけ（特徴量構築・順位化は含まない）。
合成データなので木の形は実データと違う。時間の目安としてだけ使うこと（精度の数字ではない）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import alpha_v2  # noqa: E402
import ensemble  # noqa: E402
from alpha_features import SECTOR_COLUMNS, V1_COLUMNS  # noqa: E402


def panel(n_dates: int, n_codes: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-11-04", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, [str(1300 + i) for i in range(n_codes)]], names=["Date", "Code"])
    columns = list(V1_COLUMNS)
    values = rng.random((len(index), len(columns)), dtype=np.float32)
    frame = pd.DataFrame(values, index=index, columns=columns)
    for column in SECTOR_COLUMNS:
        frame[column] = np.tile(rng.integers(1, 34, n_codes), n_dates).astype(np.float32)
    signal = 0.01 * (frame["logsize"] - 0.5) + 0.005 * (frame["bp"] - 0.5)
    label = (signal + 0.02 * rng.standard_normal(len(index))).astype(np.float32).clip(-0.05, 0.05)
    return frame, label.rename("k250")


def timed(label: str, fn):
    started = time.time()
    out = fn()
    seconds = time.time() - started
    print(f"  {label:<44}{seconds:>8.1f} s", flush=True)
    return out, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="P5 timing benchmark on a synthetic panel")
    parser.add_argument("--train-dates", type=int, default=1500)
    parser.add_argument("--predict-dates", type=int, default=2930)
    parser.add_argument("--codes", type=int, default=498)
    args = parser.parse_args()
    print(f"cpu={os.cpu_count()} train={args.train_dates}x{args.codes} predict={args.predict_dates}x{args.codes}")

    x, y = panel(args.train_dates, args.codes, 0)
    labels = y.to_frame()
    categorical = [c for c in x.columns if c in SECTOR_COLUMNS]
    params = dict(alpha_v2.DEFAULT_MODEL_PARAMS)
    lgbm, t_lgbm = timed("LGBM regression fit (350 trees, 1 model)",
                         lambda: __import__("lightgbm").LGBMRegressor(random_state=0, **params)
                         .fit(x, np.ascontiguousarray(y.to_numpy(), dtype=np.float32), categorical_feature=categorical))
    blocks = pd.DataFrame(np.random.default_rng(1).standard_normal((len(x), 4)), index=x.index,
                          columns=["size", "value", "quality", "lowrisk"])
    _, t_ridge = timed("Ridge fit (4 block z, pooled panel)",
                       lambda: ensemble.fit_ridge(blocks, labels, dict(ensemble.DEFAULT_RIDGE), 0.05))
    results = {"lgbm_fit": t_lgbm, "ridge_fit": t_ridge}
    for objective in ("lambdarank", "rank_xendcg", "quantile"):
        cfg = ensemble.merged_section({"rank_model": {"objective": objective}}, "rank_model", ensemble.DEFAULT_RANK)
        models, seconds = timed(f"rank fit {objective} (150 trees, stride 2, all sides)",
                                lambda: ensemble.fit_rank(x, labels, cfg, categorical, 0.05))
        results[f"rank_fit_{objective}"] = seconds
    del x, y, blocks

    xp, _ = panel(args.predict_dates, args.codes, 2)
    matrix = np.ascontiguousarray(xp.to_numpy(dtype=np.float32))
    _, t_pred = timed("LGBM predict (350 trees, 1 model, valid rows)", lambda: lgbm.booster_.predict(matrix))
    _, t_rank = timed("rank predict (150 trees, 1 model, valid rows)",
                      lambda: next(iter(models.values())).booster_.predict(matrix))
    results.update({"lgbm_predict": t_pred, "rank_predict": t_rank})
    print("RESULT:", {k: round(v, 1) for k, v in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
