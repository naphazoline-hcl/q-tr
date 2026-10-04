#!/usr/bin/env python3
"""bench_p7.py — P7 の feature_top_k（2 パス学習）と sample_decay_halflife の学習・推論時間を測る（実データ不要）。

    python strategies/v0_multifactor/bench_p7.py [--train-dates 1500] [--codes 498] [--top-k 20] [--models 6]

bench_p5.py と同じ合成パネル（Train 全期間 ≈ 1500 日 x 498 銘柄、モデル入力 "v1" 51 列）で 1 モデルずつ測り、
1 フォールド = models 本（既定: 2 ホライズン x 3 seed = 6）に換算する:
    feature_top_k ON : models x fit(全列) + 列選択 + models x fit(選択列)   （1 パス目 + 2 パス目）
    + sample_decay   : 1 パス目・2 パス目とも重み付き fit（重みの計算自体は日付の factorize だけ）
推論は選択列のモデルで Valid 規模の行を予測する時間（モデル本数分）。合成データなので木の形は実データと違う。
時間の目安としてだけ使うこと（精度の数字ではない）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import alpha_v2  # noqa: E402
import ensemble  # noqa: E402
from alpha_features import SECTOR_COLUMNS  # noqa: E402
from bench_p5 import panel, timed  # noqa: E402


def fit(x: pd.DataFrame, y: pd.Series, weight: np.ndarray | None) -> lgb.LGBMRegressor:
    categorical = [c for c in x.columns if c in SECTOR_COLUMNS]
    model = lgb.LGBMRegressor(random_state=0, **alpha_v2.DEFAULT_MODEL_PARAMS)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    return model.fit(x, np.ascontiguousarray(y.to_numpy(), dtype=np.float32), sample_weight=w,
                     categorical_feature=categorical)


def main() -> int:
    parser = argparse.ArgumentParser(description="P7 timing benchmark on a synthetic panel")
    parser.add_argument("--train-dates", type=int, default=1500)
    parser.add_argument("--predict-dates", type=int, default=2930)
    parser.add_argument("--codes", type=int, default=498)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--models", type=int, default=6, help="1 フォールドの LGBM 本数（ホライズン x seed）")
    args = parser.parse_args()
    print(f"cpu={os.cpu_count()} train={args.train_dates}x{args.codes} predict={args.predict_dates}x{args.codes} "
          f"top_k={args.top_k} models/fold={args.models}", flush=True)

    x, y = panel(args.train_dates, args.codes, 0)
    weight, t_weight = timed("decay weights (halflife 250, factorize dates)",
                             lambda: alpha_v2.sample_weights(x.index, {"sample_decay_halflife": 250}))
    full, t_full = timed(f"LGBM fit all {x.shape[1]} columns (pass 1, 1 model)", lambda: fit(x, y, None))
    _, t_full_w = timed("LGBM fit all columns + decay weights (1 model)", lambda: fit(x, y, weight))
    selection, t_select = timed(f"select_columns top_k={args.top_k} (gain rank)",
                                lambda: alpha_v2.select_columns({"m": full}, list(x.columns), args.top_k))
    columns = selection["columns"]
    small, t_small = timed(f"LGBM fit {len(columns)} selected columns (pass 2, 1 model)",
                           lambda: fit(x[columns], y, None))
    _, t_small_w = timed("LGBM fit selected columns + decay weights (1 model)", lambda: fit(x[columns], y, weight))
    blocks = pd.DataFrame(np.random.default_rng(1).standard_normal((len(x), 7)), index=x.index,
                          columns=list(alpha_v2.BLOCK_SETS["v3"]))
    _, t_ridge = timed("Ridge fit v3 (7 block z) + decay weights",
                       lambda: ensemble.fit_ridge(blocks, y.to_frame(), dict(ensemble.DEFAULT_RIDGE), 0.05, weight))
    del x, y, blocks

    xp, _ = panel(args.predict_dates, args.codes, 2)
    matrix = np.ascontiguousarray(xp[columns].to_numpy(dtype=np.float32))
    _, t_pred = timed("LGBM predict selected columns (1 model, valid rows)", lambda: small.booster_.predict(matrix))

    n = int(args.models)
    estimate = {
        "train_fold_top_k_s": n * t_full + t_select + n * t_small + t_ridge,
        "train_fold_top_k_decay_s": t_weight + n * t_full_w + t_select + n * t_small_w + t_ridge,
        "train_fold_baseline_s": n * t_full + t_ridge,
        "predict_models_s": n * t_pred,
    }
    print("RESULT:", {k: round(v, 1) for k, v in {
        "weights": t_weight, "fit_full": t_full, "fit_full_decay": t_full_w, "select": t_select,
        "fit_selected": t_small, "fit_selected_decay": t_small_w, "ridge_v3_decay": t_ridge, "predict": t_pred,
    }.items()})
    print("ESTIMATE (1 fold / inference, seconds):", {k: round(v, 1) for k, v in estimate.items()})
    print("ESTIMATE (minutes):", {k: round(v / 60.0, 2) for k, v in estimate.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
