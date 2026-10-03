#!/usr/bin/env python3
"""合成の重みを Train 期間の回帰で決める。

2つの特徴量をどんな比率で足すかは、手で決めずに学習で決める。
教師は `target_1day_train` だけで、Valid のラベルは一切使わない。

    python strategies/sample02_size_liquidity/train.py --data-dir input
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features import FEATURE_NAMES, build_features


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parents[1] / "input"
MODEL_PATH = HERE / "model.json"
RIDGE_ALPHA = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.out).resolve()

    # build_features() は parquet をベース名で相対読みするので、data_dir へ移動する。
    prev_cwd = Path.cwd()
    os.chdir(data_dir)
    try:
        features = build_features()
        target = pd.read_parquet("target_1day_train.parquet")
    finally:
        os.chdir(prev_cwd)

    # Train の行だけを学習に使う。Valid の特徴量は係数の決定に一切関与しない。
    common = features.index.intersection(target.index)
    x = features.loc[common, list(FEATURE_NAMES)]
    y = target.loc[common, "Return"]
    usable = y.notna() & x.notna().all(axis=1)
    x, y = x.loc[usable], y.loc[usable]
    if x.empty:
        raise RuntimeError("学習可能な行がありません")

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))

    coef = dict(zip(FEATURE_NAMES, (float(c) for c in model.coef_)))
    payload = {
        "model": "sklearn.linear_model.Ridge",
        "alpha": RIDGE_ALPHA,
        "feature_names": list(FEATURE_NAMES),
        "coef": [coef[name] for name in FEATURE_NAMES],
        "intercept": float(model.intercept_),
        "training_rows": int(len(x)),
        "training_from": str(x.index.get_level_values("Date").min().date()),
        "training_to": str(x.index.get_level_values("Date").max().date()),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    scale = abs(coef[FEATURE_NAMES[0]])
    print(f"model: {output}")
    print(f"rows:  {len(x):,}")
    print(f"range: {payload['training_from']} .. {payload['training_to']}")
    print("coef:")
    for name in FEATURE_NAMES:
        print(f"  {name:<8} {coef[name]:+.6e}   （規模を1とした比率 {coef[name]/scale:+.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
