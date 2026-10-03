#!/usr/bin/env python3
"""robust_multifactor のモデルを Train 期間だけで学習する。

ラベルは「翌日1日リターン」ではなく、**翌日から k 営業日の平均残差リターン**。
シグナルがゆっくりしか動かなくなり、採点の売買コスト（片道0.1%）を抑えられる。
Valid のラベルは一切使わない。

    python strategies/robust_multifactor/train.py --data-dir input
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rf_features import SECTOR_COLUMNS, build_features, rank_model_features


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parents[1] / "input"

HORIZONS = (126, 250)
SEEDS = (0, 1, 2)
LABEL_CLIP = 0.05

MODEL_PARAMS = dict(
    objective="regression",
    n_estimators=350,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    max_bin=127,
    n_jobs=-1,
    verbose=-1,
)


def forward_mean(values: pd.Series, window: int) -> pd.Series:
    """各時点から未来 window 営業日の平均（ラベル専用。特徴量には使わない）。"""
    return values.iloc[::-1].rolling(window, min_periods=1).mean().iloc[::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = HERE / "models"
    model_dir.mkdir(exist_ok=True)

    prev_cwd = Path.cwd()
    os.chdir(data_dir)
    try:
        features = build_features(splits=("train",))
        target = pd.read_parquet("target_1day_train.parquet")["Return"]
    finally:
        os.chdir(prev_cwd)

    X = rank_model_features(features)
    feature_names = list(X.columns)
    print(f"train features: {X.shape}, dates "
          f"{X.index.get_level_values('Date').min().date()} .. "
          f"{X.index.get_level_values('Date').max().date()}")

    model_files = []
    for horizon in HORIZONS:
        label = target.groupby(level="Code", sort=False).transform(
            lambda x: forward_mean(x, horizon)
        ).clip(-LABEL_CLIP, LABEL_CLIP)
        train_index = label.dropna().index
        for seed in SEEDS:
            params = dict(MODEL_PARAMS, random_state=seed)
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X.loc[train_index],
                label.loc[train_index],
                categorical_feature=SECTOR_COLUMNS,
            )
            name = f"lgbm_k{horizon}_s{seed}.txt"
            model.booster_.save_model(str(model_dir / name))
            model_files.append(name)
            print(f"  saved {name}  rows={len(train_index):,}  best_iter={model.best_iteration_ or params['n_estimators']}")

    meta = {
        "strategy": "robust_multifactor",
        "model_files": model_files,
        "features": feature_names,
        "sector_features": SECTOR_COLUMNS,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "label_clip": LABEL_CLIP,
        "smoothing_span": 5,
        "model_weight": 0.5,
        "block_weights": {"size": 1.0, "value": 0.5, "quality": 0.5, "lowrisk": 0.3},
        "model_params": {k: v for k, v in MODEL_PARAMS.items() if k != "n_jobs"},
        "training_rows": int(len(X)),
        "training_from": str(X.index.get_level_values("Date").min().date()),
        "training_to": str(X.index.get_level_values("Date").max().date()),
    }
    (HERE / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"meta: {HERE / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
