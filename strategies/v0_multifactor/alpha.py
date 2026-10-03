"""alpha.py — 戦略本体（walkforward.py と submission.py が共有する API 実装）。

v0_multifactor v1: 4ブロックのファクター合成 + LightGBM（長期平均ラベル）。
`docs/coding_conventions.md` の API を満たす実例でもある。v0 が生成する v2 は
このファイルを置き換える（または拡張する）形で入ってくる。

データの読み方
--------------
公式採点では `predict()` の実行ディレクトリがデータ展開先になる。本モジュールは
`input/` がリポジトリ直下にある開発時と、カレントディレクトリに parquet がある
採点時の両方で動くように `_data_dir()` で切り替える。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rf_features import FACTOR_BLOCKS, SECTOR_COLUMNS, rank_model_features, signed_block, xsec_z
import rf_features as rff

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 350,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "max_bin": 127,
    "verbose": -1,
}

DEFAULT_PARAMS = {
    "seeds": [0, 1, 2],
    "model_weight": 0.5,
    "block_weights": {"size": 1.0, "value": 0.5, "quality": 0.5, "lowrisk": 0.3},
    "model_params": DEFAULT_MODEL_PARAMS,
    "label_clip": 0.05,
}


def load_config() -> dict:
    path = HERE / "walkforward_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merged_params(params: dict | None) -> dict:
    merged = dict(DEFAULT_PARAMS)
    merged.update(load_config().get("params", {}))
    if params:
        merged.update(params)
    return merged


def _data_dir() -> Path:
    if (ROOT / "input" / "target_1day_train.parquet").exists():
        return ROOT / "input"
    return Path.cwd()


@contextmanager
def _in_data_dir():
    target_dir = _data_dir()
    previous = Path.cwd()
    os.chdir(target_dir)
    try:
        yield target_dir
    finally:
        os.chdir(previous)


def build_features(splits=("train", "valid"), start=None, end=None, codes=None) -> pd.DataFrame:
    """特徴量パネル（index=(Date, Code)、因果的な特徴量のみ）。"""
    start = pd.Timestamp(start) if start is not None else None
    with _in_data_dir():
        features = rff.build_features(splits=tuple(splits), start=start)
    if end is not None:
        features = features.loc[features.index.get_level_values("Date") <= pd.Timestamp(end)]
    return features


def make_label(k: int, start=None, end=None, clip: float | None = None) -> pd.Series:
    """翌日から k 営業日の平均残差リターン（Train target のみ・窓が end を超えない）。

    train.py と同一の `forward_mean`（逆順 rolling・min_periods=1）を、
    `end` で切り詰めた target 系列にだけ適用する。`end` より後の target は一切使わない。
    境界付近は窓が短くなる（1日ラベルに近づく）が、未来を見ないのでリークではない。
    """
    params = merged_params(None)
    clip = float(params["label_clip"] if clip is None else clip)

    with _in_data_dir():
        target = pd.read_parquet("target_1day_train.parquet").iloc[:, 0]  # check_lookahead: train-ok
    date_values = target.index.get_level_values("Date")
    if end is not None:
        target = target.loc[date_values <= pd.Timestamp(end)]
    if start is not None:
        target = target.loc[target.index.get_level_values("Date") >= pd.Timestamp(start)]

    by_code = target.groupby(level="Code", sort=False)
    forward_mean = by_code.transform(  # check_lookahead: train-ok
        lambda s: s.iloc[::-1].rolling(k, min_periods=1).mean().iloc[::-1]  # check_lookahead: train-ok
    )
    return forward_mean.clip(-clip, clip).astype(np.float32)


def fit_model(features: pd.DataFrame, labels: pd.DataFrame, params: dict | None = None) -> dict:
    """k×seed の LightGBM を学習して返す（モデルはメモリ上、保存は呼び出し側）。"""
    config = merged_params(params)
    model_params = dict(config["model_params"])
    model_params.setdefault("n_jobs", -1)
    ranked = features.copy()
    rank_model_features(ranked)
    models: dict[str, lgb.LGBMRegressor] = {}
    for horizon in labels.columns:
        k = int(horizon.lstrip("k"))
        y = labels[horizon]
        mask = y.notna()
        for seed in config["seeds"]:
            booster = lgb.LGBMRegressor(random_state=int(seed), **model_params)
            booster.fit(
                ranked.loc[mask],
                y.loc[mask],
                categorical_feature=SECTOR_COLUMNS,
            )
            models[f"{horizon}_s{seed}"] = booster
    return {"models": models}


def predict_signal(features: pd.DataFrame, model: dict, params: dict | None = None) -> pd.Series:
    """ブロック合成 + モデル平均（平滑化前の生シグナル）。"""
    config = merged_params(params)
    signal = pd.Series(0.0, index=features.index, dtype=np.float64)
    for block, weight in config["block_weights"].items():
        columns = FACTOR_BLOCKS[block]
        signal = signal + float(weight) * xsec_z(signed_block(features, columns))

    ranked = features.copy()
    rank_model_features(ranked)
    model_signal = pd.Series(0.0, index=features.index, dtype=np.float64)
    count = 0
    for booster in model["models"].values():
        columns = list(booster.feature_name_)
        prediction = booster.predict(ranked[columns])
        model_signal = model_signal + pd.Series(prediction, index=ranked.index)
        count += 1
    if count:
        model_signal = model_signal / count
    signal = signal + float(config["model_weight"]) * xsec_z(model_signal)
    return signal.astype(np.float32)
