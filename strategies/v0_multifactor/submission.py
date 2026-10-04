"""v0_multifactor v2 の推論（提出対象）。学習はしない。train_v2.py が作った成果物を読むだけ。

signal = sum_b w_b z(block_b) + w_model z(model) [+ sector_neutral: w_sector z(model - 業種平均)]
-> 銘柄ごとの EWMA(span=smoothing_span) -> 採点対象 index へ reindex。
数式は alpha_v2.predict_signal と tools/walkforward.py の平滑化に合わせ、同じ関数を呼んでいる。

同梱物（Path(__file__).resolve().parent 基準）: meta_v2.json / models_v2/*.txt / alpha_v2.py /
alpha_features.py。配布 parquet はベース名で相対読みする（cwd = データ展開先）。

メモリ・時間対策（採点は 30 分以内・非力な環境を想定）
- 特徴量は alpha_v2.build_features(start=HISTORY_START)。alpha_features が pyarrow の列指定と
  filters=[("Date", ">=", HISTORY_START)] で読み込み時に絞る。listed_info は 5 列だけ読む
  （ScaleCategory / Sector17Code / Sector33Code / MarketCode / MarginCode）。
- モデル入力は float32 の C 連続行列 1 つへ、列ごとに断面順位を in-place で書き込む
  （float64 の全体コピーや順位化済み DataFrame の複製を作らない）。生の特徴量パネルは
  ブロック合成と行列化が済んだ時点で解放する。
- LightGBM は lgb.Booster(model_file=...) で読み、meta_v2.json の列名・列順序と完全一致を
  確認してから予測する（不一致はエラーで停止）。
- 予測コードは target 系のファイルを一切読まない。採点対象 index は raw_return / beta の
  index の合併で再現する（実データでは target_1day_valid の index と完全一致を確認済み）。
- 既定は Valid だけを覆う。V0_PREDICT_SPLITS=train,valid、または tools/score.py --split train
  から呼ばれたときだけ Train も覆う（Train は学習期間と重なるので in-sample の確認用）。
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import alpha_v2  # noqa: E402

# Valid starts 2016-04-01; 2014-06-01 leaves > 250 trading days for the rolling windows.
HISTORY_START = pd.Timestamp("2014-06-01")
INDEX_SOURCES = ("raw_return_1day", "beta_1day")
SPLITS_ENV = "V0_PREDICT_SPLITS"
META_FILE = HERE / "meta_v2.json"


def load_meta() -> dict:
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    required = ("features", "model_files", "params", "smoothing_span", "block_weights", "model_weight")
    missing = [key for key in required if key not in meta]
    if missing:
        raise KeyError(f"meta_v2.json に {missing} がありません（train_v2.py を再実行してください）")
    return meta


def requested_splits() -> tuple[str, ...]:
    """既定は Valid のみ。Train は明示指定（環境変数）か tools/score.py --split train のときだけ。"""
    splits = {s.strip() for s in os.environ.get(SPLITS_ENV, "").split(",") if s.strip()}
    argv = sys.argv[1:]
    if not splits and any(
        arg == "--split=train" or (arg == "--split" and argv[i + 1:i + 2] == ["train"]) for i, arg in enumerate(argv)
    ):
        splits = {"train", "valid"}
    splits = splits or {"valid"}
    unknown = splits - {"train", "valid"}
    if unknown:
        raise ValueError(f"{SPLITS_ENV} に不明な split: {sorted(unknown)}（train / valid）")
    return tuple(s for s in ("train", "valid") if s in splits)


def scoring_index(split: str) -> pd.MultiIndex:
    """採点対象 index = raw_return と beta の (Date, Code) の合併（target 系は読まない）。"""
    index = None
    with alpha_v2._in_data_dir():
        for source in INDEX_SOURCES:
            name = f"{source}_{split}.parquet"
            if not os.path.exists(name):
                continue
            part = pd.read_parquet(name, columns=["Return"]).index
            index = part if index is None else index.union(part)
    if index is None:
        raise FileNotFoundError(f"{split} の raw_return / beta parquet がデータ展開先にありません")
    return index


def load_boosters(meta: dict) -> dict[str, list[lgb.Booster]]:
    """ホライズンごとの Booster（meta の並び = seed 順）。列名・列順序が meta と違えば停止する。"""
    model_dir = HERE / meta.get("model_dir", "models_v2")
    entries = meta.get("models") or [
        {"file": name, "horizon": name.split("_")[1]} for name in meta["model_files"]
    ]
    expected = list(meta["features"])
    boosters: dict[str, list[lgb.Booster]] = {}
    for entry in entries:
        booster = lgb.Booster(model_file=str(model_dir / entry["file"]))
        names = list(booster.feature_name())
        if names != expected:
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(names, expected)) if a != b][:5]
            raise RuntimeError(
                f"{entry['file']} の特徴量列が meta_v2.json と一致しません "
                f"(model={len(names)}列, meta={len(expected)}列, 最初の差分={diff})"
            )
        boosters.setdefault(entry["horizon"], []).append(booster)
    return boosters


def block_signal(features: pd.DataFrame, meta: dict, config: dict) -> np.ndarray:
    """alpha_v2.predict_signal のブロック部分と同一（regime_mix / fill_missing_blocks も同じ扱い）。"""
    weights = config["block_weights"]
    expansion = None
    if config["regime_mix"]:
        expansion = alpha_v2.regime_expansion(features, {"regime": meta.get("regime", {})}, config)
    expansion_weights = config["regime"]["expansion_block_weights"]
    signal = np.zeros(len(features), dtype=np.float64)
    for block, score in alpha_v2.block_scores(features, config).items():
        z = alpha_v2.xsec_z(score).to_numpy(np.float64)
        if config["fill_missing_blocks"]:
            z = np.nan_to_num(z, nan=0.0)
        weight = float(weights.get(block, 0.0))
        if expansion is not None:
            weight = np.where(expansion, float(expansion_weights.get(block, weight)), weight)
        signal = alpha_v2._add_weighted(signal, weight, z)
    return signal


def ranked_matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """モデル入力: 同一 Date の断面順位 (0, 1]（alpha_v2.rank_for_model と同じ定義）。

    meta の列順で float32 の C 連続行列へ列ごとに in-place で書き込む。SECTOR_COLUMNS はコードのまま。
    """
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise RuntimeError(f"meta_v2.json の特徴量が build_features の出力にありません: {missing}")
    matrix = np.empty((len(features), len(columns)), dtype=np.float32)
    for j, column in enumerate(columns):
        values = features[column]
        if column not in alpha_v2.SECTOR_COLUMNS:
            values = values.groupby(level="Date", sort=False).rank(pct=True)
        matrix[:, j] = values.to_numpy(dtype=np.float32)
    return matrix


def model_prediction(ranked: pd.DataFrame, boosters: dict, how: str) -> pd.Series | None:
    """alpha_v2.model_prediction と同一: seed 平均 -> ホライズンごとに断面 z（how="z"）-> ホライズン平均。

    ranked は meta の列順の DataFrame。Booster の列名・列順序と一致しなければ停止し、
    一致すれば裏の float32 C 連続行列をそのまま渡す（コピーなし）。
    """
    if not boosters:
        return None
    matrix = ranked.to_numpy(dtype=np.float32, copy=False)
    if not matrix.flags["C_CONTIGUOUS"]:
        matrix = np.ascontiguousarray(matrix)
    combined = []
    for horizon, models in boosters.items():
        predictions = []
        for booster in models:
            if list(booster.feature_name()) != list(ranked.columns):
                raise RuntimeError(f"{horizon}: Booster と推論 DataFrame の列名・列順序が一致しません")
            predictions.append(booster.predict(matrix))
        mean = pd.Series(np.mean(predictions, axis=0), index=ranked.index)
        combined.append(alpha_v2.xsec_z(mean).to_numpy(np.float64) if how == "z" else mean.to_numpy())
    return pd.Series(np.mean(combined, axis=0), index=ranked.index, dtype=np.float64)


def smooth_by_code(signal: pd.Series, span: int) -> pd.Series:
    """tools/walkforward.py と同じ銘柄ごとの EWMA（過去方向のみ）。"""
    if span <= 1:
        return signal
    return signal.groupby(level="Code", sort=False).transform(lambda x: x.ewm(span=span, min_periods=1).mean())


def predict_split(split: str, meta: dict, config: dict, boosters: dict) -> pd.Series:
    """1 split 分の平滑化済みシグナル（採点対象 index に reindex 済み）。中間物は関数内で解放する。"""
    if split == "valid":
        splits, start = ("train", "valid"), HISTORY_START
    else:
        splits, start = ("train",), pd.Timestamp(meta.get("history_start") or "2004-01-01")
    features = alpha_v2.build_features(splits=splits, start=start)
    index = features.index
    signal = block_signal(features, meta, config)
    sector = None
    if config["sector_neutral"] and "sector33" in features.columns:
        sector = features["sector33"].to_numpy(dtype=np.float64)
    ranked = None
    if boosters:
        columns = list(meta["features"])
        ranked = pd.DataFrame(ranked_matrix(features, columns), index=index, columns=columns, copy=False)
    del features
    gc.collect()

    prediction = model_prediction(ranked, boosters, config["horizon_combine"]) if ranked is not None else None
    del ranked
    gc.collect()

    if prediction is not None:
        z_model = alpha_v2.xsec_z(prediction).to_numpy(np.float64)
        signal = alpha_v2._add_weighted(signal, float(config["model_weight"]), z_model)
        if sector is not None:
            centered = alpha_v2.demean_in_group(prediction, sector)
            z_sector = alpha_v2.xsec_z(centered).to_numpy(np.float64)
            signal = alpha_v2._add_weighted(signal, float(config["sector_weight"]), z_sector)
    raw = pd.Series(signal, index=index, dtype=np.float32, name="signal")
    smoothed = smooth_by_code(raw, int(meta["smoothing_span"]))
    return smoothed.reindex(scoring_index(split))


def predict() -> pd.DataFrame:
    meta = load_meta()
    # Use the params resolved at training time (not walkforward_config.json) and the frozen blocks.
    config = dict(meta["params"])
    if meta.get("blocks"):
        config["blocks"] = meta["blocks"]
    boosters = load_boosters(meta)
    parts = [predict_split(split, meta, config, boosters) for split in requested_splits()]
    result = parts[0] if len(parts) == 1 else pd.concat(parts)
    result = result[~result.index.duplicated(keep="last")]
    frame = result.astype(np.float64).rename("Return").to_frame()
    frame.index = frame.index.set_names(["Date", "Code"])
    return frame


if __name__ == "__main__":
    started = time.time()
    frame = predict()
    print(f"prediction shape: {frame.shape}  NaN: {float(frame.iloc[:, 0].isna().mean()):.4%}  "
          f"elapsed: {time.time() - started:.0f}s")
