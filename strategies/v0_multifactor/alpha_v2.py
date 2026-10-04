"""alpha_v2.py — 戦略本体 v2（tools/walkforward.py と submission.py が共有する API 実装）。

v0_multifactor v2: 4ブロックのファクター合成（業種内順位を併用）
                 + LightGBM（k 日平均残差ラベル、k x seed 本）
                 + 業種中心化したモデル予測の合成（sector_neutral）
                 + マクロ2局面のブロック重み切替（regime_mix、既定 OFF）。

improve1（v2 -> v3 候補、CHANGELOG_v2v3.md）で追加したパラメータ（未指定なら I1 と同一の挙動）:
- model_smoothing_span: モデル予測だけに銘柄ごとの EWMA（過去方向のみ）を掛けてから z / 業種中心化する。
  1 以下で無効（I1 と同一）。ブロック側は遅いので触らず、回転の主因と見るモデル成分だけを遅くする。
- label_window: "strict"（I1: 窓 t..t+k-1 が end に収まる行だけ）| "partial"（v1 と同じ:
  end で切った target に逆順 rolling(min_periods=1) を掛ける。境界付近は短い窓になる）。
  horizon_combine（z | mean）と組み合わせて ③ の切り分けに使う。

API（docs/coding_conventions.md §2）::

    build_features(splits=("train",), start=None, end=None) -> DataFrame
    make_label(k, start=None, end=None) -> Series
    fit_model(features, labels, params) -> dict
    predict_signal(features, model, params) -> Series   # 平滑化前の生シグナル

時点管理（採点の禁止事項に対応）
- 特徴量は P1 の alpha_features.build_features（因果的）だけを使う。inf は全列 NaN に統一。
- 順位化・標準化・業種中心化はすべて同一 Date の断面内（groupby Date / (Date, sector33)）。
- 逆順の rolling はラベル生成（make_label）だけで使い、行末に許可マーカーを付ける。
- make_label は Train の target ファイルだけを読み、窓 t..t+k-1 が end を超える行は NaN。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_features import ALL_COLUMNS, V1_COLUMNS, SECTOR_COLUMNS
from alpha_features import build_features as build_v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LABEL_FILE = "target_1day_train.parquet"  # check_lookahead: train-ok

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
    "n_jobs": -1,
    "verbose": -1,
}

# ---------------------------------------------------------------- column sets
# Same value for every stock on a date: after cross-sectional ranking they only encode
# the universe size (a time proxy), so they never enter the model.
REGIME_COLUMNS = [
    "usdjpy_chg20", "usdjpy_chg60", "consumer_level_z", "consumer_diff",
    "topix_cum20", "topix_vol60",
]
# Tie-heavy / seasonal event columns (rank value depends on the tie share of the day).
EVENT_COLUMNS = ["disclosed_5d", "valid_ratio20", "disc_days_bd", "progress_1q"]
# 1-20 day windows: daily rank churn that a 126/250-day label cannot use (turnover only).
FAST_COLUMNS = [
    "r1", "rev1", "rev_oc", "rev_co", "mret", "aret", "rcc3", "rcc5", "rcc10",
    "range1", "clv1", "volratio",
    "res20", "gap20", "skew20", "kurt20", "vol5_20", "illiq5", "volume_z20",
]

MODEL_FEATURE_SETS = {
    "v2_slow": [c for c in ALL_COLUMNS if c not in set(REGIME_COLUMNS + EVENT_COLUMNS + FAST_COLUMNS)],
    "v1": list(V1_COLUMNS),
    "all": [c for c in ALL_COLUMNS if c not in REGIME_COLUMNS],
}

# Block definitions: {block: {column: sign}}. Signed cross-sectional ranks are averaged.
BLOCK_SETS = {
    "v1": {
        "size": {"logsize": -1.0, "illiq60": +1.0, "logturn60": -1.0},
        "value": {"bp": +1.0, "sp": +1.0, "ep_f": +1.0, "ep_a": +1.0, "cfy": +1.0, "div_y": +1.0},
        "quality": {"roe": +1.0, "cfo_ta": +1.0, "eq_ratio": +1.0},
        "lowrisk": {"beta": -1.0, "vol60": -1.0},
    },
    # YTD-cumulative fundamentals (sp, ep_a, cfy, roe) swapped for TTM versions.
    "v2": {
        "size": {"logsize": -1.0, "illiq60": +1.0, "logturn60": -1.0},
        "value": {"bp": +1.0, "sp_ttm": +1.0, "ep_f": +1.0, "ep_ttm": +1.0, "cfy_ttm": +1.0, "div_y": +1.0},
        "quality": {"roe_ttm": +1.0, "cfo_ta": +1.0, "eq_ratio": +1.0},
        "lowrisk": {"beta": -1.0, "vol60": -1.0},
    },
}

DEFAULT_PARAMS = {
    "seeds": [0, 1, 2],
    "label_clip": None,  # None -> top-level "label_clip" of walkforward_config.json (0.05)
    "label_min_frac": 0.5,
    "label_transform": "raw",  # raw | vol | rank
    "label_window": "strict",  # strict (I1) | partial (v1-style truncated windows near `end`)
    "min_label_days": 100,
    "model_features": "v2_slow",  # v2_slow | v1 | all | explicit list
    "drop_features": [],
    "horizon_combine": "z",  # z: per-horizon cross-sectional z then mean | mean: v1 plain mean
    "model_weight": 0.5,
    "model_smoothing_span": 1,  # <= 1: off (I1). Per-code EWMA of the raw model prediction.
    "sector_neutral": True,
    "sector_weight": 0.5,
    "block_set": "v2",  # v2 | v1
    "blocks": None,  # explicit {block: {column: sign}} overrides block_set
    "sector_rank_blocks": ["size", "value"],
    "sector_min_count": 3,
    "block_weights": {"size": 1.0, "value": 0.5, "quality": 0.5, "lowrisk": 0.3},
    "fill_missing_blocks": False,
    "regime_mix": False,
    "regime": {
        "consumer_z_min": 0.0,
        "vol_quantile": 0.5,
        "expansion_block_weights": {"size": 0.5, "value": 0.25, "quality": 0.5, "lowrisk": 0.3},
    },
    "model_params": DEFAULT_MODEL_PARAMS,
}
DEFAULT_LABEL_CLIP = 0.05


# ------------------------------------------------------------- config / data


REPLACE_KEYS = {"blocks", "model_features", "seeds", "drop_features", "sector_rank_blocks"}


def load_config() -> dict:
    path = HERE / "walkforward_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict, override: dict | None) -> dict:
    """Nested dicts are merged key by key (partial overrides); REPLACE_KEYS are replaced whole."""
    out = {k: (_deep_merge(v, None) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACE_KEYS:
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def merged_params(params: dict | None = None) -> dict:
    """DEFAULT_PARAMS <- config["params"] <- params。label_clip 未指定なら config 最上位の値。"""
    config = load_config()
    merged = _deep_merge(DEFAULT_PARAMS, config.get("params"))
    merged = _deep_merge(merged, params)
    if merged.get("label_clip") is None:
        merged["label_clip"] = float(config.get("label_clip", DEFAULT_LABEL_CLIP))
    return merged


def _data_dir() -> Path:
    if (ROOT / "input" / "raw_return_1day_train.parquet").exists():
        return ROOT / "input"
    return Path.cwd()


@contextmanager
def _in_data_dir():
    previous = Path.cwd()
    os.chdir(_data_dir())
    try:
        yield
    finally:
        os.chdir(previous)


# ------------------------------------------------- same-date cross-section ops


def _dates(values: pd.Series | pd.DataFrame) -> pd.Index:
    return values.index.get_level_values("Date")


def _group_keys(values: pd.Series, groups) -> list:
    """(Date, group) keys. Missing group codes become their own bucket (-1), masked by callers."""
    codes = pd.Series(np.asarray(groups, dtype=np.float64), index=values.index).fillna(-1.0)
    return [_dates(values), codes.to_numpy()]


def xsec_rank(values: pd.Series) -> pd.Series:
    """同一 Date の断面順位を (-0.5, 0.5] に（P1 / v1 と同じ定義）。"""
    return (values.groupby(level="Date").rank(pct=True) - 0.5).astype(np.float32)


def xsec_rank_in_group(values: pd.Series, groups, min_count: int = 3) -> pd.Series:
    """同一 (Date, group) 内の中心化順位 (rank - 0.5) / n - 0.5。

    小さい業種でも平均が 0 になる定義（n=1 なら 0）。有効銘柄が min_count 未満の
    グループと業種コード欠損の行は NaN（ブロック平均では全体順位だけが��われる）。
    """
    grouped = values.groupby(_group_keys(values, groups), sort=False)
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    centered = (rank - 0.5) / count - 0.5
    ok = (count >= min_count) & pd.notna(np.asarray(groups, dtype=np.float64))
    return centered.where(ok).astype(np.float32)


def xsec_z(values: pd.Series) -> pd.Series:
    by_date = values.groupby(level="Date")
    centered = values - by_date.transform("mean")
    scale = by_date.transform("std").replace(0.0, np.nan)
    return (centered / scale).clip(-3.0, 3.0).astype(np.float32)


def demean_in_group(values: pd.Series, groups) -> pd.Series:
    """同一 (Date, group) 内の平均を引く（業種中心化。業種コード欠損の行は同日の欠損同士で中心化）。"""
    keys = _group_keys(values, groups)
    return (values - values.groupby(keys, sort=False).transform("mean")).astype(np.float64)


# ----------------------------------------------------------------------- API


def build_features(splits=("train",), start=None, end=None, codes=None) -> pd.DataFrame:
    """P1 の因果的特徴量パネル（index=(Date, Code)、float32）。±inf は全列で NaN に統一する。

    置換した件数は ``features.attrs["inf_replaced"]``（列名 -> 行数）に残す。
    codes は v1 互換の引数（断面特徴量を変えないよう、構築後に行を絞る）。
    """
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    with _in_data_dir():
        features = build_v2(splits=tuple(splits), start=start, end=end)
    if codes is not None:
        wanted = {str(code) for code in codes}
        keep = features.index.get_level_values("Code").astype(str).isin(wanted)
        features = features.loc[keep]
    replaced: dict[str, int] = {}
    for column in features.columns:
        bad = np.isinf(features[column].to_numpy())
        if bad.any():
            replaced[column] = int(bad.sum())
            features[column] = features[column].where(~bad)
    features.attrs["inf_replaced"] = replaced
    return features


def make_label(k: int, start=None, end=None, clip: float | None = None, window: str | None = None) -> pd.Series:
    """翌日から k 営業日の平均残差リターン mean(target[t..t+k-1])。

    - 読むのは Train の target ファイルだけ。end より後の行は読み込み直後に捨てる。
    - 窓の最終日（取引日カレンダー上の t+k-1）が end を超える行は NaN。
    - 上場廃止などで窓内の観測が label_min_frac 未満の行も NaN。
    - clip(label, -label_clip, +label_clip)。label_clip は params -> config の順に解決。
    - window（未指定なら config の label_window）: "strict" は上記どおり。"partial" は v1 の
      alpha.make_label と同じく min_periods=1・窓の打ち切り判定なし（end 直前は短い窓の平均）。
      どちらも end より後の target は読まない（purge は呼び出し側の train_end が保証する）。
    """
    k = int(k)
    params = merged_params(None)
    clip = float(params["label_clip"] if clip is None else clip)
    window = str(window or params.get("label_window", "strict"))
    if window not in ("strict", "partial"):
        raise ValueError(f"unknown label_window: {window!r} (strict | partial)")
    min_periods = max(1, int(np.ceil(k * float(params["label_min_frac"]))))
    if window == "partial":
        min_periods = 1

    with _in_data_dir():
        target = pd.read_parquet(LABEL_FILE).iloc[:, 0]  # check_lookahead: train-ok
    target = target.astype(np.float64).sort_index()
    if end is not None:
        target = target.loc[_dates(target) <= pd.Timestamp(end)]
    if start is not None:
        target = target.loc[_dates(target) >= pd.Timestamp(start)]

    by_code = target.groupby(level="Code", sort=False)
    forward_mean = by_code.transform(  # check_lookahead: train-ok
        lambda s: s.iloc[::-1].rolling(k, min_periods=min_periods).mean().iloc[::-1]  # check_lookahead: train-ok
    )
    calendar = pd.DatetimeIndex(np.unique(_dates(target).to_numpy()))
    window_last = calendar.searchsorted(_dates(target)) + (k - 1)
    inside = window_last <= len(calendar) - 1
    if window == "partial":
        inside = np.ones(len(target), dtype=bool)
    label = forward_mean.where(inside).clip(-clip, clip)
    return label.astype(np.float32).rename(f"k{k}")


def model_columns(features: pd.DataFrame, config: dict) -> list[str]:
    """モデル入力列。業種・市場区分（SECTOR_COLUMNS）は drop_features に無い限り常に素通し。"""
    spec = config["model_features"]
    if isinstance(spec, str):
        if spec not in MODEL_FEATURE_SETS:
            raise ValueError(f"unknown model_features: {spec!r} (choose from {sorted(MODEL_FEATURE_SETS)})")
        spec = MODEL_FEATURE_SETS[spec]
    drop = set(config.get("drop_features") or [])
    columns = [c for c in spec if c in features.columns and c not in drop]
    columns += [c for c in SECTOR_COLUMNS if c in features.columns and c not in drop and c not in columns]
    return columns


def rank_for_model(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """日次断面の順位 (0, 1]（同一 Date のみ）。SECTOR_COLUMNS はコードのまま。列ごとに詰めて省メモリ。"""
    out = np.empty((len(features), len(columns)), dtype=np.float32, order="F")
    for j, column in enumerate(columns):
        values = features[column]
        if column not in SECTOR_COLUMNS:
            values = values.groupby(level="Date", sort=False).rank(pct=True)
        out[:, j] = values.to_numpy(dtype=np.float32)
    return pd.DataFrame(out, index=features.index, columns=columns, copy=False)


def transform_label(label: pd.Series, features: pd.DataFrame, how: str) -> pd.Series:
    """raw: そのまま / vol: 同日中央値で割った相対 vol60 で割る / rank: 同日断面の順位 (0, 1]。"""
    if how == "raw":
        return label
    if how == "vol":
        vol = features["vol60"].reindex(label.index).astype(np.float64)
        relative = vol / vol.groupby(level="Date").transform("median")
        return (label / relative.where(relative > 0)).astype(np.float32)
    if how == "rank":
        return label.groupby(level="Date").rank(pct=True).astype(np.float32)
    raise ValueError(f"unknown label_transform: {how!r} (raw | vol | rank)")


def _fit_regime(features: pd.DataFrame, config: dict) -> dict:
    """局面判定のしきい値（topix_vol60 の分位点）を学習期間の日次値だけから決める。"""
    if "topix_vol60" not in features.columns or len(features) == 0:
        return {"vol_threshold": float("nan")}
    daily = features["topix_vol60"].groupby(level="Date").first().dropna()
    quantile = float(config["regime"]["vol_quantile"])
    return {"vol_threshold": float(daily.quantile(quantile)) if len(daily) else float("nan")}


def fit_model(features: pd.DataFrame, labels: pd.DataFrame, params: dict | None = None) -> dict:
    """len(horizons) x len(seeds) 本の LightGBM を学習して dict で返す（保存は呼び出し側）。

    ラベルのある日付が min_label_days 未満のホライズンは学習しない（skipped に記録）。
    1本も学習できなければ models は空で、predict_signal はブロック合成だけを返す。
    """
    config = merged_params(params)
    model_params = dict(config["model_params"])
    model_params.pop("random_state", None)
    model_params.setdefault("n_jobs", -1)
    clip = float(config["label_clip"])
    columns = model_columns(features, config)
    result = {
        "version": "alpha_v2",
        "models": {},
        "columns": columns,
        "skipped": {},
        "label_days": {},
        "regime": _fit_regime(features, config),
    }
    labels = labels.reindex(features.index)
    if len(features) == 0:
        return result
    ranked = rank_for_model(features, columns)
    categorical = [c for c in columns if c in SECTOR_COLUMNS]
    for horizon in labels.columns:
        y = transform_label(labels[horizon].clip(-clip, clip), features, config["label_transform"])
        mask = y.notna().to_numpy()
        n_days = int(_dates(y)[mask].nunique())
        result["label_days"][horizon] = n_days
        if n_days < int(config["min_label_days"]):
            result["skipped"][horizon] = n_days
            continue
        for seed in config["seeds"]:
            booster = lgb.LGBMRegressor(random_state=int(seed), **model_params)
            booster.fit(ranked.loc[mask], y.loc[mask], categorical_feature=categorical)
            result["models"][f"{horizon}_s{seed}"] = booster
    return result


def resolve_blocks(config: dict) -> dict:
    blocks = config.get("blocks") or BLOCK_SETS.get(config["block_set"])
    if not blocks:
        raise ValueError(f"unknown block_set: {config['block_set']!r} (choose from {sorted(BLOCK_SETS)})")
    return blocks


def block_scores(features: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    """各ブロック = 符号付き断面順位の等ウェイト平均（欠損は無視）。

    sector_rank_blocks に含むブロックは、各列の sector33 内順位も平均に加える（業種の偏りを半減）。
    """
    sector = features["sector33"].to_numpy() if "sector33" in features.columns else None
    in_sector = set(config.get("sector_rank_blocks") or [])
    min_count = int(config["sector_min_count"])
    scores: dict[str, pd.Series] = {}
    for block, signs in resolve_blocks(config).items():
        missing = [c for c in signs if c not in features.columns]
        if missing:
            raise KeyError(f"block {block!r}: columns not in features: {missing}")
        parts = {}
        for column, sign in signs.items():
            parts[column] = float(sign) * xsec_rank(features[column])
            if block in in_sector and sector is not None:
                ranked = xsec_rank_in_group(features[column], sector, min_count)
                parts[f"{column}@sector33"] = float(sign) * ranked
        scores[block] = pd.DataFrame(parts).mean(axis=1, skipna=True).astype(np.float32)
    return scores


def regime_expansion(features: pd.DataFrame, model: dict, config: dict) -> np.ndarray:
    """拡張期フラグ: consumer_level_z > consumer_z_min かつ topix_vol60 < 学習期間の分位点。

    どちらも P1 の因果的特徴量（同日値）。欠損・しきい値なしは通常期として扱う。
    """
    threshold = float((model or {}).get("regime", {}).get("vol_threshold", float("nan")))
    z_min = float(config["regime"]["consumer_z_min"])
    consumer = features["consumer_level_z"].to_numpy(dtype=np.float64)
    vol = features["topix_vol60"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (consumer > z_min) & (vol < threshold)


def model_prediction(features: pd.DataFrame, model: dict, how: str = "z") -> pd.Series | None:
    """seed 平均 -> ホライズンごとに断面 z（how="z"）-> ホライズン平均。how="mean" は v1 と同じ単純平均。"""
    boosters = (model or {}).get("models", {})
    if not boosters:
        return None
    columns = model.get("columns") or list(next(iter(boosters.values())).feature_name_)
    ranked = rank_for_model(features, columns)
    per_horizon: dict[str, list[np.ndarray]] = {}
    for name, booster in boosters.items():
        horizon = name.rsplit("_s", 1)[0]
        per_horizon.setdefault(horizon, []).append(booster.predict(ranked[list(booster.feature_name_)]))
    del ranked
    combined = []
    for predictions in per_horizon.values():
        mean = pd.Series(np.mean(predictions, axis=0), index=features.index)
        combined.append(xsec_z(mean).to_numpy(np.float64) if how == "z" else mean.to_numpy())
    return pd.Series(np.mean(combined, axis=0), index=features.index, dtype=np.float64)


def _add_weighted(signal: np.ndarray, weight, z: np.ndarray) -> np.ndarray:
    """signal + weight * z。重み 0 の行は z の欠損を持ち込まない。"""
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), z.shape)
    return signal + np.where(weight == 0.0, 0.0, weight * z)


def predict_signal(features: pd.DataFrame, model: dict, params: dict | None = None) -> pd.Series:
    """signal = sum_b w_b z(block_b) + w_model z(model) + [sector_neutral] w_sector z(model - 業種平均)。

    平滑化前の生シグナル（index は features と同じ）。z は同一 Date の断面 z（±3 で clip）。
    """
    config = merged_params(params)
    weights = config["block_weights"]
    expansion = regime_expansion(features, model, config) if config["regime_mix"] else None
    expansion_weights = config["regime"]["expansion_block_weights"]

    signal = np.zeros(len(features), dtype=np.float64)
    for block, score in block_scores(features, config).items():
        z = xsec_z(score).to_numpy(np.float64)
        if config["fill_missing_blocks"]:
            z = np.nan_to_num(z, nan=0.0)
        weight = float(weights.get(block, 0.0))
        if expansion is not None:
            weight = np.where(expansion, float(expansion_weights.get(block, weight)), weight)
        signal = _add_weighted(signal, weight, z)

    prediction = model_prediction(features, model, config["horizon_combine"])
    sector = features["sector33"].to_numpy(np.float64) if "sector33" in features.columns else None
    signal = add_model_terms(signal, prediction, sector, config)
    return pd.Series(signal, index=features.index, dtype=np.float32, name="signal")


def smooth_by_code(values: pd.Series, span: int) -> pd.Series:
    """銘柄ごとの EWMA（過去方向のみ。tools/walkforward.py の平滑化と同じ式）。span <= 1 はそのまま返す。

    行は (Date, Code) の Date 昇順を前提とする（build_features の出力順）。
    """
    span = int(span or 1)
    if span <= 1:
        return values
    return values.groupby(level="Code", sort=False).transform(lambda x: x.ewm(span=span, min_periods=1).mean())


def add_model_terms(signal: np.ndarray, prediction: pd.Series | None, sector, config: dict) -> np.ndarray:
    """signal + w_model z(p) + [sector_neutral] w_sector z(p - 業種平均)、p = EWMA_code(prediction)。

    predict_signal と submission.py が共有する（モデル項の式を二重管理しない）。
    model_smoothing_span <= 1（既定・I1）なら p = prediction で従来と同一。
    """
    if prediction is None:
        return signal
    prediction = smooth_by_code(prediction, int(config.get("model_smoothing_span", 1) or 1))
    signal = _add_weighted(signal, float(config["model_weight"]), xsec_z(prediction).to_numpy(np.float64))
    if config["sector_neutral"] and sector is not None:
        centered = demean_in_group(prediction, sector)
        signal = _add_weighted(signal, float(config["sector_weight"]), xsec_z(centered).to_numpy(np.float64))
    return signal
