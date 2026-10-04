#!/usr/bin/env python3
"""train_v2.py: alpha_v2 の最終モデルを全 Train データで学習する（Valid のラベルは使わない）。

    python strategies/v0_multifactor/train_v2.py --strategy strategies/v0_multifactor [--resume]

- 設定: <strategy>/walkforward_config.json の horizons / params（seeds を含む）/ history_start /
  smoothing_span をそのまま使う（このファイルに設定値を書かない）。
- ラベル: alpha_v2.make_label(k, start=history_start, end=None)。窓 t..t+k-1 が Train 末尾を
  超える行は NaN（strict）。walk-forward と同じく「どれかのラベルがある行」だけを学習に渡す。
- 学習: alpha_v2.fit_model を (k, seed) ごとに 1 本ずつ呼び、終わるたびに
  models_v2/lgbm_k{k}_s{seed}.txt へ保存する（一時ファイル -> os.replace で書きかけを残さない）。
- 進捗: <repo>/work/progress/train_v2.json（tools/progress.py の Progress）と同じ場所の STATE.md。
  --resume を付けると、保存済みで列順序が一致するモデルを読み飛ばす。
- 終了時: <strategy>/meta_v2.json（submission.py が読む）。
- improve2（P5）: LGBM の後に alpha.fit_extras を 1 回呼ぶ（ensemble_weights が非ゼロのモデルだけ）。
  Ridge の係数は meta_v2.json の "ridge"、rank モデルは models_v2/rank_<name>.txt（"rank_models"）。
  turnover_control=auto で選ばれた span は params.model_smoothing_span に書き戻す（提出側も同じ span）。
  slow_profile のホライズン指定があれば、そのホライズンの LGBM だけを学習する。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR_NAME = "models_v2"
META_NAME = "meta_v2.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="alpha_v2 の最終モデルを全 Train データで学習する")
    parser.add_argument("--strategy", required=True, help="戦略フォルダ（例 strategies/v0_multifactor）")
    parser.add_argument("--resume", action="store_true", help="保存済みのモデルを読み飛ばして続きから学習する")
    parser.add_argument("--module", default="alpha_v2", help="戦略本体モジュール名（既定 alpha_v2）")
    parser.add_argument("--progress", default=None, help="進捗 JSONL のパス（既定 <repo>/work/progress/train_v2.json）")
    return parser.parse_args(argv)


def setup_paths(strategy_dir: Path) -> Path:
    """tools/（progress.py）と戦略フォルダを sys.path に追加し、リポジトリのルートを返す。"""
    for root in (strategy_dir.parent.parent, HERE.parent.parent):
        if (root / "tools" / "progress.py").exists():
            sys.path.insert(0, str(root / "tools"))
            sys.path.insert(0, str(strategy_dir))
            return root
    raise FileNotFoundError("tools/progress.py が見つかりません（--strategy にはリポジトリ内の戦略フォルダを指定）")


def load_config(strategy_dir: Path) -> tuple[dict, str]:
    path = strategy_dir / "walkforward_config.json"
    text = path.read_text(encoding="utf-8")
    config = json.loads(text)
    for key in ("horizons", "params"):
        if key not in config:
            raise KeyError(f"{path} に {key!r} がありません")
    return config, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_booster(booster: lgb.Booster, path: Path) -> None:
    """一時ファイルへ書いてから置き換える（中断しても壊れたモデルファイルを残さない）。"""
    tmp = path.with_name(path.name + ".tmp")
    booster.save_model(str(tmp))
    os.replace(tmp, path)


def is_complete(path: Path, columns: list[str]) -> bool:
    """--resume 用: 読み込めて、特徴量の列名と順序が今回の学習と一致するモデルだけを完了扱いにする。"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        names = lgb.Booster(model_file=str(path)).feature_name()
    except Exception:  # noqa: BLE001 - unreadable file -> retrain
        return False
    return list(names) == list(columns)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _day(value) -> str | None:
    return None if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def build_meta(alpha, strategy_dir: Path, config: dict, params: dict, horizons: list[int], seeds: list[int],
               columns: list[str], train: pd.DataFrame, labels: pd.DataFrame, models: list[dict],
               skipped: dict, history_start: pd.Timestamp, config_hash: str, extras: dict | None = None) -> dict:
    """submission.py が推論に使う設定一式。params は学習時に解決済みの完全な辞書を残す。"""
    extras = extras or {}
    dates = train.index.get_level_values("Date")
    label_dates = {c: labels.index.get_level_values("Date")[labels[c].notna().to_numpy()] for c in labels.columns}
    return {
        "strategy": strategy_dir.name,
        "version": "alpha_v2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": MODEL_DIR_NAME,
        "model_files": [m["file"] for m in models],
        "models": models,
        "features": list(columns),
        "sector_features": [c for c in columns if c in alpha.SECTOR_COLUMNS],
        "horizons": horizons,
        "seeds": seeds,
        "label_clip": float(params["label_clip"]),
        "label_transform": params["label_transform"],
        "label_min_frac": float(params["label_min_frac"]),
        "smoothing_span": int(config.get("smoothing_span", 5)),
        "model_weight": float(params["model_weight"]),
        "horizon_combine": params["horizon_combine"],
        "sector_neutral": bool(params["sector_neutral"]),
        "sector_weight": float(params["sector_weight"]),
        "regime_mix": bool(params["regime_mix"]),
        "regime": alpha._fit_regime(train, params),
        "block_weights": dict(params["block_weights"]),
        "blocks": alpha.resolve_blocks(params),
        "params": params,
        "training_rows": int(len(train)),
        "training_from": _day(dates.min()) if len(dates) else None,
        "training_to": _day(dates.max()) if len(dates) else None,
        "label_rows": {c: int(labels[c].notna().sum()) for c in labels.columns},
        "label_to": {c: _day(d.max()) if len(d) else None for c, d in label_dates.items()},
        "skipped": skipped,
        "history_start": _day(history_start),
        "config_sha256_16": config_hash,
        "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__},
        "ensemble_weights": dict(params.get("ensemble_weights") or {}),
        "ridge": extras.get("ridge"),
        "rank_models": extras.get("rank_models", []),
        "regime_v4": extras.get("regime_v4") or alpha.regime_v4.fit_thresholds(train, params["regime_v4"]),
        "turnover_cap": float(params.get("turnover_cap", float("nan"))),
        "turnover_estimates": extras.get("turnover_estimates"),
    }


def fit_and_save_extras(alpha, train: pd.DataFrame, labels: pd.DataFrame, params: dict, columns: list[str],
                        fitted_models: dict, model_dir: Path) -> dict:
    """alpha.fit_extras を 1 回呼び、rank モデルを models_v2/rank_<name>.txt に保存する（重み 0 は何もしない）。"""
    model = {"models": fitted_models, "columns": list(columns),
             "regime": alpha._fit_regime(train, params),
             "regime_v4": alpha.regime_v4.fit_thresholds(train, params["regime_v4"])}
    alpha.fit_extras(train, labels, params, model)
    extras = {"ridge": model.get("ridge"), "regime_v4": model["regime_v4"], "rank_models": [],
              "turnover_estimates": model.get("turnover_estimates"),
              "model_smoothing_span": model.get("model_smoothing_span")}
    for name, fitted in (model.get("rank") or {}).items():
        file_name = f"rank_{name}.txt"
        save_booster(fitted.booster_, model_dir / file_name)
        extras["rank_models"].append({"file": file_name, "name": name})
    return extras


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strategy_dir = Path(args.strategy).resolve()
    root = setup_paths(strategy_dir)
    from progress import Progress  # noqa: E402  (tools/ is on sys.path now)

    alpha = importlib.import_module(args.module)
    config, config_hash = load_config(strategy_dir)
    params = alpha.merged_params(config["params"])
    if "seeds" in config:
        params["seeds"] = list(config["seeds"])
    horizons = [int(k) for k in config["horizons"]]
    seeds = [int(s) for s in params["seeds"]]
    history_start = pd.Timestamp(config.get("history_start", "2004-01-01"))
    if params.get("profile_horizons"):
        horizons = [k for k in horizons if f"k{k}" in params["profile_horizons"]]
    weights = alpha.ensemble.ensemble_weights(params)
    jobs = [(k, seed) for k in horizons for seed in seeds] if weights["lgbm"] != 0.0 else []

    model_dir = strategy_dir / MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress) if args.progress else root / "work" / "progress" / "train_v2.json"
    prog = Progress(
        progress_path,
        total=len(jobs) + 3,
        meta={"strategy": strategy_dir.name, "module": args.module, "resume": args.resume,
              "jobs": [f"k{k}_s{s}" for k, s in jobs], "config_sha256_16": config_hash},
    )
    try:
        t0 = time.time()
        features = alpha.build_features(splits=("train",), start=history_start)
        prog.update(1, f"特徴量構築 完了 shape={features.shape} ({time.time() - t0:.0f}s)")
        print(f"[train_v2] features {features.shape} ({time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        labels = pd.DataFrame({f"k{k}": alpha.make_label(k, start=history_start, end=None) for k in horizons})
        labels = labels.loc[labels.index.intersection(features.index)].dropna(how="all")
        train = features.loc[features.index.intersection(labels.index)]
        del features
        labels = labels.reindex(train.index)
        columns = alpha.model_columns(train, params)
        label_rows = {c: int(labels[c].notna().sum()) for c in labels.columns}
        prog.update(2, f"ラベル作成 完了 rows={len(train):,} label_rows={label_rows} ({time.time() - t0:.0f}s)")
        print(f"[train_v2] train rows={len(train):,} label_rows={label_rows} columns={len(columns)}", flush=True)

        models: list[dict] = []
        skipped: dict[str, int] = {}
        fitted_models: dict[str, object] = {}
        # Per-job LGBM fits must not also fit the extras (ridge / rank / auto span): done once below.
        lgbm_only = dict(params, ensemble_weights=dict(weights, ridge=0.0, rank=0.0), turnover_control="off")
        for i, (k, seed) in enumerate(jobs, start=1):
            horizon, name = f"k{k}", f"lgbm_k{k}_s{seed}.txt"
            path, step = model_dir / name, 2 + i
            entry = {"file": name, "horizon": horizon, "seed": seed}
            if args.resume and is_complete(path, columns):
                prog.update(step, f"{name} 保存済みのため読み飛ばし（--resume）", artifact=str(path))
                print(f"[train_v2] skip {name} (resume)", flush=True)
                models.append(entry)
                fitted_models[f"{horizon}_s{seed}"] = lgb.Booster(model_file=str(path))
                continue
            prog.update(step, f"{name} 学習開始 rows={label_rows[horizon]:,}")
            t0 = time.time()
            result = alpha.fit_model(train, labels[[horizon]], dict(lgbm_only, seeds=[seed]))
            if list(result["columns"]) != list(columns):
                raise RuntimeError(f"fit_model の列順序が想定と異なります: {result['columns'][:5]}...")
            fitted = result["models"].get(f"{horizon}_s{seed}")
            if fitted is None:
                skipped[name] = int(result["skipped"].get(horizon, 0))
                prog.update(step, f"{name} 学習なし（ラベル日数 {skipped[name]} < min_label_days）")
                continue
            save_booster(fitted.booster_, path)
            seconds = round(time.time() - t0, 1)
            prog.update(step, f"{name} 完了 ({seconds:.0f}s)", artifact=str(path), seconds=seconds)
            print(f"[train_v2] saved {name} rows={label_rows[horizon]:,} ({seconds:.0f}s)", flush=True)
            models.append(entry)
            fitted_models[f"{horizon}_s{seed}"] = fitted

        t0 = time.time()
        extras = fit_and_save_extras(alpha, train, labels[[f"k{k}" for k in horizons]], params, columns,
                                     fitted_models, model_dir)
        if extras.get("model_smoothing_span"):
            params["model_smoothing_span"] = int(extras["model_smoothing_span"])
        prog.update(len(jobs) + 3, f"extras 完了 ridge={bool(extras.get('ridge'))} "
                    f"rank={len(extras.get('rank_models', []))} span={params.get('model_smoothing_span')} "
                    f"({time.time() - t0:.0f}s)", turnover_estimates=extras.get("turnover_estimates"))
        meta = build_meta(alpha, strategy_dir, config, params, horizons, seeds, columns, train, labels,
                          models, skipped, history_start, config_hash, extras)
        write_json_atomic(strategy_dir / META_NAME, meta)
        prog.done("全ジョブ完了 meta_v2.json 書き出し", model_files=meta["model_files"], skipped=skipped)
        print(f"[train_v2] meta: {strategy_dir / META_NAME} models={len(models)} skipped={skipped}", flush=True)
    except BaseException as exc:
        prog.failed(f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
