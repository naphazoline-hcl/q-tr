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
               skipped: dict, history_start: pd.Timestamp, config_hash: str) -> dict:
    """submission.py が推論に使う設定一式。params は学習時に解決済みの完全な辞書を残す。"""
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
    }


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
    jobs = [(k, seed) for k in horizons for seed in seeds]

    model_dir = strategy_dir / MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress) if args.progress else root / "work" / "progress" / "train_v2.json"
    prog = Progress(
        progress_path,
        total=len(jobs) + 2,
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
        for i, (k, seed) in enumerate(jobs, start=1):
            horizon, name = f"k{k}", f"lgbm_k{k}_s{seed}.txt"
            path, step = model_dir / name, 2 + i
            entry = {"file": name, "horizon": horizon, "seed": seed}
            if args.resume and is_complete(path, columns):
                prog.update(step, f"{name} 保存済みのため読み飛ばし（--resume）", artifact=str(path))
                print(f"[train_v2] skip {name} (resume)", flush=True)
                models.append(entry)
                continue
            prog.update(step, f"{name} 学習開始 rows={label_rows[horizon]:,}")
            t0 = time.time()
            result = alpha.fit_model(train, labels[[horizon]], dict(params, seeds=[seed]))
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

        meta = build_meta(alpha, strategy_dir, config, params, horizons, seeds, columns, train, labels,
                          models, skipped, history_start, config_hash)
        write_json_atomic(strategy_dir / META_NAME, meta)
        prog.done("全ジョブ完了 meta_v2.json 書き出し", model_files=meta["model_files"], skipped=skipped)
        print(f"[train_v2] meta: {strategy_dir / META_NAME} models={len(models)} skipped={skipped}", flush=True)
    except BaseException as exc:
        prog.failed(f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
