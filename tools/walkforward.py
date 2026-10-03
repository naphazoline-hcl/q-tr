"""Purged walk-forward 検証（Train 期間内で、未来ラベルを混ぜずにモデル選択する）。

方針決定はすべて Train（2008-11〜2016-03）で行い、Valid は最終確認にだけ使う。
各フォールド（既定: 2010〜2015 の各年）について

1. その年の初日より前のデータだけで特徴量を作る
2. ラベルは「窓全体が学習終端より前に収まる」行だけ使う（purge）
3. モデルを学習し、その年のシグナルを予測する

を繰り返し、得られた OOS シグナルを公式と同じコスト・重みで採点する。

戦略フォルダの `alpha.py` が実装すべき API（docs/coding_conventions.md に詳述）::

    build_features(splits, start, end=None, codes=None) -> DataFrame   # index=(Date, Code)
    make_label(k, start, end=None) -> Series                         # Train target のみ・窓<=end
    fit_model(features, labels, params) -> model                     # labels は k ごとの DataFrame
    predict_signal(features, model, params) -> Series                # 合成済みの生シグナル

使い方::

    python tools/walkforward.py --strategy strategies/v0_multifactor
    python tools/walkforward.py --strategy strategies/v0_multifactor --folds 2013 2014 2015
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_script as ev  # noqa: E402
from progress import Progress  # noqa: E402

DEFAULT_CONFIG = {
    "history_start": "2004-01-01",
    "fold_years": [2010, 2011, 2012, 2013, 2014, 2015],
    "horizons": [126, 250],
    "label_clip": 0.05,
    "purge_days": 260,
    "smoothing_span": 5,
    "params": {},
}


def load_config(strategy_dir: Path, config_path: Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    candidate = config_path or (strategy_dir / "walkforward_config.json")
    if candidate.exists():
        config.update(json.loads(Path(candidate).read_text(encoding="utf-8")))
    return config


def trading_dates_before(dates: pd.DatetimeIndex, timestamp: pd.Timestamp) -> pd.DatetimeIndex:
    return dates[dates < timestamp]


def evaluate_signal(signal: pd.Series, target: pd.DataFrame) -> dict:
    frame = signal.dropna().to_frame("Return")
    common = frame.index.intersection(target.index)
    frame = frame.loc[common]
    weight = ev.compute_weight(frame).iloc[:, 0]
    target_series = target.iloc[:, 0].reindex(weight.index)
    delta = weight.groupby(level="Code").diff().abs().fillna(weight)
    pl = weight * target_series - ev.TRANSACTION_COST_RATE * delta
    daily = pl.groupby("Date").sum()
    return {
        "sharpe": ev.compute_sr(pl),
        "gross_annual": float((weight * target_series).groupby("Date").sum().mean() * 252),
        "cost_annual": float((ev.TRANSACTION_COST_RATE * delta).groupby("Date").sum().mean() * 252),
        "turnover": float(delta.groupby("Date").sum().mean()),
        "net_annual": float(daily.mean() * 252),
        "vol_annual": float(daily.std() * np.sqrt(252)),
        "yearly_sharpe": {
            str(year): (round(float(g.mean() / g.std() * np.sqrt(252)), 2) if len(g) > 5 and g.std() > 0 else None)
            for year, g in daily.groupby(daily.index.year)
        },
    }


def run(strategy_dir: Path, module_name: str, config: dict, folds: list[int] | None, save_signal: Path | None) -> dict:
    sys.path.insert(0, str(strategy_dir))
    alpha = importlib.import_module(module_name)
    for name in ("build_features", "make_label", "fit_model", "predict_signal"):
        if not hasattr(alpha, name):
            raise AttributeError(f"{module_name}.py に {name}() がありません（docs/coding_conventions.md 参照）")

    fold_years = folds or config["fold_years"]
    purge_days = int(config["purge_days"])
    horizons = list(config["horizons"])
    params = dict(config.get("params", {}))
    smoothing_span = int(config.get("smoothing_span", 5))

    prog = Progress(
        ROOT / "work" / "progress" / f"walkforward_{strategy_dir.name}.json",
        total=len(fold_years) + 2,
        meta={"strategy": strategy_dir.name, "module": module_name, "folds": fold_years, "config": config},
    )

    t0 = time.time()
    features = alpha.build_features(splits=("train",), start=pd.Timestamp(config["history_start"]))
    prog.update(1, f"特徴量構築 完了 shape={features.shape} ({time.time() - t0:.0f}s)")

    signals = []
    per_fold: dict[str, dict] = {}
    dates = pd.DatetimeIndex(features.index.get_level_values("Date").unique()).sort_values()
    date_values = pd.Series(features.index.get_level_values("Date")).to_numpy()

    for offset, year in enumerate(fold_years):
        fold_start = pd.Timestamp(f"{year}-01-01")
        fold_end = pd.Timestamp(f"{year}-12-31")
        train_dates = dates[dates < fold_start]
        train_end = train_dates[-purge_days] if len(train_dates) > purge_days else train_dates[0]

        train_mask = date_values <= np.datetime64(train_end)
        fold_mask = (date_values >= np.datetime64(fold_start)) & (date_values <= np.datetime64(fold_end))
        features_train = features[train_mask]
        features_fold = features[fold_mask]

        labels = {
            f"k{k}": alpha.make_label(k, start=pd.Timestamp(config["history_start"]), end=train_end)
            for k in horizons
        }
        label_frame = pd.DataFrame(labels)
        label_frame = label_frame.loc[label_frame.index.intersection(features_train.index)].dropna(how="all")

        t1 = time.time()
        model = alpha.fit_model(
            features_train.loc[features_train.index.intersection(label_frame.index)], label_frame, params
        )
        signal = alpha.predict_signal(features_fold, model, params)
        if smoothing_span and smoothing_span > 1:
            signal = signal.groupby(level="Code", sort=False).transform(
                lambda x: x.ewm(span=smoothing_span, min_periods=1).mean()
            )
        signals.append(signal)
        per_fold[str(year)] = {
            "train_end": str(pd.Timestamp(train_end).date()),
            "train_rows": int(len(features_train)),
            "label_rows": int(len(label_frame)),
            "predict_rows": int(len(signal)),
            "seconds": round(time.time() - t1, 1),
        }
        prog.update(
            2 + offset,
            f"fold {year}: train_end={pd.Timestamp(train_end).date()} rows={len(signal)} ({time.time() - t1:.0f}s)",
        )

    oos = pd.concat(signals)
    oos = oos[~oos.index.duplicated(keep="first")].sort_index()
    target = pd.read_parquet(ROOT / "input" / "target_1day_train.parquet")
    metrics = evaluate_signal(oos, target)
    metrics["per_fold"] = per_fold
    metrics["config_summary"] = {k: v for k, v in config.items() if k != "params"}
    metrics["params"] = config.get("params", {})
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    if save_signal is not None:
        save_signal.parent.mkdir(parents=True, exist_ok=True)
        oos.to_frame("Return").to_parquet(save_signal)
        metrics["saved_signal"] = str(save_signal)
    prog.done("walk-forward 完了", sharpe=metrics["sharpe"], yearly=metrics["yearly_sharpe"])
    return metrics


def print_report(metrics: dict) -> None:
    print("=" * 72)
    print("purged walk-forward（Train OOS）")
    print(f"  SHARPE      : {metrics['sharpe']:+.4f}")
    print(f"  グロス年率  : {metrics['gross_annual'] * 100:+.2f}%")
    print(f"  コスト年率  : {metrics['cost_annual'] * 100:+.2f}%  （回転率 {metrics['turnover']:.4f}）")
    print(f"  ネット年率  : {metrics['net_annual'] * 100:+.2f}%  ボラ {metrics['vol_annual'] * 100:.2f}%")
    print("  年別 Sharpe : " + "  ".join(f"{k}:{v}" for k, v in metrics["yearly_sharpe"].items()))
    for year, info in metrics["per_fold"].items():
        print(
            f"    fold {year}: train_end={info['train_end']} train={info['train_rows']:,} "
            f"predict={info['predict_rows']:,} ({info['seconds']}s)"
        )
    print(f"  経過時間    : {metrics['elapsed_sec']}s")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="purged walk-forward 検証")
    parser.add_argument("--strategy", required=True, help="戦略フォルダ（例 strategies/v0_multifactor）")
    parser.add_argument("--module", default="alpha")
    parser.add_argument("--config", default=None, help="walkforward_config.json のパス")
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="fold 年（省略時は設定ファイル）")
    parser.add_argument("--json", default=None, help="結果JSONの書き出し先")
    parser.add_argument("--save-signal", default=None, help="OOSシグナルの parquet 保存先")
    args = parser.parse_args()

    strategy_dir = Path(args.strategy).resolve()
    config = load_config(strategy_dir, Path(args.config).resolve() if args.config else None)
    metrics = run(
        strategy_dir,
        args.module,
        config,
        args.folds,
        Path(args.save_signal).resolve() if args.save_signal else None,
    )
    print_report(metrics)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
