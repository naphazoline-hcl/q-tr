"""採点レポート（公式 evaluate_script と同一の数式で、内訳まで出す）。

公式採点は「日次5分位ロング・ショート → 片道0.1%コスト → 年率Sharpe」。
このスクリプトは `evaluate_script.py` の関数をそのまま import して同じ数式を使い、
Sharpe に加えてグロス／コスト／回転率／ボラ／勝率／RankIC／年別Sharpe／分位別リターン
／最大DDを1枚のレポートにする。モデル選択の判断材料を1コマンドで揃えるための道具。

使い方::

    python tools/score.py --submission strategies/v0_multifactor
    python tools/score.py --submission strategies/v0_multifactor --split train
    python tools/score.py --submission strategies/v0_multifactor --guard --json work/reports/v0_split_valid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate_script as ev  # noqa: E402

TRANSACTION_COST_RATE = ev.TRANSACTION_COST_RATE

SPLIT_FILES = {
    "valid": "target_1day_valid.parquet",
    "train": "target_1day_train.parquet",
}


def load_target(data_dir: Path, split: str) -> tuple[pd.DataFrame, str]:
    name = SPLIT_FILES[split]
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path), path.name


def daily_metrics(pred: pd.DataFrame, target: pd.DataFrame) -> dict:
    weight = ev.compute_weight(pred).iloc[:, 0]
    target_series = target.iloc[:, 0].reindex(weight.index)
    gross = weight * target_series
    delta = weight.groupby(level="Code").diff().abs().fillna(weight)
    cost = TRANSACTION_COST_RATE * delta
    pl = gross - cost

    daily_gross = gross.groupby("Date").sum()
    daily_cost = cost.groupby("Date").sum()
    daily_turnover = delta.groupby("Date").sum()
    daily_pl = pl.groupby("Date").sum()

    result = {
        "sharpe": ev.compute_sr(pl),
        "gross_annual": float(daily_gross.mean() * 252),
        "cost_annual": float(daily_cost.mean() * 252),
        "net_annual": float(daily_pl.mean() * 252),
        "vol_annual": float(daily_pl.std() * np.sqrt(252)),
        "daily_turnover": float(daily_turnover.mean()),
        "hit_rate": float((daily_pl > 0).mean()),
        "max_drawdown": float((daily_pl.cumsum() - daily_pl.cumsum().cummax()).min()),
        "days": int(daily_pl.shape[0]),
        "start": str(daily_pl.index.min().date()),
        "end": str(daily_pl.index.max().date()),
    }
    result["yearly_sharpe"] = {
        str(year): (round(float(group.mean() / group.std() * np.sqrt(252)), 2) if len(group) > 5 and group.std() > 0 else None)
        for year, group in daily_pl.groupby(daily_pl.index.year)
    }
    result["yearly_net_annual_pct"] = {
        str(year): round(float(group.mean() * 252 * 100), 2) for year, group in daily_pl.groupby(daily_pl.index.year)
    }
    result["quantile_mean_target"] = quantile_means(weight, target_series)
    result.update(ic_metrics(pred, target))
    return result


def quantile_means(weight: pd.Series, target_series: pd.Series) -> dict:
    """5分位ごとの平均残差リターン（グロス・等ウェイト、bp/日）。

    分位の割当は公式採点 compute_weight と事実上同一（順位 → 5等分）。
    公式の qcut と違い、ties が多い日でも破綻しないよう順位を5等分する。
    """
    frame = pd.DataFrame({"w": weight, "t": target_series}).dropna(subset=["t"])
    frame["rank"] = frame.groupby(level="Date")["w"].rank(method="first")
    counts = frame.groupby(level="Date")["w"].transform("size")
    frame["q"] = np.ceil(frame["rank"] / (counts / 5)).clip(1, 5).astype("int64")
    means = frame.groupby("q")["t"].mean() * 1e4
    return {f"Q{int(q)}": round(float(v), 2) for q, v in means.items()}


def ic_metrics(pred: pd.DataFrame, target: pd.DataFrame) -> dict:
    frame = pd.DataFrame({"s": pred.iloc[:, 0], "t": target.iloc[:, 0]}).reindex(target.index)
    frame = frame.dropna(subset=["t"])
    by_date = frame.groupby(level="Date")
    ranks = pd.DataFrame({"rs": by_date["s"].rank(), "rt": by_date["t"].rank()})
    grouped = ranks.groupby(level="Date")
    count = grouped["rs"].transform("size")
    cov = (
        (ranks["rs"] - grouped["rs"].transform("mean")) * (ranks["rt"] - grouped["rt"].transform("mean"))
    ).groupby(level="Date").sum()
    sd = grouped["rs"].std(ddof=0) * grouped["rt"].std(ddof=0)
    ic = (cov / (count * sd)).replace([np.inf, -np.inf], np.nan).dropna()
    icir = float(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() > 0 else None
    return {
        "rank_ic": float(ic.mean()),
        "rank_ic_std": float(ic.std()),
        "rank_ic_icir": icir,
        "rank_ic_positive_rate": float((ic > 0).mean()),
    }


def run(submission: str, data_dir: Path, split: str, guard: bool, save_pred: Path | None) -> dict:
    target, target_name = load_target(data_dir, split)
    submission_path = Path(submission).resolve()

    def guarded_read(path, *pargs, **kwargs):
        if any(token in str(path) for token in ("target_1day", "raw_target_1day")):
            raise RuntimeError(f"[lookahead guard] 予測コードが禁止ファイルを読もうとした: {path}")
        return real_read_parquet(path, *pargs, **kwargs)

    if guard:
        real_read_parquet = pd.read_parquet
        pd.read_parquet = guarded_read
        try:
            pred = ev.load_prediction(submission_path, data_dir)
        finally:
            pd.read_parquet = real_read_parquet
    else:
        pred = ev.load_prediction(submission_path, data_dir)

    pred = ev.align_prediction(pred, target)
    metrics = daily_metrics(pred, target)
    metrics["submission"] = str(submission_path)
    metrics["target_file"] = target_name
    if save_pred is not None:
        save_pred.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(save_pred)
        metrics["saved_prediction"] = str(save_pred)
    return metrics


def print_report(metrics: dict) -> None:
    print("=" * 72)
    print(f"submission : {metrics['submission']}")
    print(f"target     : {metrics['target_file']}  ({metrics['start']} .. {metrics['end']}, {metrics['days']} days)")
    print("-" * 72)
    print(f"SHARPE            : {metrics['sharpe']:+.4f}")
    print(f"グロス年率        : {metrics['gross_annual'] * 100:+.2f}%")
    print(f"コスト年率        : {metrics['cost_annual'] * 100:+.2f}%  （日次回転率 {metrics['daily_turnover']:.4f}）")
    print(f"ネット年率        : {metrics['net_annual'] * 100:+.2f}%")
    print(f"年率ボラティリティ: {metrics['vol_annual'] * 100:.2f}%")
    print(f"日次勝率          : {metrics['hit_rate'] * 100:.1f}%")
    print(f"最大DD            : {metrics['max_drawdown'] * 100:.2f}%")
    icir = metrics["rank_ic_icir"]
    icir_text = "n/a" if icir is None else f"{icir:+.2f}"
    print(
        f"RankIC            : {metrics['rank_ic']:+.4f}  (ICIR {icir_text}, "
        f"正率 {metrics['rank_ic_positive_rate'] * 100:.1f}%)"
    )
    print("-" * 72)
    print("分位別 平均残差リターン (bp/日): " + "  ".join(f"{k} {v:+.2f}" for k, v in metrics["quantile_mean_target"].items()))
    print("-" * 72)
    print("年別 Sharpe（ネット）:")
    for year, value in metrics["yearly_sharpe"].items():
        shown = "   n/a" if value is None else f"{value:+.2f}"
        print(f"  {year}: {shown}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="採点レポート")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "input"))
    parser.add_argument("--split", default="valid", choices=["valid", "train"])
    parser.add_argument("--guard", action="store_true", help="予測コードから target 系の読み込みを禁止して実行")
    parser.add_argument("--json", default=None, help="メトリクスの書き出し先")
    parser.add_argument("--save-pred", default=None, help="予測値の parquet 保存先")
    args = parser.parse_args()

    metrics = run(
        args.submission,
        Path(args.data_dir).resolve(),
        args.split,
        args.guard,
        Path(args.save_pred).resolve() if args.save_pred else None,
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
