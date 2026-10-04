"""sweep_improve1.py — improve1 の切り分け（①回転率 ②年別劣化 ③horizon_combine z / strict ラベル）。

学習はラベル設定（label_window = strict | partial）ごとに1回だけ行い、予測側の設定
（重み・平滑化・horizon_combine・sector_rank_blocks）は保存した「成分」から再合成して評価する。
予測側の変更は再学習が要らないので、1回の学習で多数の候補を Train OOS と Valid の両方で比べられる。

Stage oos  : tools/walkforward.py と同じ fold / purge / train_end で fold ごとに学習し、fold 年の成分を保存。
Stage valid: train_v2.py と同じく Train 全期間（end=None）で学習し、submission.py と同じ期間
             （2014-06-01〜）の成分を保存（Valid の評価は採点対象 index に揃える）。
Stage eval : VARIANTS を再合成 -> 銘柄ごと EWMA -> Train OOS は walkforward.evaluate_signal、
             Valid は score.daily_metrics（どちらも既存ツールと同じ式）で評価し JSON / Markdown を出力。

成分（保存列）: blk_<block>（sector_rank_blocks あり）/ blkg_<block>（全体順位のみ）/
pred_k<h>（ホライズン別 seed 平均の生予測）/ sector33 / fold。

使い方（リポジトリ直下で）:
  python tools/sweep_improve1.py --strategy strategies/v0_multifactor --module alpha_v2
  python tools/sweep_improve1.py --stage eval          # 成分の保存後、評価だけやり直す
  python tools/sweep_improve1.py --labels strict       # partial ラベルの学習を省く（③の一部が欠ける）
中断しても work/sweep_improve1/ に保存済みの成分は読み飛ばす（進捗: work/progress/sweep_improve1.json）。
target_1day_valid.parquet を読むのはこのローカル評価ツールだけ（提出フォルダのコードは読まない）。
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
import score as sc  # noqa: E402
import walkforward as wf  # noqa: E402
from progress import Progress  # noqa: E402

VALID_HISTORY_START = pd.Timestamp("2014-06-01")  # same as submission.HISTORY_START
WORK = ROOT / "work" / "sweep_improve1"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="improve1 の切り分けスイープ（学習1回 + 予測側の再合成）")
    parser.add_argument("--strategy", default="strategies/v0_multifactor")
    parser.add_argument("--module", default="alpha_v2")
    parser.add_argument("--stage", default="all", choices=["all", "oos", "valid", "eval"])
    parser.add_argument("--labels", default="strict,partial", help="学習するラベル設定（カンマ区切り）")
    parser.add_argument("--out", default="work/reports/sweep_improve1", help="出力の接頭辞（.json / .md）")
    return parser.parse_args(argv)


def load_alpha(strategy_dir: Path, module_name: str):
    sys.path.insert(0, str(strategy_dir))
    return importlib.import_module(module_name)


def comp_path(label: str, part: str) -> Path:
    return WORK / f"comp_{label}_{part}.parquet"


def components(alpha, features: pd.DataFrame, model: dict, params: dict, horizons: list[int]) -> pd.DataFrame:
    """予測側の再合成に必要な成分（alpha_v2.predict_signal の部品と同じ関数で作る）。"""
    config = alpha.merged_params(params)
    out: dict[str, np.ndarray] = {}
    for prefix, rank_blocks in (("blk", config["sector_rank_blocks"]), ("blkg", [])):
        scores = alpha.block_scores(features, {**config, "sector_rank_blocks": rank_blocks})
        for block, score in scores.items():
            out[f"{prefix}_{block}"] = score.to_numpy(np.float32)
    boosters = (model or {}).get("models", {})
    per_horizon: dict[str, list[np.ndarray]] = {}
    if boosters:
        ranked = alpha.rank_for_model(features, model["columns"])
        for name, booster in boosters.items():
            horizon = name.rsplit("_s", 1)[0]
            per_horizon.setdefault(horizon, []).append(booster.predict(ranked[list(booster.feature_name_)]))
        del ranked
    for k in horizons:
        preds = per_horizon.get(f"k{k}")
        out[f"pred_k{k}"] = np.mean(preds, axis=0) if preds else np.full(len(features), np.nan)
    out["sector33"] = features["sector33"].to_numpy(np.float64)
    return pd.DataFrame(out, index=features.index)


def stage_oos(alpha, config: dict, label: str, prog: Progress, step: int) -> int:
    """walkforward.run と同じ fold 分割で学習し、fold 年の成分を fold ごとに保存（再開可）。"""
    params = {**config.get("params", {}), "label_window": label}
    horizons = [int(k) for k in config["horizons"]]
    history_start = pd.Timestamp(config["history_start"])
    purge_days = int(config["purge_days"])
    todo = [y for y in config["fold_years"] if not comp_path(label, str(y)).exists()]
    if not todo:
        return step + len(config["fold_years"])
    features = alpha.build_features(splits=("train",), start=history_start)
    dates = pd.DatetimeIndex(features.index.get_level_values("Date").unique()).sort_values()
    date_values = pd.Series(features.index.get_level_values("Date")).to_numpy()
    for year in config["fold_years"]:
        step += 1
        path = comp_path(label, str(year))
        if path.exists():
            prog.update(step, f"oos {label} {year} skip（保存済み）")
            continue
        t0 = time.time()
        fold_start, fold_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        train_dates = dates[dates < fold_start]
        train_end = train_dates[-purge_days] if len(train_dates) > purge_days else train_dates[0]
        features_train = features[date_values <= np.datetime64(train_end)]
        features_fold = features[(date_values >= np.datetime64(fold_start)) & (date_values <= np.datetime64(fold_end))]
        labels = pd.DataFrame({
            f"k{k}": alpha.make_label(k, start=history_start, end=train_end, window=label) for k in horizons
        })
        labels = labels.loc[labels.index.intersection(features_train.index)].dropna(how="all")
        model = alpha.fit_model(features_train.loc[features_train.index.intersection(labels.index)], labels, params)
        frame = components(alpha, features_fold, model, params, horizons)
        frame["fold"] = int(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        prog.update(step, f"oos {label} {year} 完了 models={len(model['models'])} ({time.time() - t0:.0f}s)",
                    artifact=str(path), skipped=model.get("skipped", {}))
    return step


def stage_valid(alpha, config: dict, label: str, prog: Progress, step: int) -> int:
    """Train 全期間で学習（train_v2 と同じ）し、2014-06-01 以降の成分を保存（再開可）。"""
    step += 1
    path = comp_path(label, "valid")
    if path.exists():
        prog.update(step, f"valid {label} skip（保存済み）")
        return step
    t0 = time.time()
    params = {**config.get("params", {}), "label_window": label}
    horizons = [int(k) for k in config["horizons"]]
    history_start = pd.Timestamp(config["history_start"])
    features = alpha.build_features(splits=("train",), start=history_start)
    labels = pd.DataFrame({f"k{k}": alpha.make_label(k, start=history_start, end=None, window=label) for k in horizons})
    labels = labels.loc[labels.index.intersection(features.index)].dropna(how="all")
    model = alpha.fit_model(features.loc[features.index.intersection(labels.index)], labels, params)
    del features
    features = alpha.build_features(splits=("train", "valid"), start=VALID_HISTORY_START)
    frame = components(alpha, features, model, params, horizons)
    frame["fold"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    prog.update(step, f"valid {label} 完了 models={len(model['models'])} ({time.time() - t0:.0f}s)", artifact=str(path))
    return step


# ------------------------------------------------------------------ variants
# Overrides relative to I1 (= current walkforward_config params with model_smoothing_span=1, strict labels).
VARIANTS: dict[str, dict] = {
    "I1": {},
    "K1_msmooth10": {"model_smoothing_span": 10},  # shipped default of improve1
    # (1) turnover levers
    "msmooth5": {"model_smoothing_span": 5},
    "msmooth20": {"model_smoothing_span": 20},
    "msmooth40": {"model_smoothing_span": 40},
    "span8": {"smoothing_span": 8},
    "span10": {"smoothing_span": 10},
    "neutral_only": {"model_weight": 0.0, "sector_weight": 0.5},
    "mw025": {"model_weight": 0.25},
    "K1_mw025": {"model_smoothing_span": 10, "model_weight": 0.25},
    "model_half": {"model_weight": 0.25, "sector_weight": 0.25},
    # (3) horizon_combine x label_window (2x2 together with I1)
    "hc_mean": {"horizon_combine": "mean"},
    "partial": {"label": "partial"},
    "partial_hc_mean": {"label": "partial", "horizon_combine": "mean"},
    # (2) sector-bet hypothesis and sanity checks against measured A / v1
    "no_sector_rank_blocks": {"sector_rank_blocks": False},
    "A_like": {"sector_neutral": False},
    "v1_like": {"label": "partial", "horizon_combine": "mean", "sector_rank_blocks": False, "sector_neutral": False},
    # combinations with the shipped default
    "K1_no_sector_rank_blocks": {"model_smoothing_span": 10, "sector_rank_blocks": False},
    "K1_partial_hc_mean": {"model_smoothing_span": 10, "label": "partial", "horizon_combine": "mean"},
    "K1_msmooth20_span8": {"model_smoothing_span": 20, "smoothing_span": 8},
}
# Single-component signals (yearly Sharpe of each part alone, for diagnosis (2)).
SOLO: dict[str, dict] = {
    **{f"solo_{b}": {"block_weights": {b: 1.0}, "model_weight": 0.0, "sector_weight": 0.0}
       for b in ("size", "value", "quality", "lowrisk")},
    "solo_model": {"block_weights": {}, "model_weight": 1.0, "sector_weight": 0.0},
    "solo_model_sector": {"block_weights": {}, "model_weight": 0.0, "sector_weight": 1.0},
}


def base_spec(config: dict) -> dict:
    params = config.get("params", {})
    return {
        "label": "strict",
        "horizon_combine": params.get("horizon_combine", "z"),
        "sector_rank_blocks": bool(params.get("sector_rank_blocks", ["size", "value"])),
        "block_weights": dict(params.get("block_weights", {})),
        "model_weight": float(params.get("model_weight", 0.5)),
        "sector_weight": float(params.get("sector_weight", 0.5)),
        "sector_neutral": bool(params.get("sector_neutral", True)),
        "model_smoothing_span": 1,
        "smoothing_span": int(config.get("smoothing_span", 5)),
    }


def compose_segment(alpha, seg: pd.DataFrame, spec: dict, horizons: list[int]) -> pd.Series:
    """1 区間（OOS の1 fold / Valid 全体）の平滑化済みシグナル。式は predict_signal + walkforward と同じ。"""
    prefix = "blk" if spec["sector_rank_blocks"] else "blkg"
    signal = np.zeros(len(seg), dtype=np.float64)
    for block, weight in spec["block_weights"].items():
        if float(weight) != 0.0:
            z = alpha.xsec_z(seg[f"{prefix}_{block}"].astype(np.float64)).to_numpy(np.float64)
            signal = alpha._add_weighted(signal, float(weight), z)
    parts = []
    for k in horizons:
        pred = seg[f"pred_k{k}"].astype(np.float64)
        parts.append((alpha.xsec_z(pred) if spec["horizon_combine"] == "z" else pred).to_numpy(np.float64))
    stacked = np.column_stack(parts)
    available = ~np.isnan(stacked).all(axis=0)  # horizons trained in this segment (skipped ones are all-NaN)
    if available.any():
        prediction = pd.Series(stacked[:, available].mean(axis=1), index=seg.index)
        model_cfg = {key: spec[key] for key in ("model_weight", "sector_weight", "sector_neutral", "model_smoothing_span")}
        signal = alpha.add_model_terms(signal, prediction, seg["sector33"].to_numpy(np.float64), model_cfg)
    raw = pd.Series(signal, index=seg.index, dtype=np.float32, name="signal")
    return alpha.smooth_by_code(raw, int(spec["smoothing_span"]))


def sector_split(pred: pd.DataFrame, target: pd.DataFrame, sector: pd.Series) -> dict:
    """年別グロス（%/年）を業種間（業種平均リターンへの賭け）と業種内（銘柄選択）に分解する。"""
    weight = ev.compute_weight(pred).iloc[:, 0]
    t = target.iloc[:, 0].reindex(weight.index)
    keys = [weight.index.get_level_values("Date"), sector.reindex(weight.index).fillna(-1.0).to_numpy()]
    t_sector = t.groupby(keys).transform("mean")
    between = (weight * t_sector).groupby(level="Date").sum()
    within = (weight * (t - t_sector)).groupby(level="Date").sum()
    exposure = weight.groupby(keys).sum().abs().groupby(level=0).sum()

    def yearly(series: pd.Series, scale: float) -> dict:
        return {str(y): round(float(g.mean() * scale), 3) for y, g in series.groupby(series.index.year)}

    return {
        "between_sector_gross_pct": yearly(between, 252 * 100),
        "within_sector_gross_pct": yearly(within, 252 * 100),
        "abs_sector_net_exposure": yearly(exposure, 1.0),
    }


def evaluate_spec(alpha, frames: dict, spec: dict, horizons: list[int], targets: dict) -> dict:
    result: dict = {"spec": spec}
    oos = frames.get(("oos", spec["label"]))
    if oos is not None:
        signal = pd.concat([compose_segment(alpha, seg, spec, horizons) for _, seg in oos.groupby("fold", sort=True)])
        signal = signal[~signal.index.duplicated(keep="first")].sort_index()
        result["oos"] = wf.evaluate_signal(signal, targets["train"])
    valid = frames.get(("valid", spec["label"]))
    if valid is not None:
        signal = compose_segment(alpha, valid, spec, horizons)
        # submission.py reindexes to the scoring index (NaN -> neutral via compute_weight's fillna(0)).
        pred = signal.astype(np.float64).reindex(targets["valid"].index).rename("Return").to_frame()
        pred = ev.align_prediction(pred, targets["valid"])
        metrics = sc.daily_metrics(pred, targets["valid"])
        metrics.update(sector_split(pred, targets["valid"], valid["sector33"]))
        result["valid"] = metrics
    return result


# ------------------------------------------------------------------ eval / report
TRAIN_MIN, VALID_V1, TURNOVER_MAX = 2.0, 0.754, 0.017  # targets from the S3 measurements


def load_frames(config: dict, labels: list[str]) -> dict:
    frames = {}
    for label in labels:
        parts = [comp_path(label, str(y)) for y in config["fold_years"]]
        if all(p.exists() for p in parts):
            frames[("oos", label)] = pd.concat([pd.read_parquet(p) for p in parts])
        if comp_path(label, "valid").exists():
            frames[("valid", label)] = pd.read_parquet(comp_path(label, "valid"))
    return frames


def stage_eval(alpha, config: dict, labels: list[str], out: Path) -> dict:
    frames = load_frames(config, labels)
    horizons = [int(k) for k in config["horizons"]]
    targets = {split: sc.load_target(ROOT / "input", split)[0] for split in ("train", "valid")}
    base = base_spec(config)
    results = {}
    for name, override in {**VARIANTS, **SOLO}.items():
        spec = {**base, **override}
        if not any(key[1] == spec["label"] for key in frames):
            results[name] = {"spec": spec, "skipped": f"label={spec['label']} の成分がない"}
            continue
        t0 = time.time()
        results[name] = evaluate_spec(alpha, frames, spec, horizons, targets)
        o, v = results[name].get("oos", {}), results[name].get("valid", {})
        print(f"{name:26s} OOS {o.get('sharpe', float('nan')):+.3f}  Valid {v.get('sharpe', float('nan')):+.3f}  "
              f"TO {v.get('daily_turnover', float('nan')):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(results), encoding="utf-8")
    return results


def _f(value, fmt: str = "+.3f") -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, fmt)


def render_markdown(results: dict) -> str:
    ok = {name: r for name, r in results.items() if "skipped" not in r}
    lines = ["# sweep_improve1 結果（実測。tools/sweep_improve1.py が自動生成）", "",
             f"判定: T = Train OOS >= {TRAIN_MIN} / V = Valid > {VALID_V1}（v1）/ TO = Valid 回転率 <= {TURNOVER_MAX}", "",
             "| variant | label | OOS SR | OOS 回転率 | Valid SR | グロス% | コスト% | 回転率 | RankIC | Q1〜Q5 (bp/日) | T | V | TO |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in ok.items():
        o, v = r.get("oos", {}), r.get("valid", {})
        q = v.get("quantile_mean_target", {})
        lines.append("| " + " | ".join([
            name, r["spec"]["label"], _f(o.get("sharpe")), _f(o.get("turnover"), ".4f"), _f(v.get("sharpe")),
            _f(100 * v["gross_annual"], "+.2f") if "gross_annual" in v else "—",
            _f(100 * v["cost_annual"], ".2f") if "cost_annual" in v else "—",
            _f(v.get("daily_turnover"), ".4f"), _f(v.get("rank_ic"), "+.4f"),
            " ".join(_f(q.get(f"Q{i}"), "+.2f") for i in range(1, 6)),
            "OK" if o.get("sharpe", -9) >= TRAIN_MIN else "NG",
            "OK" if v.get("sharpe", -9) > VALID_V1 else "NG",
            "OK" if v.get("daily_turnover", 9) <= TURNOVER_MAX else "NG",
        ]) + " |")
    lines += ["", "## ③ horizon_combine × label_window の 2×2 分解（Sharpe 差）", ""]
    cells = {("strict", "z"): "I1", ("strict", "mean"): "hc_mean", ("partial", "z"): "partial", ("partial", "mean"): "partial_hc_mean"}
    for split, key in (("oos", "sharpe"), ("valid", "sharpe"), ("valid", "daily_turnover")):
        try:
            s = {c: ok[n][split][key] for c, n in cells.items()}
        except KeyError:
            lines.append(f"- {split}.{key}: 4 セルがそろわない（--labels strict,partial で再実行）")
            continue
        z_eff = (s["strict", "z"] - s["strict", "mean"] + s["partial", "z"] - s["partial", "mean"]) / 2
        strict_eff = (s["strict", "z"] - s["partial", "z"] + s["strict", "mean"] - s["partial", "mean"]) / 2
        inter = (s["strict", "z"] - s["strict", "mean"]) - (s["partial", "z"] - s["partial", "mean"])
        lines.append(f"- {split}.{key}: z の寄与 {z_eff:+.4f} / strict の寄与 {strict_eff:+.4f} / 交互作用 {inter:+.4f}")
    for split, title in (("valid", "Valid"), ("oos", "Train OOS")):
        years = sorted({y for r in ok.values() for y in r.get(split, {}).get("yearly_sharpe", {})})
        lines += ["", f"## {title} 年別 Sharpe", "", "| variant | " + " | ".join(years) + " |", "|---|" + "---|" * len(years)]
        for name, r in ok.items():
            yearly = r.get(split, {}).get("yearly_sharpe", {})
            if yearly:
                lines.append(f"| {name} | " + " | ".join(_f(yearly.get(y), "+.2f") for y in years) + " |")
    lines += ["", "## ② Valid 年別グロスの業種間／業種内分解（%/年）", ""]
    for name in ("I1", "K1_msmooth10", "v1_like", "no_sector_rank_blocks"):
        v = ok.get(name, {}).get("valid", {})
        if "between_sector_gross_pct" in v:
            years = sorted(v["between_sector_gross_pct"])
            lines += [f"### {name}", "", "| | " + " | ".join(years) + " |", "|---|" + "---|" * len(years),
                      "| 業種間 | " + " | ".join(_f(v["between_sector_gross_pct"][y], "+.2f") for y in years) + " |",
                      "| 業種内 | " + " | ".join(_f(v["within_sector_gross_pct"][y], "+.2f") for y in years) + " |", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    strategy_dir = (ROOT / args.strategy).resolve()
    alpha = load_alpha(strategy_dir, args.module)
    config = wf.load_config(strategy_dir, None)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    per_label = len(config["fold_years"]) + 1
    prog = Progress(ROOT / "work" / "progress" / "sweep_improve1.json", total=per_label * len(labels) + 1,
                    meta={"strategy": strategy_dir.name, "module": args.module, "labels": labels})
    step = 0
    for label in labels:
        if args.stage in ("all", "oos"):
            step = stage_oos(alpha, config, label, prog, step)
        if args.stage in ("all", "valid"):
            step = stage_valid(alpha, config, label, prog, step)
    if args.stage in ("all", "eval"):
        stage_eval(alpha, config, labels, ROOT / args.out)
        prog.done("sweep 完了", report=str(ROOT / args.out) + ".md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
