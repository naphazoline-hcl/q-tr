"""sweep_improve2.py — improve2 の候補（ensemble 重み x モデル項 span x 予測側の重み x 局面）を一括評価する。

学習は fold ごとに 1 回だけ（LGBM + Ridge、--rank 指定時は rank モデルも）。予測側の候補は保存した
「成分」から再合成するので、1 回の学習で多数の候補を Train OOS と Valid の両方で比べられる。

Stage oos  : tools/walkforward.py と同じ fold / purge / train_end で fit_model し、fold 年の成分を保存。
Stage valid: train_v2.py と同じく Train 全期間（end=None）で学習し、2014-06-01 以降の成分を保存。
Stage eval : VARIANTS を再合成 -> 銘柄ごと EWMA（smoothing_span）-> walkforward.evaluate_signal（採点と同じ式。
             Valid は採点対象 index に揃えて NaN=0 として評価）。JSON / Markdown を出力。
成分: blk_<block>（同日 z）/ rw_<block>（regime_v4 の行別ブロック重み）/ p_lgbm / p_ridge / p_rank /
      sector33 / fold。

使い方（リポジトリ直下で）:
  python tools/sweep_improve2.py --strategy strategies/v0_multifactor --module alpha_v2 [--rank]
  python tools/sweep_improve2.py --stage eval          # 成分の保存後、評価だけやり直す
中断しても work/sweep_improve2/ に保存済みの成分は読み飛ばす（進捗: work/progress/sweep_improve2.json）。
target_1day_valid.parquet を読むのはこのローカル評価ツールだけ（提出フォルダのコードは読まない）。
採用は T（Train OOS >= 2.0）かつ V（Valid > K1 +0.7591）かつ TO（Valid 回転率 <= 0.017）。Valid だけ
良い候補は採らない（v0/計画.md §3）。
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

import score as sc  # noqa: E402
import walkforward as wf  # noqa: E402
from progress import Progress  # noqa: E402

WORK = ROOT / "work" / "sweep_improve2"
VALID_HISTORY_START = pd.Timestamp("2014-06-01")  # same as submission.HISTORY_START
TRAIN_MIN, VALID_K1, TURNOVER_MAX = 2.0, 0.7591, 0.017  # S4 measurements / P5 targets


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="improve2 sweep (ensemble / span / regime)")
    parser.add_argument("--strategy", default="strategies/v0_multifactor")
    parser.add_argument("--module", default="alpha_v2")
    parser.add_argument("--stage", default="all", choices=["all", "oos", "valid", "eval"])
    parser.add_argument("--rank", action="store_true", help="rank モデルも学習する（CPU 時間が増える）")
    parser.add_argument("--out", default=str(ROOT / "work" / "reports" / "sweep_improve2"))
    return parser.parse_args(argv)


def load_alpha(strategy_dir: Path, module_name: str):
    sys.path.insert(0, str(strategy_dir))
    return importlib.import_module(module_name)


def fit_params(config: dict, rank: bool) -> dict:
    """成分を作るための学習設定: Ridge（と rank）を必ず学習し、auto span はここでは使わない。"""
    weights = {"lgbm": 1.0, "ridge": 1.0, "rank": 1.0 if rank else 0.0}
    return {**config.get("params", {}), "ensemble_weights": weights, "turnover_control": "off"}


def components(alpha, features: pd.DataFrame, model: dict, config: dict) -> pd.DataFrame:
    """成分 DataFrame（index = features.index）。各モデル予測は合成前の値（float64）で残す。"""
    out = pd.DataFrame(index=features.index)
    blocks_z = {b: alpha.xsec_z(s).to_numpy(np.float64) for b, s in alpha.block_scores(features, config).items()}
    for block, z in blocks_z.items():
        out[f"blk_{block}"] = z
    thresholds = model.get("regime_v4") or {"vol_threshold": float("nan")}
    weights = alpha.regime_v4.block_weight_arrays(features, thresholds, config["regime_v4"],
                                                  config.get("regime_weights") or {}, config["block_weights"],
                                                  list(blocks_z))
    for block, w in weights.items():
        out[f"rw_{block}"] = np.asarray(w, dtype=np.float64)
    columns = model.get("columns")
    ranked = alpha.rank_for_model(features, columns) if columns else None
    lgbm = alpha.model_prediction(features, model, config["horizon_combine"], ranked=ranked)
    ridge = alpha.ridge_prediction(features, model.get("ridge"), blocks_z, config)
    rank = None
    if model.get("rank") and ranked is not None:
        raw = {n: m.predict(ranked[alpha._feature_names(m)]) for n, m in model["rank"].items()}
        rank = alpha.ensemble.combine_rank(raw, features.index)
    for key, pred in (("lgbm", lgbm), ("ridge", ridge), ("rank", rank)):
        out[f"p_{key}"] = pred.to_numpy(np.float64) if pred is not None else np.nan
    out["sector33"] = features["sector33"].to_numpy(np.float64)
    return out


def stage_oos(alpha, config: dict, rank: bool, prog: Progress, step: int) -> int:
    """walkforward.run と同じ fold 分割で学習し、fold 年の成分を fold ごとに保存（再開可）。"""
    params = alpha.merged_params(fit_params(config, rank))
    horizons = [int(k) for k in config["horizons"]]
    history_start, purge_days = pd.Timestamp(config["history_start"]), int(config["purge_days"])
    if all((WORK / f"oos_{y}.parquet").exists() for y in config["fold_years"]):
        return step + len(config["fold_years"])
    features = alpha.build_features(splits=("train",), start=history_start)
    dates = pd.DatetimeIndex(features.index.get_level_values("Date").unique()).sort_values()
    date_values = pd.Series(features.index.get_level_values("Date")).to_numpy()
    for year in config["fold_years"]:
        step += 1
        path = WORK / f"oos_{year}.parquet"
        if path.exists():
            prog.update(step, f"oos {year} skip（保存済み）")
            continue
        t0 = time.time()
        fold_start, fold_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        train_dates = dates[dates < fold_start]
        train_end = train_dates[-purge_days] if len(train_dates) > purge_days else train_dates[0]
        features_train = features[date_values <= np.datetime64(train_end)]
        fold_mask = (date_values >= np.datetime64(fold_start)) & (date_values <= np.datetime64(fold_end))
        labels = pd.DataFrame({f"k{k}": alpha.make_label(k, start=history_start, end=train_end) for k in horizons})
        labels = labels.loc[labels.index.intersection(features_train.index)].dropna(how="all")
        model = alpha.fit_model(features_train.loc[features_train.index.intersection(labels.index)], labels, params)
        frame = components(alpha, features[fold_mask], model, params)
        frame["fold"] = int(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        prog.update(step, f"oos {year} 完了 lgbm={len(model['models'])} rank={len(model.get('rank') or {})} "
                    f"({time.time() - t0:.0f}s)", artifact=str(path), seconds=round(time.time() - t0, 1))
    return step


def stage_valid(alpha, config: dict, rank: bool, prog: Progress, step: int) -> int:
    """Train 全期間で学習（train_v2 と同じ）し、2014-06-01 以降の成分を保存（再開可）。"""
    step += 1
    path = WORK / "valid.parquet"
    if path.exists():
        prog.update(step, "valid skip（保存済み）")
        return step
    t0 = time.time()
    params = alpha.merged_params(fit_params(config, rank))
    horizons = [int(k) for k in config["horizons"]]
    history_start = pd.Timestamp(config["history_start"])
    features = alpha.build_features(splits=("train",), start=history_start)
    labels = pd.DataFrame({f"k{k}": alpha.make_label(k, start=history_start, end=None) for k in horizons})
    labels = labels.loc[labels.index.intersection(features.index)].dropna(how="all")
    model = alpha.fit_model(features.loc[features.index.intersection(labels.index)], labels, params)
    del features
    features = alpha.build_features(splits=("train", "valid"), start=VALID_HISTORY_START)
    frame = components(alpha, features, model, params)
    frame["fold"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    prog.update(step, f"valid 完了 ({time.time() - t0:.0f}s)", artifact=str(path))
    return step


# ------------------------------------------------------------------ variants
# Overrides relative to K1 (current params with ensemble lgbm only, regime off, model span 10).
R03 = {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.3, "rank": 0.0}}
VARIANTS: dict[str, dict] = {
    "K1": {},
    "E1_ridge03": R03,  # shipped walkforward_config.json
    "ridge06": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0}},
    "ridge10": {"ensemble_weights": {"lgbm": 1.0, "ridge": 1.0, "rank": 0.0}},
    "ridge_only": {"ensemble_weights": {"lgbm": 0.0, "ridge": 1.0, "rank": 0.0}},
    "rank03": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.3}},
    "ridge03_rank03": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.3, "rank": 0.3}},
    # C: model-term EWMA span 5 / 10 / 20 (labels fixed; slow_profile label sets need retraining)
    "span5": {"model_smoothing_span": 5},
    "span20": {"model_smoothing_span": 20},
    "ridge03_span20": {**R03, "model_smoothing_span": 20},
    # S4: strong but too fast (turnover 0.0198-0.0207) -> slowed down
    "neutral_only_span20": {"model_weight": 0.0, "model_smoothing_span": 20},
    "mw025_span20": {"model_weight": 0.25, "model_smoothing_span": 20},
    "neutral_only_ridge03_span20": {**R03, "model_weight": 0.0, "model_smoothing_span": 20},
    "size_heavy": {"block_weights": {"size": 1.0, "value": 0.3, "quality": 0.3, "lowrisk": 0.2}},
    "size_heavy_ridge03_span20": {**R03, "model_smoothing_span": 20,
                                  "block_weights": {"size": 1.0, "value": 0.3, "quality": 0.3, "lowrisk": 0.2}},
    "final_span8_ridge03": {**R03, "smoothing_span": 8},
    # B (default OFF): 3-regime block mixing with the configured regime_weights
    "regime_v4": {"regime_mix": True},
    "regime_v4_ridge03": {**R03, "regime_mix": True},
    # P5 local follow-up: ridge strength x model weight x span (from the saved components)
    "ridge08": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.8, "rank": 0.0}},
    "ridge10_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 1.0, "rank": 0.0}, "model_smoothing_span": 20},
    "ridge10_span15": {"ensemble_weights": {"lgbm": 1.0, "ridge": 1.0, "rank": 0.0}, "model_smoothing_span": 15},
    "mw025_ridge06_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 20},
    "mw025_ridge10_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 1.0, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 20},
    "neutral_only_ridge10_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 1.0, "rank": 0.0},
                                    "model_weight": 0.0, "model_smoothing_span": 20},
    # P5 local follow-up 2: around mw025_ridge06_span20
    "mw025_ridge05_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.5, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 20},
    "mw025_ridge07_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.7, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 20},
    "mw02_ridge06_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                            "model_weight": 0.2, "model_smoothing_span": 20},
    "mw03_ridge06_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                            "model_weight": 0.3, "model_smoothing_span": 20},
    "mw025_ridge06_span25": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 25},
    "mw025_ridge06_span30": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 30},
    "mw025_ridge06_span15": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 15},
    # P5 local follow-up 3: span refine around the adopted mw025_ridge06 (base spec is pinned to K1 now)
    "mw025_ridge04_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.4, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 20},
    "mw025_ridge06_span10": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 10},
    "mw025_ridge06_span12": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 12},
    "mw025_ridge06_span35": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.0},
                             "model_weight": 0.25, "model_smoothing_span": 35},
    "neutral_only_span15": {"model_weight": 0.0, "model_smoothing_span": 15},
    # P5 local follow-up 4: rank on top of the adopted combo (rank preds are in the components)
    "rank06": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.6}},
    "rank10": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 1.0}},
    "mw025_rank03_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.3},
                            "model_weight": 0.25, "model_smoothing_span": 20},
    "mw025_ridge06_rank03_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.3},
                                    "model_weight": 0.25, "model_smoothing_span": 20},
    "mw025_ridge06_rank06_span20": {"ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.6},
                                    "model_weight": 0.25, "model_smoothing_span": 20},
}


def base_spec(config: dict) -> dict:
    """候補の基準 = K1（S4）。walkforward_config.json の現在値（採用版の mw/span 等）に依存しないよう固定する。"""
    params = {**config.get("params", {})}
    params.update({"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.0}, "regime_mix": False,
                   "model_weight": 0.5, "model_smoothing_span": 10,
                   "block_weights": {"size": 1.0, "value": 0.5, "quality": 0.5, "lowrisk": 0.3}})
    params["smoothing_span"] = int(config.get("smoothing_span", 5))
    return params


def compose(alpha, seg: pd.DataFrame, spec: dict) -> pd.Series:
    """alpha_v2.predict_signal + walkforward の最終 EWMA を成分から再現する。"""
    blocks = [c[4:] for c in seg.columns if c.startswith("blk_")]
    base = np.zeros(len(seg), dtype=np.float64)
    for block in blocks:
        z = seg[f"blk_{block}"].to_numpy(np.float64)
        if spec.get("fill_missing_blocks"):
            z = np.nan_to_num(z, nan=0.0)
        weight = seg[f"rw_{block}"].to_numpy(np.float64) if spec.get("regime_mix") else \
            float(spec["block_weights"].get(block, 0.0))
        base = alpha._add_weighted(base, weight, z)
    predictions = {k: (seg[f"p_{k}"] if seg[f"p_{k}"].notna().any() else None) for k in ("lgbm", "ridge", "rank")}
    prediction = alpha.ensemble.combine(predictions, alpha.ensemble.ensemble_weights(spec))
    raw = alpha.add_model_terms(base, prediction, seg["sector33"].to_numpy(np.float64), spec)
    signal = pd.Series(raw, index=seg.index, dtype=np.float32)
    return alpha.slowdown.ewm_by_code(signal, int(spec["smoothing_span"]))


def evaluate(alpha, frames: dict, spec: dict, targets: dict) -> dict:
    result: dict = {}
    if "oos" in frames:
        parts = [compose(alpha, seg, spec) for _, seg in frames["oos"].groupby("fold", sort=True)]
        signal = pd.concat(parts)
        result["oos"] = wf.evaluate_signal(signal[~signal.index.duplicated(keep="first")].sort_index(), targets["train"])
    if "valid" in frames:
        signal = compose(alpha, frames["valid"], spec).astype(np.float64)
        # submission.py reindexes to the scoring index; the scorer treats NaN as 0 (fillna(0)).
        result["valid"] = wf.evaluate_signal(signal.reindex(targets["valid"].index).fillna(0.0), targets["valid"])
    return result


def render_markdown(results: dict) -> str:
    def f(value, fmt="+.3f"):
        return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, fmt)
    lines = ["# sweep_improve2 結果（実測。tools/sweep_improve2.py が自動生成）", "",
             f"判定: T = Train OOS >= {TRAIN_MIN} / V = Valid > {VALID_K1}（K1）/ TO = Valid 回転率 <= {TURNOVER_MAX}", "",
             "| variant | OOS SR | OOS 回転率 | Valid SR | グロス% | コスト% | 回転率 | 2020 | 2026 | T | V | TO |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o, v = r.get("oos", {}), r.get("valid", {})
        yearly = v.get("yearly_sharpe", {})
        flags = [o.get("sharpe", -9) >= TRAIN_MIN, v.get("sharpe", -9) > VALID_K1, v.get("turnover", 9) <= TURNOVER_MAX]
        lines.append("| " + " | ".join([
            name, f(o.get("sharpe")), f(o.get("turnover"), ".4f"), f(v.get("sharpe")),
            f(100 * v["gross_annual"], "+.2f") if "gross_annual" in v else "—",
            f(100 * v["cost_annual"], ".2f") if "cost_annual" in v else "—", f(v.get("turnover"), ".4f"),
            f(yearly.get("2020"), "+.2f"), f(yearly.get("2026"), "+.2f"),
        ] + ["OK" if flag else "NG" for flag in flags]) + " |")
    return "\n".join(lines) + "\n"


def stage_eval(alpha, config: dict, out: Path) -> dict:
    frames = {}
    oos_parts = [WORK / f"oos_{y}.parquet" for y in config["fold_years"]]
    if all(p.exists() for p in oos_parts):
        frames["oos"] = pd.concat([pd.read_parquet(p) for p in oos_parts])
    if (WORK / "valid.parquet").exists():
        frames["valid"] = pd.read_parquet(WORK / "valid.parquet")
    targets = {"train": pd.read_parquet(ROOT / "input" / "target_1day_train.parquet"),
               "valid": sc.load_target(ROOT / "input", "valid")[0]}
    base = base_spec(config)
    results = {}
    for name, override in VARIANTS.items():
        t0 = time.time()
        spec = alpha._deep_merge(base, override)
        results[name] = evaluate(alpha, frames, spec, targets)
        o, v = results[name].get("oos", {}), results[name].get("valid", {})
        print(f"{name:30s} OOS {o.get('sharpe', float('nan')):+.3f}  Valid {v.get('sharpe', float('nan')):+.3f}  "
              f"TO {v.get('turnover', float('nan')):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(results), encoding="utf-8")
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    strategy_dir = (ROOT / args.strategy).resolve() if not Path(args.strategy).is_absolute() else Path(args.strategy)
    alpha = load_alpha(strategy_dir, args.module)
    config = wf.load_config(strategy_dir, None)
    prog = Progress(ROOT / "work" / "progress" / "sweep_improve2.json", total=len(config["fold_years"]) + 2,
                    meta={"strategy": strategy_dir.name, "rank": args.rank, "variants": list(VARIANTS)})
    step = 0
    if args.stage in ("all", "oos"):
        step = stage_oos(alpha, config, args.rank, prog, step)
    if args.stage in ("all", "valid"):
        step = stage_valid(alpha, config, args.rank, prog, step)
    if args.stage in ("all", "eval"):
        stage_eval(alpha, config, Path(args.out))
    prog.done("sweep_improve2 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
