"""alpha_features.build_features の自己検査。

使い方（配布 parquet を置いたディレクトリで実行するか、--data-dir で指定する）::

    python selftest_features.py --data-dir input --start 2010-01-01
    python selftest_features.py --data-dir input --cut 2014-06-30

表示する項目:
  1. shape・列数・期間・ユニーク銘柄数（＋ v1 列の保持・列数レンジ・FEATURE_GROUPS の整合）
  2. 列別 NaN 率の上位20
  3. 主要列の記述統計（mean / std / min / max / 1% / 99%）
  4. 壊れた特徴量の検出（全欠損・全同一値・inf を含む列）
  5. 簡易因果性検査（end=cut で再構築し、Date <= cut の値が全列で一致するか）
  6. 実行時間（秒）
終了コード: 4 または 5 で問題があれば 1、それ以外は 0。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_features as af  # noqa: E402

KEY_V1 = ["r1", "rcc20", "mom", "vol60", "logturn60", "logsize", "bp", "ep_f", "roe", "beta"]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def report_shape(df: pd.DataFrame) -> bool:
    section("1. shape / 列数 / 期間 / 銘柄数")
    dates = df.index.get_level_values("Date")
    print(f"shape: {df.shape}  列数: {df.shape[1]}")
    print(f"期間: {dates.min().date()} .. {dates.max().date()}  営業日数: {dates.nunique()}")
    print(f"ユニーク銘柄数: {df.index.get_level_values('Code').nunique()}")
    print(f"dtype: {sorted(set(map(str, df.dtypes)))}")
    v1_ok = list(df.columns[: len(af.V1_COLUMNS)]) == af.V1_COLUMNS
    range_ok = 75 <= df.shape[1] <= 95
    groups_ok = sorted(af.NEW_COLUMNS) == sorted(df.columns[len(af.V1_COLUMNS):])
    print(f"v1 の51列を同名・同順で保持: {'OK' if v1_ok else 'NG'}")
    print(f"列数が 75〜95: {'OK' if range_ok else 'NG'}（v1 {len(af.V1_COLUMNS)} + 新規 {len(af.NEW_COLUMNS)}）")
    print(f"FEATURE_GROUPS と新規列が一致: {'OK' if groups_ok else 'NG'}")
    for group, names in af.FEATURE_GROUPS.items():
        print(f"  {group:15s} {len(names):2d}列: {', '.join(names)}")
    return v1_ok and range_ok and groups_ok


def report_nan(df: pd.DataFrame) -> None:
    section("2. 列別 NaN 率（上位20）")
    rate = df.isna().mean().sort_values(ascending=False).head(20)
    for name, value in rate.items():
        print(f"  {name:20s} {value:7.2%}")


def report_stats(df: pd.DataFrame) -> None:
    section("3. 主要列の記述統計")
    columns = [c for c in KEY_V1 if c in df.columns] + af.NEW_COLUMNS
    rows = []
    for name in columns:
        values = df[name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append((name, *([np.nan] * 6)))
            continue
        q01, q99 = np.quantile(values, [0.01, 0.99])
        rows.append((name, values.mean(), values.std(), values.min(), values.max(), q01, q99))
    table = pd.DataFrame(rows, columns=["column", "mean", "std", "min", "max", "p01", "p99"]).set_index("column")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(table.to_string(float_format=lambda v: f"{v: .4g}"))


def detect_broken(df: pd.DataFrame) -> list[str]:
    section("4. 壊れた特徴量の検出")
    problems = []
    for name in df.columns:
        values = df[name].to_numpy()
        finite = values[np.isfinite(values)]
        if np.isinf(values).any():
            problems.append(f"{name}: inf を含む（{int(np.isinf(values).sum())} 行）")
        if np.isnan(values).all():
            problems.append(f"{name}: 全欠損")
        elif finite.size and finite.min() == finite.max():
            problems.append(f"{name}: 全て同一値（{finite.min()}）")
    for line in problems:
        print(f"  NG {line}")
    print("  問題なし" if not problems else f"  計 {len(problems)} 件")
    return problems


def check_causality(full: pd.DataFrame, splits, start, cut: pd.Timestamp) -> list[str]:
    section(f"5. 簡易因果性検査（end={cut.date()} で再構築して Date <= end を比較）")
    t0 = time.perf_counter()
    part = af.build_features(splits=splits, start=start, end=cut)
    print(f"  再構築: shape={part.shape}  {time.perf_counter() - t0:.1f} 秒")
    problems = []
    max_date = part.index.get_level_values("Date").max()
    if max_date > cut:
        problems.append(f"end 引数: Date > end の行がある（最大 {max_date.date()}）")
    head = full.loc[full.index.get_level_values("Date") <= cut]
    if not head.index.equals(part.index):
        problems.append(f"index 不一致: full={len(head)} 行 / end 指定={len(part)} 行")
        part = part.reindex(head.index)
    for name in full.columns:
        a = head[name].to_numpy(dtype=np.float64)
        b = part[name].to_numpy(dtype=np.float64)
        nan_diff = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        max_diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        if nan_diff or max_diff > 0.0:
            problems.append(f"{name}: 最大差 {max_diff:.3g} / NaN 位置の不一致 {nan_diff} 行")
    for line in problems:
        print(f"  NG {line}")
    print("  全列一致（先読みの兆候なし）" if not problems else f"  計 {len(problems)} 件")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="alpha_features の自己検査")
    parser.add_argument("--data-dir", default=".", help="配布 parquet のあるディレクトリ")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--start", default=None, help="build_features の start（例 2010-01-01）")
    parser.add_argument("--cut", default="2014-06-30", help="因果性検査の end")
    args = parser.parse_args()

    os.chdir(args.data_dir)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    start = pd.Timestamp(args.start) if args.start else None
    t0 = time.perf_counter()
    full = af.build_features(splits=splits, start=start)
    build_sec = time.perf_counter() - t0

    report_shape(full)
    report_nan(full)
    report_stats(full)
    broken = detect_broken(full)
    leaks = check_causality(full, splits, start, pd.Timestamp(args.cut))

    section("6. 実行時間")
    print(f"  build_features（全期間）: {build_sec:.1f} 秒")
    print(f"  合計: {time.perf_counter() - t0:.1f} 秒")
    status = 1 if (broken or leaks) else 0
    print(f"\nRESULT: {'NG' if status else 'OK'}（壊れた列 {len(broken)} 件 / 因果性の不一致 {len(leaks)} 件）")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
