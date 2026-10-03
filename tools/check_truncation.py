"""特徴量の因果性を機械的に検証する（未来のデータを足しても過去の値が変わらない）。

考え方: 「未来を知った状態で作った特徴量」と「その時点までのデータだけで作った特徴量」で、
過去の行の値が完全に一致すれば、その特徴量は未来を使っていない。

手順:
1. `alpha.build_features(splits=("train",), start=...)` を **全期間** で構築（F_all）
2. 同じものを `end=<cut>` で構築（F_cut）
3. `Date <= cut` の共通行で全列を比較し、差があればその列と最大差を報告

```bash
python tools/check_truncation.py --strategy strategies/v0_multifactor --cut 2014-06-30
```

`make_label` についても、`end=<cut>` のとき窓が cut を超える行が NaN になっているかを確認する。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="特徴量の因果性（truncation invariance）検証")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--module", default="alpha")
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--cut", default="2014-06-30", help="この日までを過去とみなす")
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    strategy_dir = Path(args.strategy).resolve()
    sys.path.insert(0, str(strategy_dir))
    alpha = importlib.import_module(args.module)
    cut = pd.Timestamp(args.cut)
    start = pd.Timestamp(args.start)
    splits = tuple(args.splits)

    print(f"build_features (full) : splits={splits} start={start.date()}")
    full = alpha.build_features(splits=splits, start=start)
    print(f"build_features (cut)  : end={cut.date()}")
    truncated = alpha.build_features(splits=splits, start=start, end=cut)

    if truncated.index.get_level_values("Date").max() > cut:
        print("ERROR: end を指定しても cut より後の行が返っている（build_features の end 実装を確認）")
        return 1

    common = truncated.index.intersection(full.index)
    left = truncated.loc[common]
    right = full.loc[common]
    print(f"compare: {len(common):,} rows × {left.shape[1]} cols")

    failures = 0
    for column in left.columns:
        a = left[column].to_numpy(dtype="float64")
        b = right[column].to_numpy(dtype="float64")
        diff = np.abs(a - b)
        both_nan = np.isnan(a) & np.isnan(b)
        diff[both_nan] = 0.0
        nan_mismatch = int((np.isnan(a) ^ np.isnan(b)).sum())
        worst = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
        if nan_mismatch or worst > args.tol:
            failures += 1
            print(f"  [NG] {column}: max|diff|={worst:.3e} NaN不一致={nan_mismatch}")
    if failures:
        print(f"RESULT: NG ({failures} columns are not causal)")
        return 1
    print("RESULT: OK（全列で truncation invariance を確認）")

    # ラベルの purge 確認
    for k in (126, 250):
        label = alpha.make_label(k, start=start, end=cut)
        dates = pd.DatetimeIndex(label.index.get_level_values("Date"))
        if (dates > cut).any():
            print(f"  [NG] make_label(k={k}) が end より後の行を返している")
            failures += 1
        tail = label.loc[dates == dates.max()].notna().mean()
        print(f"  make_label(k={k}): rows={label.notna().sum():,} 末尾日NaN率={1 - tail:.2%}（末尾は窓不足のためNaNが正常）")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
