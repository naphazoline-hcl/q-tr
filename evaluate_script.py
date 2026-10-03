#!/usr/bin/env python3
"""Local evaluator for the stock competition submission format.

The official scorer imports ``submission.predict()`` and expects it to return a
one-column DataFrame indexed by (Date, Code).  This script mirrors that flow for
local smoke tests using files under ``input/``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TRANSACTION_COST_RATE = 0.1 * 0.01  # score_omni_patched.py と同一の片道コスト
TARGET_CANDIDATES = (
    "target_1day_valid.parquet",
    "target_1day_train.parquet",
    "raw_target_1day_valid.parquet",
    "raw_target_1day_train.parquet",
)


def compute_weight(signal: pd.DataFrame) -> pd.DataFrame:
    quantile = (
        signal.sort_index()
        .fillna(0)
        .groupby("Date")
        .rank(method="first")
        .groupby("Date")
        .transform(lambda x: pd.qcut(x, 5, labels=False))
    )
    return (quantile - 2) / quantile.groupby("Date").count() / 1.2


def compute_pl(signal: pd.DataFrame, target_1day: pd.DataFrame) -> pd.Series:
    weight = compute_weight(signal).iloc[:, 0]
    return (
        weight * target_1day.iloc[:, 0]
        - TRANSACTION_COST_RATE * weight.groupby("Code").diff().abs().fillna(weight)
    )


def compute_sr(pl: pd.Series) -> float:
    daily_return = pl.groupby("Date").sum()
    return float(daily_return.mean() / daily_return.std() * np.sqrt(252))


def find_target_file(data_dir: Path, explicit_target: str | None) -> Path:
    if explicit_target:
        path = Path(explicit_target)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")
        return path

    for filename in TARGET_CANDIDATES:
        path = data_dir / filename
        if path.exists():
            return path

    names = ", ".join(TARGET_CANDIDATES)
    raise FileNotFoundError(f"No target file found in {data_dir}. Tried: {names}")


def locate_submission_root(search_dir: Path) -> Path:
    """Find the directory that actually contains submission.py.

    Tolerates folder-wrapped zips (submission/submission.py at the zip root)
    and Finder's __MACOSX/ junk by picking the shallowest submission.py.
    Falls back to search_dir itself so the caller's existence check fires.
    """
    candidates = []
    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]
        if "submission.py" in files:
            candidates.append(Path(root))
    if not candidates:
        return search_dir
    return min(candidates, key=lambda p: len(p.parts))


def prepare_submission_import(submission_path: Path) -> tuple[Path | None, Path]:
    """Put a submission on sys.path and return (temp_dir, import_root)."""
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission path not found: {submission_path}")

    if submission_path.is_dir():
        import_root = locate_submission_root(submission_path)
        sys.path.insert(0, str(import_root))
        return None, import_root

    if zipfile.is_zipfile(submission_path):
        temp_dir = Path(tempfile.mkdtemp(prefix="stock_comp_submission_"))
        with zipfile.ZipFile(submission_path) as zf:
            zf.extractall(temp_dir)
        import_root = locate_submission_root(temp_dir)
        sys.path.insert(0, str(import_root))
        return temp_dir, import_root

    if submission_path.suffix == ".py":
        sys.path.insert(0, str(submission_path.parent))
        return None, submission_path.parent

    raise ValueError("Submission must be a directory, a .zip file, or a .py file.")


def load_prediction(submission_path: Path, data_dir: Path) -> pd.DataFrame:
    temp_dir: Path | None = None
    prev_cwd = Path.cwd()
    try:
        temp_dir, import_root = prepare_submission_import(submission_path)
        if not (import_root / "submission.py").exists():
            raise FileNotFoundError(f"submission.py not found under {import_root}")

        sys.modules.pop("submission", None)
        submission = importlib.import_module("submission")

        if not hasattr(submission, "predict"):
            raise AttributeError("submission.py must define predict().")

        # predict() は配布 parquet をベース名で相対読みするため、CWD を data_dir に移して実行する
        # （本番採点では cwd 直下に parquet が置かれる前提。run_strategy.py と同じ扱い）。
        os.chdir(data_dir)
        pred = submission.predict()
    finally:
        os.chdir(prev_cwd)
        if temp_dir is not None:
            # Keep sys.path entry harmless for this short-lived process; remove files.
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"predict() must return pandas.DataFrame, got {type(pred).__name__}.")
    if pred.shape[1] != 1:
        raise ValueError(f"predict() must return exactly one column, got {pred.shape[1]}.")
    if list(pred.index.names) != ["Date", "Code"]:
        raise ValueError(f"prediction index names must be ['Date', 'Code'], got {pred.index.names}.")

    return pred


def align_prediction(pred: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    missing = target.index.difference(pred.index)
    if len(missing) > 0:
        sample = list(missing[:5])
        raise ValueError(f"Prediction is missing {len(missing)} target rows. Sample: {sample}")

    aligned = pred.loc[target.index]
    if not aligned.index.equals(target.index):
        raise AssertionError("The indices of prediction and target do not match.")
    return aligned


def resolve_submission(script_dir: Path, arg: str | None) -> Path:
    """--submission 未指定時の探索。旧 ./submission → ./strategies 内の唯一の戦略の順。"""
    if arg:
        return Path(arg).resolve()
    legacy = script_dir / "submission"
    if legacy.exists():
        return legacy
    strategies = script_dir / "strategies"
    if strategies.is_dir():
        cands = sorted(
            d for d in strategies.iterdir()
            if d.is_dir() and (d / "submission.py").exists()
        )
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            names = ", ".join(d.name for d in cands)
            raise SystemExit(
                f"strategies/ に複数の戦略があります: {names}\n"
                "採点する戦略を --submission strategies/<name> で指定してください。"
            )
    raise SystemExit(
        "提出が見つかりません。strategies/<name>/submission.py を作るか、"
        "--submission でディレクトリ / zip / .py を指定してください。"
    )


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run a local stock_comp scoring smoke test.")
    parser.add_argument(
        "--data-dir",
        default=str(script_dir / "input"),
        help="Directory containing distributed parquet files. Default: ./input",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help="Submission directory, zip, or submission.py. "
             "Default: auto-detect (./submission or the only folder in ./strategies).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target parquet filename/path. Default: auto-detect target/raw_target train/valid.",
    )
    return parser


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    parser = build_parser()
    defaults = []
    if os.environ.get("DATA_DIR"):
        defaults.extend(["--data-dir", os.environ["DATA_DIR"]])
    if os.environ.get("USERSUBMISSION"):
        defaults.extend(["--submission", os.environ["USERSUBMISSION"]])
    if os.environ.get("TARGET_FILE"):
        defaults.extend(["--target", os.environ["TARGET_FILE"]])
    args = parser.parse_args(defaults + sys.argv[1:])

    data_dir = Path(args.data_dir).resolve()
    submission_path = resolve_submission(script_dir, args.submission)
    target_path = find_target_file(data_dir, args.target)

    target = pd.read_parquet(target_path)
    if list(target.index.names) != ["Date", "Code"]:
        raise ValueError(f"target index names must be ['Date', 'Code'], got {target.index.names}.")

    pred = align_prediction(load_prediction(submission_path, data_dir), target)
    score = compute_sr(compute_pl(pred, target))

    print(f"data_dir: {data_dir}")
    print(f"submission: {submission_path}")
    print(f"target: {target_path.name} shape={target.shape}")
    print(f"prediction: shape={pred.shape}")
    print(f"SHARPE: {score:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
