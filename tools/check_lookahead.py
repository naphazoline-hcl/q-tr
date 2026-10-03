"""先読み（look-ahead）禁止パターンの静的検査＋実行時ガード。

コンペのルール（README「すべてのサンプルに共通のルール」）:

- `target_1day_valid.parquet` / `raw_target_1day_*` を予測コードから読まない
- 特徴量で負の `shift`、`center` 付き rolling、`bfill`、
  別日付を混ぜた順位化・標準化を使わない

このスクリプトは戦略フォルダ内の `.py` を走査して違反の疑いを報告する。
`--runtime` を付けると、対象フォルダの `submission.py` を import して
`predict()` を実行し、禁止ファイルの読み込みが起きた瞬間に例外を出す。

使い方::

    python tools/check_lookahead.py --submission strategies/v0_multifactor
    python tools/check_lookahead.py --submission strategies/v0_multifactor --runtime

戻り値: 違反（ERROR）があれば 1、警告のみなら 0。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (正規表現, 重大度, 説明)
PATTERNS: list[tuple[str, str, str]] = [
    (r"\bbfill\b|\bbackfill\b|method\s*=\s*[\"']bfill[\"']", "ERROR", "bfill（未来の値で埋める）"),
    (r"\bffill\b.*\blimit_area\s*=\s*[\"']both[\"']", "ERROR", "limit_area='both' は未来方向にも埋める"),
    (r"center\s*=\s*True", "ERROR", "center 付き rolling（前後を使う）"),
    (r"\.shift\(\s*-", "ERROR", "負の shift（未来参照）"),
    (r"shift\(\s*-\s*\d", "ERROR", "負の shift（未来参照）"),
    (r"\[\s*::\s*-1\s*\]", "ERROR", "逆順スライス（未来から過去の並び）"),
    (r"\.iloc\[\s*::\s*-1", "ERROR", "iloc 逆順"),
    (r"read_parquet\([^)]*target", "ERROR", "target 系 parquet の読み込み"),
    (r"read_parquet\([^)]*raw_target", "ERROR", "raw_target 系 parquet の読み込み"),
    (r"[\"']raw_target_1day", "ERROR", "raw_target ファイル名の参照"),
    (r"[\"']target_1day", "ERROR", "target ファイル名の参照（学習用の明示許可時を除く）"),
    (r"\.interpolate\(", "WARN", "interpolate は前後両方向の値を使う（時系列では要注意）"),
    (r"sort_index\(\s*ascending\s*=\s*False", "WARN", "降順ソート（直後の操作によっては未来参照）"),
    (r"expanding\([^)]*\)\.apply", "WARN", "expanding.apply は計算量が大きく採点時間を圧迫しうる"),
]

# 学習コード（train系）は target_1day_train を読んでよい。submission/推論コードでは禁止。
TRAIN_ALLOWED_FILES = {"train.py", "train_v2.py", "walkforward_fold.py", "alpha.py"}
ALLOW_MARKER = "check_lookahead: allow"


def scan_file(path: Path, role: str) -> list[tuple[str, int, str, str]]:
    issues: list[tuple[str, int, str, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if ALLOW_MARKER in stripped:
            continue
        if stripped.startswith("#"):
            continue
        for pattern, severity, message in PATTERNS:
            if not re.search(pattern, line):
                continue
            if severity == "ERROR" and "target_1day" in message and role == "train":
                continue
            issues.append((severity, number, message, stripped[:120]))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="先読み禁止パターンの検査")
    parser.add_argument("--submission", required=True, help="戦略フォルダ（または zip / submission.py）")
    parser.add_argument("--runtime", action="store_true", help="submission.predict() を実行時ガード付きで動かす")
    parser.add_argument("--data-dir", default=str(ROOT / "input"))
    args = parser.parse_args()

    target_path = Path(args.submission)
    if target_path.is_file() and target_path.suffix == ".zip":
        import zipfile

        work = ROOT / "work" / "runs" / "_lookahead_tmp"
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_path) as zf:
            zf.extractall(work)
        folder = work
    elif target_path.is_file():
        folder = target_path.parent
    else:
        folder = target_path

    files = sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)
    total_errors = 0
    print(f"scan: {folder}  ({len(files)} python files)")
    for path in files:
        role = "train" if path.name in TRAIN_ALLOWED_FILES else "predict"
        for severity, number, message, snippet in scan_file(path, role):
            print(f"  [{severity}] {path.name}:{number} {message} | {snippet}")
            if severity == "ERROR":
                total_errors += 1

    if args.runtime:
        print("runtime guard: submission.predict() を実行する（禁止ファイル読込で失敗させる）")
        import pandas as pd

        real_read_parquet = pd.read_parquet
        banned = ("target_1day", "raw_target_1day")

        def guarded_read_parquet(path, *pargs, **kwargs):
            name = str(path)
            if any(token in name for token in banned):
                raise RuntimeError(f"[lookahead guard] 予測コードから禁止ファイルを読もうとした: {name}")
            return real_read_parquet(path, *pargs, **kwargs)

        sys.path.insert(0, str(ROOT))
        import evaluate_script as ev  # noqa: E402

        pd.read_parquet = guarded_read_parquet
        try:
            pred = ev.load_prediction(target_path.resolve(), Path(args.data_dir).resolve())
            print(f"runtime guard OK: prediction shape={pred.shape}")
        finally:
            pd.read_parquet = real_read_parquet

    if total_errors:
        print(f"RESULT: NG ({total_errors} errors)")
        return 1
    print("RESULT: OK (no ERROR patterns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
