"""提出用 zip を作る（create_zip.ipynb と同一の除外ルール）。

`submission.py` が無いフォルダはエラーで停止する。作った zip は
`evaluate_script.py --submission <zip>` でそのままローカル採点できる
（採点コードは zip 内を探索して submission.py を見つける）。

使い方::

    python tools/make_zip.py --name v0_multifactor
    python tools/make_zip.py --name v0_multifactor --score     # zip で採点まで実行
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", "__MACOSX", ".mypy_cache", ".pytest_cache"}


def build(name: str, strategies_dir: Path) -> Path:
    strategy_dir = strategies_dir / name
    if not strategy_dir.is_dir():
        raise FileNotFoundError(f"戦略フォルダが見つかりません: {strategy_dir.resolve()}")
    submission_file = strategy_dir / "submission.py"
    if not submission_file.is_file():
        raise FileNotFoundError(f"提出条件を満たしていません。submission.py がありません: {submission_file.resolve()}")

    zip_path = strategies_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(strategy_dir.rglob("*")):
            relative = path.relative_to(strategy_dir)
            if (
                not path.is_file()
                or any(part in JUNK_DIRS for part in relative.parts)
                or path.name == ".DS_Store"
                or path.name.startswith("._")
            ):
                continue
            archive.write(path, Path(name) / relative)

    print(f"提出ZIP: {zip_path.resolve()}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        total = sum(item.file_size for item in archive.infolist())
    print(f"ZIP内容: {names}")
    print(f"展開後サイズ: {total / 1024 / 1024:.2f} MB")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="提出用 zip を作成")
    parser.add_argument("--name", required=True, help="strategies/<name> を zip 化")
    parser.add_argument("--strategies-dir", default=str(ROOT / "strategies"))
    parser.add_argument("--score", action="store_true", help="作成した zip を evaluate_script.py で採点")
    parser.add_argument("--split", default="valid", choices=["valid", "train"])
    args = parser.parse_args()

    zip_path = build(args.name, Path(args.strategies_dir))
    if args.score:
        target = f"target_1day_{args.split}.parquet"
        command = [
            sys.executable,
            str(ROOT / "evaluate_script.py"),
            "--submission",
            str(zip_path),
            "--target",
            target,
        ]
        print("run:", " ".join(command))
        return subprocess.call(command, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
