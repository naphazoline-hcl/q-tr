"""docs/data_schema.md から列名だけを抜き出す補助ツール（プロンプト作成用）。

    python tools/dump_columns.py                 # 主要3ファイルの列名
    python tools/dump_columns.py --file all      # 全ファイル
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="data_schema.md から列名を抜き出す")
    parser.add_argument("--schema", default=str(ROOT / "docs" / "data_schema.md"))
    parser.add_argument("--file", default="main", help="'main' か 'all' か ファイル名の一部")
    args = parser.parse_args()

    text = Path(args.schema).read_text(encoding="utf-8")
    sections = re.split(r"^### ", text, flags=re.M)
    for section in sections[1:]:
        title = section.split("\n")[0].strip("`")
        if args.file == "main":
            if not any(key in title for key in ("fins_statements_train", "prices_daily_quotes_train", "listed_info_train")):
                continue
        elif args.file != "all" and args.file not in title:
            continue
        columns = re.findall(r"^\| `([A-Za-z0-9_]+)`", section, flags=re.M)
        print(f"\n## {title}  ({len(columns)} columns)")
        for column in columns:
            print(f"  {column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
