"""配布 parquet を走査して `docs/data_schema.md` を生成する。

v0.app は input/ を取得できないため、列名・型・欠損・値の例をここで1枚にまとめ、
公開リポジトリ経由でプロンプトから参照させる。あわせて target の定義式
（raw_target = raw_return の2営業日先、target = 市場残差）を実データで検算する。

使い方::

    python tools/profile_data.py                 # docs/data_schema.md を生成
    python tools/profile_data.py --out docs/data_schema.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"

SMALL_FILE_BYTES = 30_000_000  # これ以下は全行を読んで欠損・ユニーク数を数える


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024.0
    return f"{size:.1f}GB"


def sample_frame(path: Path, rows: int = 5) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    return next(pf.iter_batches(batch_size=rows)).to_pandas()


def date_range(path: Path, column: str = "Date") -> tuple[str, str, int]:
    table = pq.read_table(path, columns=[column])[column]
    series = pd.Series(table.to_pandas())
    return str(series.min()), str(series.max()), int(series.nunique())


def column_table(path: Path, sample: pd.DataFrame, full: pd.DataFrame | None) -> list[str]:
    lines = ["| 列名 | dtype | 欠損率 | ユニーク数 | 例（先頭の非欠損値） |", "|---|---|---|---|---|"]
    for name in sample.columns:
        dtype = str(sample[name].dtype)
        if full is not None:
            null_rate = float(full[name].isna().mean())
            nunique = int(full[name].nunique(dropna=True))
        else:
            null_rate = float(sample[name].isna().mean())
            nunique = int(sample[name].nunique(dropna=True))
        values = sample[name].dropna()
        example = "" if values.empty else str(values.iloc[0])[:48]
        lines.append(f"| `{name}` | {dtype} | {null_rate * 100:.1f}% | {nunique} | {example} |")
    return lines


def profile_file(path: Path) -> list[str]:
    pf = pq.ParquetFile(path)
    size = path.stat().st_size
    rows = pf.metadata.num_rows
    sample = sample_frame(path)
    full = None
    if size <= SMALL_FILE_BYTES:
        try:
            full = pd.read_parquet(path)
            sample = full.head(5)
        except Exception:
            full = None

    lines = [f"### `{path.name}`", ""]
    index_names = list(sample.index.names)
    lines.append(f"- 行数: {rows:,} / サイズ: {human(size)} / 列数: {len(sample.columns)}")
    lines.append(f"- 索引: `{index_names}`")
    if "Date" in index_names:
        try:
            start, end, days = date_range(path)
            lines.append(f"- Date 範囲: {start} .. {end}（{days} 日）")
        except Exception as exc:  # pragma: no cover
            lines.append(f"- Date 範囲: 取得失敗 ({exc})")
    if full is not None:
        lines.append("- 統計は全行ベース。")
    else:
        lines.append(f"- 統計は先頭 {len(sample)} 行ベース（大容量のため）。")
    lines.append("")
    lines += column_table(path, sample, full)
    lines.append("")
    lines.append("<details><summary>先頭3行の値</summary>")
    lines.append("")
    lines.append("```text")
    lines.append(sample.head(3).to_string())
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def verify_relations() -> list[str]:
    """target / raw_target の定義と、index の関係を実データで検算する。"""
    lines = ["## 派生系列の検算（実データで確認した事実）", ""]
    target = pd.read_parquet(INPUT / "target_1day_valid.parquet").iloc[:, 0]
    raw_target = pd.read_parquet(INPUT / "raw_target_1day_valid.parquet").iloc[:, 0]
    raw_return = pd.read_parquet(INPUT / "raw_return_1day_valid.parquet").iloc[:, 0]
    beta = pd.read_parquet(INPUT / "beta_1day_valid.parquet").iloc[:, 0]
    topix = pd.read_parquet(INPUT / "topix_return_1day_valid.parquet").iloc[:, 0]

    raw_return_shifted = raw_return.groupby(level="Code", sort=False).shift(-2)
    diff = (raw_target - raw_return_shifted).abs().dropna()
    ok1 = (not diff.empty) and diff.max() < 1e-9
    lines.append(
        f"- `raw_target[t] == raw_return[t+2]`: {'一致' if ok1 else '不一致あり'}"
        f"（最大絶対差 {diff.max():.3e}, 比較可能 {len(diff):,} 行）"
    )

    beta_shifted = beta.groupby(level="Code", sort=False).shift(-2)
    topix_shifted = topix.shift(-2)
    topix_broadcast = pd.Series(
        topix_shifted.reindex(target.index.get_level_values("Date")).to_numpy(),
        index=target.index,
    )
    resid = (target - (raw_target - beta_shifted * topix_broadcast)).abs().dropna()
    ok2 = (not resid.empty) and resid.max() < 1e-9
    lines.append(
        f"- `target[t] == raw_target[t] - beta[t+2] * topix_return[t+2]`: {'一致' if ok2 else '不一致あり'}"
        f"（最大絶対差 {resid.max():.3e}, 比較可能 {len(resid):,} 行）"
    )
    lines.append("")

    pairs = [
        ("target_1day_valid", pd.read_parquet(INPUT / "target_1day_valid.parquet").index),
        ("raw_return_1day_valid", pd.read_parquet(INPUT / "raw_return_1day_valid.parquet").index),
        ("beta_1day_valid", pd.read_parquet(INPUT / "beta_1day_valid.parquet").index),
    ]
    lines.append("| ファイル | 行数 |")
    lines.append("|---|---|")
    for name, index in pairs:
        lines.append(f"| {name} | {len(index):,} |")
    base = pairs[0][1]
    for name, index in pairs[1:]:
        lines.append(f"- `{name}` に無い (Date, Code) 行（target 基準）: {len(base.difference(index)):,}")
    lines.append("")
    lines.append("→ 特徴量とラベルは **index の intersection** で揃えること（sample02 が実例）。")

    codes = pd.Series(base.get_level_values("Code"))
    per_date = pd.Series(base.get_level_values("Date")).value_counts()
    lines.append(f"- valid 期間の銘柄数（target 基準）: {codes.nunique()}（ユニコード文字列。例 `{codes.iloc[0]}`）")
    lines.append(
        f"- 1日あたり行数: 中央値 {int(per_date.median())} / 最小 {int(per_date.min())} / 最大 {int(per_date.max())}"
    )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="docs/data_schema.md を生成")
    parser.add_argument("--out", default=str(ROOT / "docs" / "data_schema.md"))
    args = parser.parse_args()

    manifest_path = ROOT / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = sorted(INPUT.glob("*.parquet"))

    lines = [
        "# 配布データ スキーマ（自動生成）",
        "",
        f"- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "- 生成スクリプト: `tools/profile_data.py`（ローカル実行・再生成可）",
        f"- `input_manifest.json` as_of: {manifest.get('as_of', '-')}",
        f"- ファイル数: {len(files)}",
        "",
        "このファイルは v0.app に渡す参照資料である。v0 は実データを取得できないため、",
        "**ここに書かれた列名・型・欠損・値の例が唯一の根拠**になる。推測で列名を作らないこと。",
        "",
        "## ファイル一覧",
        "",
        "| ファイル | 行数 | 列数 | サイズ |",
        "|---|---|---|---|",
    ]
    for path in files:
        meta = pq.ParquetFile(path).metadata
        sample = sample_frame(path, rows=1)
        lines.append(f"| `{path.name}` | {meta.num_rows:,} | {len(sample.columns)} | {human(path.stat().st_size)} |")
    lines.append("")
    lines += verify_relations()

    for path in files:
        print(f"profiling {path.name} ...", flush=True)
        lines += profile_file(path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
