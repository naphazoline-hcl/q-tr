"""長時間処理の進捗をファイルへ書き出す小さなヘルパー（中断対応の共通規約）。

v0.app の 1 メッセージ 25 分制限やローカルの長時間学習で処理が中断されても、
「どこまで完了したか」をファイルから復元できるようにするための共通部品。

使い方（生成コード側）::

    from progress import Progress

    prog = Progress("work/progress/train_v2.json", total=6, meta={"strategy": "v0_multifactor"})
    for i, (k, seed) in enumerate(jobs, start=1):
        prog.update(i, f"k={k} seed={seed} 学習開始")
        model = fit(...)
        model.save_model(...)          # 成果物は都度保存（中断しても再利用できるように）
        prog.update(i, f"k={k} seed={seed} 完了", artifact=f"models/lgbm_k{k}_s{seed}.txt")
    prog.done(note="全ジョブ完了")

書き出されるもの:
- `<path>`: 1 行 1 レコードの JSON Lines（追記のみ・壊れにくい）
- 同じディレクトリの `STATE.md`: 最新状態の要約（人間が読む用）

CLI::

    python tools/progress.py show work/progress/train_v2.json   # 現在の状態を表示
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


class Progress:
    def __init__(self, path: str | os.PathLike[str], total: int | None = None, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = self.path.with_name("STATE.md")
        self.total = total
        self.meta = dict(meta or {})
        self.started = time.time()
        self.last: dict[str, Any] = {}
        self._write(
            {
                "event": "start",
                "total": total,
                "meta": self.meta,
                "time": _now(),
            }
        )

    # -- public ---------------------------------------------------------
    def update(self, step: int, message: str = "", **extra: Any) -> None:
        record = {
            "event": "update",
            "step": step,
            "total": self.total,
            "message": message,
            "elapsed_sec": round(time.time() - self.started, 1),
            "time": _now(),
        }
        record.update(extra)
        self.last = record
        self._write(record)

    def done(self, message: str = "completed", **extra: Any) -> None:
        record = {
            "event": "done",
            "total": self.total,
            "message": message,
            "elapsed_sec": round(time.time() - self.started, 1),
            "time": _now(),
        }
        record.update(extra)
        self.last = record
        self._write(record)

    def failed(self, message: str, **extra: Any) -> None:
        record = {
            "event": "failed",
            "total": self.total,
            "message": message,
            "elapsed_sec": round(time.time() - self.started, 1),
            "time": _now(),
        }
        record.update(extra)
        self.last = record
        self._write(record)

    # -- internals ------------------------------------------------------
    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.state_path.write_text(_render_state(self.path, record), encoding="utf-8")

    # -- resume support -------------------------------------------------
    @staticmethod
    def completed_steps(path: str | os.PathLike[str], key: str = "message") -> list[str]:
        """再開用: これまでに完了した更新メッセージの一覧を返す。"""
        records = load_records(path)
        return [str(r.get(key, "")) for r in records if r.get("event") == "update"]


def load_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _render_state(path: Path, record: dict[str, Any]) -> str:
    lines = [
        "# 進捗状態（自動生成・編集しない）",
        "",
        f"- ファイル: `{path.as_posix()}`",
        f"- 最終更新: {record.get('time', '-')}",
        f"- 状態: {record.get('event', '-')}",
        f"- 進捗: {record.get('step', '-')} / {record.get('total', '-')}",
        f"- メッセージ: {record.get('message', '-')}",
        f"- 経過秒: {record.get('elapsed_sec', '-')}",
    ]
    meta = record.get("meta")
    if meta:
        lines += ["", "## meta", "", "```json", json.dumps(meta, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="進捗ファイルの内容を表示する")
    parser.add_argument("path", help="progress JSONL のパス")
    args = parser.parse_args()
    records = load_records(args.path)
    if not records:
        print(f"no records: {args.path}")
        return 1
    last = records[-1]
    print(f"file   : {args.path}")
    print(f"records: {len(records)}")
    print(f"state  : {last.get('event')} step={last.get('step')}/{last.get('total')} elapsed={last.get('elapsed_sec')}s")
    print(f"message: {last.get('message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
