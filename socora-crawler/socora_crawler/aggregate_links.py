from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _find_run_root(start: Path) -> Optional[Path]:
    cur = start
    while True:
        if cur.name.startswith("run-") and cur.is_dir():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def _find_repo_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent


def aggregate_to_jsonl(run_dir: Path) -> Path:
    repo_root = _find_repo_root(run_dir)
    out_dir = repo_root / "output" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "links_index.jsonl"

    with out_path.open("w", encoding="utf-8") as outf:
        for links_file in run_dir.rglob("links.json"):
            try:
                data = json.loads(links_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            page_url = data.get("page_url")
            page_title = data.get("page_title")
            for link in data.get("links", []):
                row = {
                    "run_id": run_dir.name,
                    "page_url": page_url,
                    "page_title": page_title,
                    "link_text": link.get("text"),
                    "href": link.get("href"),
                    "heading_path": link.get("heading_path") or [],
                    "xpath": link.get("xpath"),
                    "flags": link.get("flags") or {},
                }
                outf.write(json.dumps(row, ensure_ascii=False))
                outf.write("\n")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate per-page links.json files into a run-level JSONL index")
    p.add_argument("run_dir", help="Run directory (output/run-YYYYmmdd-HHMMSS)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = aggregate_to_jsonl(run_dir)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

