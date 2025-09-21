#!/usr/bin/env python
"""
List all page output directories where content.md is effectively empty
after the YAML front matter AND metadata.json either has no "files"
property or has an empty "files": [] list.

Usage:
  python scripts/crawler/find_empty_content_md.py [BASE_DIR]

- BASE_DIR defaults to "output" (the project output root).
- Prints one directory path per line for each page whose content.md has no
  body (only front matter and optional blank lines) AND whose metadata.json
  either does not exist, has no "files" field, or has "files": [].
"""
from __future__ import annotations

import sys
from pathlib import Path
import json


def is_effectively_empty_markdown(md_text: str) -> bool:
    """Return True if the markdown has no body beyond the front matter.

    Rules:
    - If the file starts with a front matter block delimited by '---' lines,
      any non-blank content after the closing '---' counts as a body.
    - If there is no front matter, any non-blank content anywhere counts as a body.
    - Lines containing only whitespace are ignored.
    """
    lines = md_text.splitlines()
    i = 0
    n = len(lines)
    # Skip initial blank lines
    while i < n and not lines[i].strip():
        i += 1

    # Check for front matter
    if i < n and lines[i].strip() == "---":
        i += 1
        # Find closing '---'
        while i < n and lines[i].strip() != "---":
            i += 1
        if i < n and lines[i].strip() == "---":
            i += 1  # move past closing delimiter
        # Skip a single optional blank line after front matter
        while i < n and not lines[i].strip():
            i += 1
        # If nothing but blanks remain, it's empty
        return i >= n
    else:
        # No front matter: any non-blank line means non-empty
        for line in lines:
            if line.strip():
                return False
        return True


def _meta_has_files(meta_path: Path) -> bool:
    try:
        if not meta_path.exists():
            return False
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        files = data.get("files")
        # Only consider the "files" array; non-empty means we treat the page as having files
        if isinstance(files, list) and len(files) > 0:
            return True
    except Exception:
        # If metadata is unreadable or malformed, treat as no files
        return False
    return False


def find_empty_pages(base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for md_path in base_dir.rglob("content.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # If unreadable, skip
            continue
        if is_effectively_empty_markdown(text):
            page_dir = md_path.parent
            meta_path = page_dir / "metadata.json"
            if not _meta_has_files(meta_path):
                out.append(page_dir)
    return out


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else Path("output")
    if not base.exists() or not base.is_dir():
        sys.stderr.write(f"[ERROR] Base directory not found: {base}\n")
        return 2
    for d in find_empty_pages(base):
        print(str(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
