from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_links(text_doc: Dict, *, input_path: Optional[Path] = None) -> Dict:
    """
    Build a per-page links sidecar from a text_content.json document produced by the crawler.

    Output schema:
    {
      "page_url": str,
      "page_title": str|None,
      "links": [
        {
          "text": str,
          "href": str,
          "heading_path": [str, ...],
          "xpath": str|None,
          "flags": {"isNav": bool, "isAction": bool, "isParagraph": bool, "isTitle": bool}
        }, ...
      ]
    }
    """
    page_url = text_doc.get("source")
    links: List[Dict] = []

    # Optional page title from sibling metadata.json
    page_title = None
    if input_path is not None:
        meta_path = input_path.parent / "metadata.json"
        try:
            if meta_path.exists():
                meta = _load_json(meta_path)
                page_title = meta.get("title")
        except Exception:
            page_title = None

    heading_path: List[str] = []
    for it in text_doc.get("content") or []:
        text = (it.get("content") or "").strip()
        if not text:
            continue
        meta = it.get("meta") or {}
        # Track headings in order of appearance
        if meta.get("isTitle") and text not in heading_path:
            heading_path.append(text)
            continue
        href = meta.get("href")
        if href:
            links.append({
                "text": text,
                "href": href,
                "heading_path": list(heading_path),
                "xpath": it.get("xpath"),
                "flags": {
                    "isNav": bool(meta.get("isNav")),
                    "isAction": bool(meta.get("isAction")),
                    "isParagraph": bool(meta.get("isParagraph")),
                    "isTitle": bool(meta.get("isTitle")),
                },
            })

    return {"page_url": page_url, "page_title": page_title, "links": links}


def _file_page_links(input_path: Path) -> Dict:
    """Build minimal links.json for file-backed pages (no text_content.json)."""
    page_dir = input_path if input_path.is_dir() else input_path.parent
    meta_path = page_dir / "metadata.json"
    page_url = None
    page_title = None
    try:
        if meta_path.exists():
            meta = _load_json(meta_path)
            page_url = meta.get("final_url") or meta.get("url")
            page_title = meta.get("title")
    except Exception:
        pass
    return {"page_url": page_url, "page_title": page_title, "links": [], "isFile": True}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Extract links from a text_content.json into links.json structure")
    p.add_argument("input", help="Path to text_content.json")
    p.add_argument("--write", action="store_true", help="Write to sibling links.json instead of stdout")
    args = p.parse_args(argv)

    input_path = Path(args.input)

    # Attempt to process depending on presence of text_content.json / content.txt / page dir
    if input_path.is_dir():
        out_obj = _file_page_links(input_path)
    elif input_path.suffix.lower() == ".json" and input_path.name == "text_content.json" and input_path.exists():
        try:
            doc = _load_json(input_path)
            out_obj = extract_links(doc, input_path=input_path)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to parse {input_path}: {e}\n")
            out_obj = _file_page_links(input_path)
    elif input_path.suffix.lower() == ".txt" or not input_path.exists():
        # content.txt or missing json -> file page
        out_obj = _file_page_links(input_path)
    else:
        # Try generic JSON
        try:
            doc = _load_json(input_path)
            out_obj = extract_links(doc, input_path=input_path)
        except Exception as e:
            sys.stderr.write(f"[WARN] Could not read/parse {input_path}: {e}\n")
            out_obj = _file_page_links(input_path)

    out_path = (input_path.parent if input_path.is_file() else input_path) / "links.json"
    if args.write:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
    else:
        json.dump(out_obj, fp=sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
