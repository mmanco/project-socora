from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_UTILITY_WORDS = {"home", "search"}


def _get_utility_words(input_path: Optional[Path]) -> Set[str]:
    """Return the set of utility/navigation words to exclude.
    Sources, in priority:
      1) .output/<run-id>/normalize_config.json: {"utility_words": [...]} or {"extra_utility_words": [...]}
      2) env NORM_UTILITY_WORDS (comma-separated)
      3) default minimal set {"home", "search"}
    """
    words: Set[str] = set(DEFAULT_UTILITY_WORDS)
    # From env var
    env = os.getenv("NORM_UTILITY_WORDS")
    if env:
        for tok in env.split(","):
            tok = tok.strip().lower()
            if tok:
                words.add(tok)
    # From run-level config
    try:
        if input_path is not None:
            run_root = _find_run_root_from_input(input_path)
            if run_root is not None:
                repo_root = _find_repo_root(run_root)
                cfg_path = repo_root / ".output" / run_root.name / "normalize_config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if isinstance(cfg, dict):
                        if isinstance(cfg.get("utility_words"), list):
                            words = {str(x).strip().lower() for x in cfg["utility_words"] if str(x).strip()}
                        elif isinstance(cfg.get("extra_utility_words"), list):
                            words.update({str(x).strip().lower() for x in cfg["extra_utility_words"] if str(x).strip()})
    except Exception:
        pass
    return words


def _is_utility_word(text: str, utility_words: Set[str]) -> bool:
    t = text.strip().lower()
    return t in utility_words


def _looks_like_content(text: str, utility_words: Optional[Set[str]] = None) -> bool:
    s = text.strip()
    if not s:
        return False
    low = s.lower()

    # Exclude obvious non-content
    if utility_words is not None and _is_utility_word(s, utility_words):
        return False
    if low in {"ok", "yes", "no"}:
        return False
    if all(ch in "|•·•-–—_" for ch in low):
        return False
    if any(bad in low for bad in ("powered by", "cookies", "cookie", "privacy policy", "terms of use", "copyright", "©")):
        return False

    # Dates: 11.18.2025, 11/18/2025, 2025-11-18, Nov 18, 2025
    date_patterns = [
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?\b",
    ]
    if any(re.search(p, low) for p in date_patterns):
        return True

    # Times and hours like 8:00 - 4:00, 8:30 AM – 5:00 PM
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*[\-–—]\s*\d{1,2}:\d{2}\s*(?:am|pm)?\b", low):
        return True

    # Standalone time or office hours hints
    if re.search(r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", low) and any(w in low for w in ("am", "pm", "hours", "open", "close", "closing", "opening")):
        return True

    # Phone numbers
    if re.search(r"\b(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b", low):
        return True

    # Addresses: number + street type
    street_types = ("st", "street", "ave", "avenue", "rd", "road", "blvd", "drive", "dr", "court", "ct", "lane", "ln", "way", "place", "pl", "circle", "cir")
    if re.search(r"\b\d{1,6}\s+\S+(?:\s+\S+){0,3}\s+(?:" + "|".join(street_types) + r")\b", low):
        return True

    # Short factual lines (<= 80 chars) with digits or colon
    if len(s) <= 80 and (any(ch.isdigit() for ch in s) or ":" in s):
        return True

    # Moderate-length factual sentence (contains a period but not marketing-ish words)
    if 20 <= len(s) <= 240 and "." in s and not any(w in low for w in ("learn more", "click here", "read more")):
        return True

    return False


def _is_vendor_or_footer(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    bads = [
        "powered by",
        "all rights reserved",
        "copyright",
        "©",
        "privacy policy",
        "terms of use",
        "website terms",
        "cookie",
        "cookies",
    ]
    return any(b in low for b in bads)


_INVISIBLE_CHARS_RE = re.compile(r"[\u200B\u200C\u200D\u2060\ufeff]")


def _strip_invisible(s: str) -> str:
    return _INVISIBLE_CHARS_RE.sub("", s)


def _link_display_text(text: str) -> str:
    """Return a display label for a link using only the given text.
    If text is empty or only invisible characters, return an empty string so callers can skip.
    """
    raw = (text or "").strip()
    cleaned = _strip_invisible(raw)
    return raw if cleaned.strip() else ""


def _find_run_root_from_input(input_path: Path) -> Optional[Path]:
    for p in [input_path] + list(input_path.parents):
        if p.name.startswith("run-") and p.is_dir():
            return p
    return None


def _find_repo_root(start: Path) -> Path:
    cur = start
    while True:
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent


def _enumerate_text_content_files(run_root: Path) -> List[Path]:
    return sorted(run_root.rglob("content.json"))


def _infer_crawler_depth(run_root: Path) -> Optional[int]:
    env_vars = [
        "NORM_CRAWLER_DEPTH",
        "CRAWLER_DEPTH",
        "CRAWLER_MAX_DEPTH",
        "MAX_DEPTH",
    ]
    for var in env_vars:
        val = os.getenv(var)
        if not val:
            continue
        try:
            depth = int(val)
        except (TypeError, ValueError):
            continue
        if depth >= 0:
            return depth
    repo_root = _find_repo_root(run_root)
    candidate_files = [
        repo_root / ".output" / run_root.name / "normalize_config.json",
        repo_root / ".output" / run_root.name / "run_config.json",
        run_root / "normalize_config.json",
        run_root / "run_config.json",
        run_root / "crawl_config.json",
        run_root / "run_meta.json",
    ]
    keys = ("crawler_depth", "max_depth", "depth", "maxDepth", "crawl_depth", "crawlerDepth")
    for cfg_path in candidate_files:
        try:
            if not cfg_path.exists():
                continue
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key in keys:
                if key in data:
                    val = data.get(key)
                    try:
                        depth = int(val)
                    except (TypeError, ValueError):
                        continue
                    if depth >= 0:
                        return depth
    return None


def _prune_commonalities_by_depth(commons: Dict, depth: Optional[int]) -> Dict:
    if not isinstance(commons, dict):
        return commons
    if depth is None:
        return commons
    try:
        threshold = int(depth)
    except (TypeError, ValueError):
        return commons
    if threshold < 0:
        return commons
    for key in ("text_page_freq", "nav_text_page_freq", "action_text_page_freq"):
        freq_map = commons.get(key)
        if isinstance(freq_map, dict):
            commons[key] = {k: v for k, v in freq_map.items() if _safe_int(v) > threshold}
    commons["crawler_depth"] = threshold
    return commons


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_commonalities(run_root: Path) -> Dict:
    files = _enumerate_text_content_files(run_root)
    pages_count = 0
    text_to_pages: Dict[str, Set[str]] = {}
    nav_text_to_pages: Dict[str, Set[str]] = {}
    action_text_to_pages: Dict[str, Set[str]] = {}

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue
        items = doc.get("content") or []
        if not isinstance(items, list):
            continue
        pages_count += 1
        page_id = str(f.parent)
        seen_texts_page: Set[str] = set()
        seen_texts_nav_page: Set[str] = set()
        seen_texts_action_page: Set[str] = set()
        for it in items:
            text = (it.get("content") or "").strip()
            if not text:
                continue
            meta = it.get("meta") or {}
            seen_texts_page.add(text)
            if meta.get("isNav"):
                seen_texts_nav_page.add(text)
            if meta.get("isAction"):
                seen_texts_action_page.add(text)
        for t in seen_texts_page:
            text_to_pages.setdefault(t, set()).add(page_id)
        for t in seen_texts_nav_page:
            nav_text_to_pages.setdefault(t, set()).add(page_id)
        for t in seen_texts_action_page:
            action_text_to_pages.setdefault(t, set()).add(page_id)

    commons = {
        "run_id": run_root.name,
        "pages_count": pages_count,
        "text_page_freq": {k: len(v) for k, v in text_to_pages.items()},
        "nav_text_page_freq": {k: len(v) for k, v in nav_text_to_pages.items()},
        "action_text_page_freq": {k: len(v) for k, v in action_text_to_pages.items()},
    }
    depth = _infer_crawler_depth(run_root)
    commons = _prune_commonalities_by_depth(commons, depth if depth is not None else commons.get("crawler_depth"))
    return commons


def _load_or_build_commonalities(input_path: Path, force: bool = False) -> Optional[Dict]:
    run_root = _find_run_root_from_input(input_path)
    if not run_root:
        return None
    repo_root = _find_repo_root(run_root)
    cache_dir = repo_root / "output" / run_root.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "text_commonalities.json"
    if cache_file.exists() and not force:
        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if not isinstance(cached, dict):
                return cached
            inferred_depth = _infer_crawler_depth(run_root)
            depth_to_use = inferred_depth if inferred_depth is not None else cached.get("crawler_depth")
            cached = _prune_commonalities_by_depth(cached, depth_to_use)
            if depth_to_use is not None:
                try:
                    with cache_file.open("w", encoding="utf-8") as fh:
                        json.dump(cached, fh, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            return cached
        except Exception:
            pass
    commons = _compute_commonalities(run_root)
    try:
        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(commons, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return commons


def normalize_page_content(
    doc: Dict,
    *,
    input_path: Optional[Path] = None,
    force_recompute: bool = False,
    common_threshold: Optional[float] = None,
    disable_commonalities: bool = False,
) -> str:
    items = doc.get("content") or []
    lines: List[str] = []
    current_heading: Optional[str] = None
    seen_texts: Set[str] = set()

    # Dynamic utility words (site-agnostic defaults; configurable via env or run config)
    utility_words = _get_utility_words(input_path)

    # Cross-run commonalities
    commons = None
    if (not disable_commonalities) and (input_path is not None):
        commons = _load_or_build_commonalities(input_path, force=force_recompute)
    if common_threshold is None:
        try:
            common_threshold = float(os.getenv("NORM_COMMON_RATIO", "0.4"))
        except Exception:
            common_threshold = 0.4

    # Pre-scan to group table cells by ancestor table and their row/col indices
    def _parse_table_info(xpath: str) -> Optional[Tuple[str, int, int, bool]]:
        # Returns (table_key, row_index, col_index, is_header_cell)
        try:
            # Find nearest table ancestor key
            last_table = None
            for m in re.finditer(r"/table(?:\[\d+\])?", xpath):
                last_table = m
            if not last_table:
                return None
            table_key = xpath[: last_table.end()]
            # Row index (last tr[n])
            row = None
            for m in re.finditer(r"/tr\[(\d+)\]", xpath):
                row = int(m.group(1))
            if row is None:
                return None
            # Column index: last td[n] or th[n]
            last_td = None
            for m in re.finditer(r"/td\[(\d+)\]", xpath):
                last_td = m
            last_th = None
            for m in re.finditer(r"/th\[(\d+)\]", xpath):
                last_th = m
            is_header = False
            col = None
            if last_td and last_th:
                if last_th.start() > last_td.start():
                    is_header = True
                    col = int(last_th.group(1))
                else:
                    col = int(last_td.group(1))
            elif last_th:
                is_header = True
                col = int(last_th.group(1))
            elif last_td:
                col = int(last_td.group(1))
            else:
                # Might be nested deeper; try to catch td/th earlier in the path tail
                return None
            return table_key, row, col, is_header
        except Exception:
            return None

    tables: Dict[str, Dict] = {}
    item_table_key: Dict[int, str] = {}
    for idx, it in enumerate(items):
        xp = it.get("xpath") or ""
        info = _parse_table_info(xp)
        if not info:
            continue
        table_key, row, col, is_header = info
        meta = it.get("meta") or {}
        # Hard exclusion for action items even inside tables
        if bool(meta.get("isAction")):
            continue
        text = (it.get("content") or "").strip()
        if not text:
            continue
        # If this text node is part of a link, render as Markdown link with safe label
        href = (meta.get("href") or "").strip()
        if href:
            label = _link_display_text(text)
            display = f"[{label}]({href})" if label else ""
        else:
            display = text
        agg = tables.setdefault(table_key, {
            "first_index": idx,
            "cells": {},  # (row, col) -> List[str]
            "header_rows": set(),
            "row_set": set(),
            "col_set": set(),
        })
        agg["row_set"].add(row)
        agg["col_set"].add(col)
        if is_header:
            agg["header_rows"].add(row)
        if display:
            agg["cells"].setdefault((row, col), []).append(display)
        item_table_key[idx] = table_key

    # Decide which tables to include (data tables only)
    include_table: Dict[str, bool] = {}
    for key, agg in tables.items():
        rows = len(agg["row_set"]) if agg["row_set"] else 0
        cols = len(agg["col_set"]) if agg["col_set"] else 0
        has_header = bool(agg["header_rows"])
        include_table[key] = has_header or (rows >= 2 and cols >= 2)

    # Render helper for a single table to Markdown
    def _render_table_md(agg: Dict) -> List[str]:
        row_indices = sorted(agg["row_set"]) if agg["row_set"] else []
        col_indices = sorted(agg["col_set"]) if agg["col_set"] else []
        if not row_indices or not col_indices:
            return []
        header_row_index: Optional[int] = None
        if agg["header_rows"]:
            header_row_index = min(agg["header_rows"])  # pick first header row
        else:
            header_row_index = row_indices[0]  # first row as header

        def cell_text(r: int, c: int) -> str:
            vals = agg["cells"].get((r, c))
            if not vals:
                return ""
            # Join multiple text nodes within a cell with a space
            s = " ".join(v.strip() for v in vals if v and v.strip())
            return s

        header_cells = [cell_text(header_row_index, c) for c in col_indices]
        # If header row is completely empty, skip rendering
        if not any(h.strip() for h in header_cells):
            return []
        md_lines: List[str] = []
        md_lines.append("| " + " | ".join(header_cells) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(col_indices)) + " |")
        for r in row_indices:
            if r == header_row_index:
                continue
            row_cells = [cell_text(r, c) for c in col_indices]
            md_lines.append("| " + " | ".join(row_cells) + " |")
        return md_lines

    def include_item(text: str, meta: Dict[str, bool]) -> Tuple[bool, str]:
        t = text.strip()
        if not t:
            return False, ""
        is_nav = bool(meta.get("isNav"))
        is_action = bool(meta.get("isAction"))
        is_par = bool(meta.get("isParagraph"))
        is_title = bool(meta.get("isTitle"))

        # Hard exclusions by text value (utility nav words, vendor credits)
        if _is_utility_word(t, utility_words):
            return False, ""
        if _is_vendor_or_footer(t):
            return False, ""

        # Exclude globally common strings across the run for all non-title items
        if commons and commons.get("pages_count") and not is_title:
            freq = (commons.get("text_page_freq", {}).get(t) or 0)
            ratio = freq / max(1, commons["pages_count"])
            if ratio >= (common_threshold or 0.4):
                return False, ""

        # Inclusions
        if is_title:
            return True, t
        if is_par:
            return True, t
        # Plain items — include if not globally common (repetitive) and not utility/vendor
        if not is_title and not is_par:
            return True, t

        return False, ""

    emitted_tables: Set[str] = set()
    first_line = True
    for idx, it in enumerate(items):
        # If this item belongs to a data table, render the table once at the first occurrence
        tkey = item_table_key.get(idx)
        if tkey and include_table.get(tkey) and tkey not in emitted_tables:
            if not first_line and (not lines or lines[-1] != ""):
                lines.append("")
            md_tbl = _render_table_md(tables[tkey])
            if md_tbl:
                lines.extend(md_tbl)
                lines.append("")
                first_line = False
            emitted_tables.add(tkey)
            # Skip normal processing for this and subsequent cells of the same table
            continue

        text = it.get("content") or ""
        meta = it.get("meta") or {}

        ok, t = include_item(text, meta)
        if not ok:
            continue

        # Deduplicate body lines only (titles can repeat legitimately)
        is_title = bool(meta.get("isTitle"))
        if not is_title:
            if t in seen_texts:
                continue
            seen_texts.add(t)

        if is_title:
            # New section heading
            if not first_line and (not lines or lines[-1] != ""):
                lines.append("")
            lines.append(f"## {t}")
            current_heading = t
            first_line = False
        else:
            # Body line
            href = (meta.get("href") or "").strip()
            if href:
                label = _link_display_text(t)
                if not label:
                    # Skip links with no visible text
                    continue
                rendered = f"[{label}]({href})"
            else:
                rendered = t
            lines.append(rendered)
            first_line = False

    # Build YAML front matter using metadata.json if available
    fm_lines: List[str] = []
    title_val: Optional[str] = None
    url_val: Optional[str] = None
    fetched_at_val: Optional[str] = None
    run_id_val: Optional[str] = None

    if input_path is not None:
        meta_path = input_path.parent / "metadata.json"
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                title_val = meta.get("title") or title_val
                url_val = meta.get("final_url") or meta.get("url") or url_val
                fetched_at_val = meta.get("fetched_at") or fetched_at_val
        except Exception:
            pass
        run_root = _find_run_root_from_input(input_path)
        if run_root is not None:
            run_id_val = run_root.name

    if not title_val:
        for it in items:
            if (it.get("meta") or {}).get("isTitle"):
                cand = (it.get("content") or "").strip()
                if cand:
                    title_val = cand
                    break
    if not url_val:
        url_val = doc.get("source")

    def _yaml_kv(k: str, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return f"{k}: {json.dumps(v, ensure_ascii=False)}"

    for line in [
        _yaml_kv("title", title_val),
        _yaml_kv("url", url_val),
        _yaml_kv("run_id", run_id_val),
        _yaml_kv("fetched_at", fetched_at_val),
    ]:
        if line:
            fm_lines.append(line)

    output_parts: List[str] = []
    if fm_lines:
        output_parts.append("---")
        output_parts.extend(fm_lines)
        output_parts.append("---")
        output_parts.append("")
    output_parts.extend(lines)

    # Ensure single trailing newline in output
    return "\n".join(output_parts).rstrip() + ("\n" if output_parts else "")


def _load_json_file(path: str) -> Dict:
    p = Path(path)
    if str(p) == "-":
        return json.load(sys.stdin)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a content.json or content.txt into Markdown content")
    parser.add_argument("input", help="Path to content.json or content.txt (or - for stdin)")
    parser.add_argument("--force-commonalities", action="store_true", help="Recompute cross-page commonalities cache for this run")
    parser.add_argument("--common-threshold", type=float, default=None, help="Ratio [0,1] above which strings are considered common and excluded (default 0.4 or NORM_COMMON_RATIO env)")
    parser.add_argument("--disable-commonalities", action="store_true", help="Disable cross-page commonality filtering (include all items that pass other gates)")
    args = parser.parse_args(argv)

    input_path = None if args.input == "-" else Path(args.input)
    # Support direct text files (e.g., content.txt) by emitting front matter + raw text
    if input_path and input_path.suffix.lower() in {".txt"}:
        try:
            raw_text = input_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            sys.stderr.write(f"[WARN] Could not read text file: {input_path}: {e}\n")
            return 0
        # Build front matter similar to page normalization
        fm_lines: List[str] = []
        title_val: Optional[str] = None
        url_val: Optional[str] = None
        fetched_at_val: Optional[str] = None
        run_id_val: Optional[str] = None

        meta_path = input_path.parent / "metadata.json"
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                title_val = meta.get("title") or title_val
                url_val = meta.get("final_url") or meta.get("url") or url_val
                fetched_at_val = meta.get("fetched_at") or fetched_at_val
        except Exception:
            pass
        run_root = _find_run_root_from_input(input_path)
        if run_root is not None:
            run_id_val = run_root.name

        def _yaml_kv(k: str, v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            return f"{k}: {json.dumps(v, ensure_ascii=False)}"

        for line in [
            _yaml_kv("title", title_val),
            _yaml_kv("url", url_val),
            _yaml_kv("run_id", run_id_val),
            _yaml_kv("fetched_at", fetched_at_val),
        ]:
            if line:
                fm_lines.append(line)

        parts: List[str] = []
        if fm_lines:
            parts.append("---")
            parts.extend(fm_lines)
            parts.append("---")
            parts.append("")
        parts.append(raw_text.rstrip())
        parts.append("")
        sys.stdout.write("\n".join(parts))
    else:
        # Default: treat as JSON text_content document
        try:
            doc = _load_json_file(args.input)
        except Exception as e:
            sys.stderr.write(f"[WARN] Could not read/parse JSON file: {args.input}: {e}\n")
            return 0
        md = normalize_page_content(
            doc,
            input_path=input_path,
            force_recompute=args.force_commonalities,
            common_threshold=args.common_threshold,
            disable_commonalities=args.disable_commonalities,
        )
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
