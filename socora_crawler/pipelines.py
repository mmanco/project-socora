import json
import os
import re
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Set
from scrapy import Item
from .tika_client import extract_with_tika, TikaError

MAX_SLUG_LEN = 80


def _slugify(text: str, max_len: int = MAX_SLUG_LEN) -> str:
    slug = re.sub(r"[^A-Za-z0-9\-_.]+", "-", text)
    slug = re.sub(r"-+", "-", slug).strip("-_")
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip("-_")
    return slug or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _slug_with_suffix(base_slug: str, index: int, max_len: int = MAX_SLUG_LEN) -> str:
    suffix = f"-{index}"
    if max_len <= len(suffix):
        digits = str(index)
        return digits if max_len <= 0 else digits[-max_len:]
    available = max_len - len(suffix)
    trimmed = base_slug[:available].rstrip("-_")
    if not trimmed:
        trimmed = base_slug[:available]
    if not trimmed:
        trimmed = "0"
    return f"{trimmed}{suffix}"



class OutputWriterPipeline:
    """
    Writes, for each item, a directory containing:
    - metadata.json: basic metadata including file references if any
    - content.html (for HTML pages) or content.txt (for plain text)
    - screenshot.png (if screenshot bytes present in item)
    Files downloaded by FilesPipeline are referenced via the 'files' field.
    """

    def _make_unique_slug(self, text: str, max_len: int = MAX_SLUG_LEN) -> str:
        if not hasattr(self, '_slug_counts'):
            self._slug_counts = {}
        if not hasattr(self, '_used_slugs'):
            self._used_slugs = set()
        base_slug = _slugify(text, max_len=max_len)
        count = self._slug_counts.get(base_slug, 0)
        slug = base_slug
        run_dir = getattr(self, "run_dir", None)
        if slug in self._used_slugs or (run_dir and (run_dir / slug).exists()):
            while True:
                count += 1
                slug = _slug_with_suffix(base_slug, count, max_len=max_len)
                if slug not in self._used_slugs and not (run_dir and (run_dir / slug).exists()):
                    break
        self._slug_counts[base_slug] = count
        self._used_slugs.add(slug)
        return slug

    def open_spider(self, spider):
        settings = spider.settings
        base = settings.get("OUTPUT_BASE_DIR") or os.getenv("SCRAPY_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
        run_id = settings.get("RUN_ID")
        if not run_id:
            files_store = settings.get("FILES_STORE")
            if files_store:
                try:
                    fs_path = Path(files_store)
                    candidates = [fs_path.name]
                    if fs_path.parent:
                        candidates.append(fs_path.parent.name)
                    for candidate in candidates:
                        if candidate and candidate.lower() != 'files':
                            run_id = candidate
                            break
                except Exception:
                    pass
        if not run_id:
            run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        try:
            settings.set("RUN_ID", run_id, priority="spider")
        except Exception:
            pass
        self.run_dir = Path(base) / run_id
        self.run_id = run_id
        self._slug_counts: Dict[str, int] = {}
        self._used_slugs: Set[str] = set()
        # Expose on spider for other components (if needed)
        try:
            setattr(spider, "run_id", run_id)
        except Exception:
            pass
        self.run_dir.mkdir(parents=True, exist_ok=True)
        spider.logger.info(f"Writing outputs to: {self.run_dir}")

    def process_item(self, item: Item | Dict[str, Any], spider):
        data: Dict[str, Any] = dict(item)
        url = data.get("final_url") or data.get("url") or ""
        folder_name = self._make_unique_slug(url)
        target_dir = self.run_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Write content
        content_type = (data.get("content_type") or "").lower()
        html = data.get("html")
        text = data.get("text")
        if html:
            (target_dir / "content.html").write_text(html, encoding="utf-8")
        elif text:
            (target_dir / "content.txt").write_text(text, encoding="utf-8")

        # Write structured content elements (text nodes, embeds, etc.)
        content_list = None
        if isinstance(data.get("content"), list):
            content_list = data.get("content")
        elif isinstance(data.get("text_nodes"), list):
            # Backward compatibility
            content_list = data.get("text_nodes")
        if content_list is not None:
            content_doc = {
                "source": data.get("final_url") or data.get("url"),
                "content": content_list,
            }
            with (target_dir / "content.json").open("w", encoding="utf-8") as f:
                json.dump(content_doc, f, ensure_ascii=False, indent=2)

        screenshot_bytes = data.get("screenshot")
        screenshot_rel = None
        if isinstance(screenshot_bytes, (bytes, bytearray)):
            screenshot_path = target_dir / "screenshot.png"
            try:
                screenshot_path.write_bytes(bytes(screenshot_bytes))
                screenshot_rel = "screenshot.png"
            except Exception as exc:
                spider.logger.warning(f"Failed to write screenshot for {url}: {exc}")

        if screenshot_rel:
            try:
                item["screenshot"] = screenshot_rel
            except Exception:
                pass
        elif "screenshot" in data:
            try:
                del item["screenshot"]
            except Exception:
                pass

        # Prepare metadata
        metadata = {
            "url": data.get("url"),
            "final_url": data.get("final_url"),
            "status": data.get("status"),
            "title": data.get("title"),
            "fetched_at": data.get("fetched_at"),
            "content_type": content_type,
            "links": data.get("links", []),
            "note": data.get("note"),
        }
        if screenshot_rel:
            metadata["screenshot"] = screenshot_rel
        extra_meta = data.get("_metadata_extra")
        if isinstance(extra_meta, dict):
            for key, value in extra_meta.items():
                if isinstance(value, list) and isinstance(metadata.get(key), list):
                    metadata[key] = list(dict.fromkeys(metadata[key] + value))
                else:
                    metadata[key] = value

        # Reference downloaded files from FilesPipeline
        if data.get("files"):
            # Prefix run_id so paths are relative to output/files/<run_id>
            prefixed = []
            for f in data["files"]:
                try:
                    d = dict(f)
                    p = d.get("path")
                    if isinstance(p, str):
                        normalized = p.replace('\\', '/')
                        if self.run_id:
                            normalized = f"{self.run_id}/{normalized}"
                        d["path"] = normalized
                    prefixed.append(d)
                except Exception:
                    prefixed.append(f)
            metadata["files"] = prefixed
        elif data.get("file_urls"):
            # If FilesPipeline is disabled, still record intended file URLs
            metadata["file_urls"] = data["file_urls"]

        with (target_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return item


class TikaExtractPipeline:
    """
    If TIKA_SERVER_URL is set, send downloaded files to Tika Server to extract
    text and metadata. Adds the following to the item:
    - 'tika': list of { 'path': str, 'text': str|None, 'metadata': dict|None }
    - If the item has no 'html' and no 'text', but Tika returned text, sets 'text'.
    """

    def open_spider(self, spider):
        self.server_url = os.getenv("TIKA_SERVER_URL")
        self.timeout = float(os.getenv("TIKA_TIMEOUT", "30"))
        # Resolve FILES_STORE to absolute path
        files_store = spider.settings.get("FILES_STORE")
        self.files_store = Path(files_store).resolve() if files_store else None
        if not self.server_url:
            # Default to local Tika if env var not provided
            self.server_url = "http://localhost:9998"
            spider.logger.info(
                "TIKA_SERVER_URL not set. Falling back to local Tika at http://localhost:9998"
            )

    def _ext_from_mime(self, mime: str | None) -> str | None:
        if not mime:
            return None
        # Strip parameters like "; charset=binary"
        mime = mime.split(";", 1)[0].strip().lower()
        # Preferred overrides where mimetypes may be ambiguous or None
        overrides = {
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "text/html": ".html",
            "text/markdown": ".md",
            "text/csv": ".csv",
            "application/json": ".json",
            "application/xml": ".xml",
            "text/xml": ".xml",
            "application/zip": ".zip",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "text/calendar": ".ics",
            "application/postscript": ".ps",
            "application/vnd.ms-office": ".doc",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        if mime in overrides:
            return overrides[mime]
        ext = mimetypes.guess_extension(mime)
        return ext

    def _mime_from_tika_meta(self, meta: Dict[str, Any] | None) -> str | None:
        if not isinstance(meta, dict):
            return None
        # Normalize keys to lowercase for flexible lookups
        lc = {str(k).lower(): v for k, v in meta.items()}
        candidate_keys = [
            "content-type",
            "content_type",
            "dc:format",
            "meta:content-type",
        ]
        for key in candidate_keys:
            val = lc.get(key)
            if isinstance(val, list) and val:
                val = val[0]
            if isinstance(val, str):
                mime = val.split(";", 1)[0].strip()
                if mime:
                    return mime
        return None

    def _mime_from_magic(self, path: Path) -> str | None:
        try:
            with path.open("rb") as f:
                head = f.read(512)
        except Exception:
            return None
        if not head:
            return None
        if head.startswith(b"%PDF-"):
            return "application/pdf"
        if head.startswith(b"%!PS"):
            return "application/postscript"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if head[0:3] == b"GIF" and head[3:6] in (b"87a", b"89a"):
            return "image/gif"
        if head.startswith(b"\xFF\xD8\xFF"):
            return "image/jpeg"
        if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
            return "application/zip"
        if head.startswith(b"\xD0\xCF\x11\xE0"):
            return "application/vnd.ms-office"
        ascii_head = head.decode("ascii", errors="ignore").strip().lower()
        if ascii_head.startswith("<?xml"):
            return "application/xml"
        if ascii_head.startswith("<!doctype html") or "<html" in ascii_head:
            return "text/html"
        if ascii_head.startswith("{") and "}" in ascii_head:
            return "application/json"
        if ascii_head.startswith("[") and "]" in ascii_head:
            return "application/json"
        return None

    def process_item(self, item: Item | Dict[str, Any], spider):
        if not self.server_url or not self.files_store:
            return item

        data: Dict[str, Any] = dict(item)
        files_info = data.get("files") or []
        if not files_info:
            return item

        tika_results = []
        for f in files_info:
            rel = f.get("path")
            if not rel:
                continue
            abs_path = self.files_store / rel
            text = None
            meta = None
            try:
                if abs_path.exists():
                    text, meta = extract_with_tika(abs_path, self.server_url, timeout=self.timeout)
            except TikaError as e:
                spider.logger.warning(f"Tika extraction failed for {abs_path}: {e}")
            # Try to rename file to have a proper extension if missing
            new_rel = rel
            try:
                mime_hint = self._mime_from_tika_meta(meta)
                if not mime_hint:
                    content_type = data.get("content_type")
                    if isinstance(content_type, str):
                        mime_hint = content_type.split(";", 1)[0].strip() or None
                if not mime_hint:
                    mime_hint = self._mime_from_magic(abs_path)
                want_ext = self._ext_from_mime(mime_hint) if mime_hint else None
                suffix = Path(rel).suffix
                normalized_suffix = suffix.lower() if suffix else ""
                normalized_want = want_ext.lower() if want_ext else ""
                needs_rename = False
                if want_ext and (not suffix or normalized_suffix != normalized_want):
                    needs_rename = True
                if needs_rename:
                    target = Path(rel).with_suffix(want_ext)
                    target_abs = self.files_store / target
                    target_abs.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        abs_path.rename(target_abs)
                        new_rel = str(target)
                        f["path"] = new_rel
                        # Also update abs_path for any downstream use
                        abs_path = target_abs
                        spider.logger.info(f"Renamed downloaded file based on MIME {mime_hint}: {rel} -> {new_rel}")
                    except Exception as re:
                        spider.logger.warning(f"Could not rename {abs_path} to {target_abs}: {re}")
            except Exception as e:
                spider.logger.debug(f"Extension inference failed for {rel}: {e}")

            tika_results.append({
                "path": str(new_rel),
                "text": text,
                "metadata": meta,
            })

        if tika_results:
            item["tika"] = tika_results
            # If no page content present, but we have text from Tika, use the first non-empty
            if not item.get("html") and not item.get("text"):
                first_text = next((r["text"] for r in tika_results if r.get("text")), None)
                if first_text:
                    item["text"] = first_text

        return item

