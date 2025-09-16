import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from scrapy import Item

from .tika_client import extract_with_tika, TikaError
import mimetypes

def _slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9\-_.]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-_")
    return text or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


class OutputWriterPipeline:
    """
    Writes, for each item, a directory containing:
    - metadata.json: basic metadata including file references if any
    - content.html (for HTML pages) or content.txt (for plain text)
    Files downloaded by FilesPipeline are referenced via the 'files' field.
    """

    def open_spider(self, spider):
        base = getattr(spider.settings, "OUTPUT_BASE_DIR", None)
        if not base:
            base = os.getenv("SCRAPY_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
        run_id = getattr(spider.settings, "RUN_ID", None) or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        self.run_dir = Path(base) / run_id
        self.run_id = run_id
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
        folder_name = _slugify(url)[:80] or _slugify(spider.name)
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

        # Write structured text nodes (ordered by appearance)
        if data.get("text_nodes") and isinstance(data["text_nodes"], list):
            text_doc = {
                "source": data.get("final_url") or data.get("url"),
                "content": data["text_nodes"],
            }
            with (target_dir / "text_content.json").open("w", encoding="utf-8") as f:
                json.dump(text_doc, f, ensure_ascii=False, indent=2)

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

        # Reference downloaded files from FilesPipeline
        if data.get("files"):
            # Prefix run_id so paths are relative to output/files/<run_id>
            prefixed = []
            for f in data["files"]:
                try:
                    d = dict(f)
                    p = d.get("path")
                    if isinstance(p, str) and self.run_id:
                        d["path"] = f"{self.run_id}/{p}"
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
        }
        if mime in overrides:
            return overrides[mime]
        ext = mimetypes.guess_extension(mime)
        return ext

    def _ext_from_tika_meta(self, meta: Dict[str, Any] | None) -> str | None:
        if not isinstance(meta, dict):
            return None
        # Normalize keys to lowercase
        lc = {str(k).lower(): v for k, v in meta.items()}
        # Prefer filename hints from metadata
        name_keys = [
            "resourcename",  # some Tika outputs (case-insensitive)
            "resource-name",
            "x-tika:origresourcename",
            "orig:resourcename",
            "filename",
            "file-name",
            "meta:filename",
            "file",
            "name",
            "content-disposition",
            "dc:title",
            "pdf:docinfo:title",
        ]
        for k in name_keys:
            v = lc.get(k)
            if isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, str):
                ext = Path(v).suffix
                if ext:
                    return ext
        # Fall back to Content-Type derived extension
        mime = lc.get("content-type") or lc.get("content_type") or lc.get("dc:format")
        if isinstance(mime, list) and mime:
            mime = mime[0]
        if isinstance(mime, str):
            return self._ext_from_mime(mime)
        # As a last resort, scan all metadata values for something that looks like a filename.ext
        try:
            import re
            pattern = re.compile(r"(?:^|[^A-Za-z0-9_\-.])([A-Za-z0-9_\-]+\.(pdf|docx?|xlsx?|pptx?|csv|json|xml|html?|txt|ics))(?=$|[^A-Za-z0-9_\-.])", re.IGNORECASE)
            for val in meta.values():
                if isinstance(val, list):
                    vals = val
                else:
                    vals = [val]
                for s in vals:
                    if not isinstance(s, str):
                        continue
                    m = pattern.search(s)
                    if m:
                        return Path(m.group(1)).suffix
        except Exception:
            pass
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
                # Prefer resourceName-derived extension; then MIME-based
                want_ext = self._ext_from_tika_meta(meta)
                suffix = Path(rel).suffix
                if (not suffix) and want_ext:
                    target = Path(rel).with_suffix(want_ext)
                    target_abs = self.files_store / target
                    target_abs.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        abs_path.rename(target_abs)
                        new_rel = str(target)
                        f["path"] = new_rel
                        # Also update abs_path for any downstream use
                        abs_path = target_abs
                        spider.logger.info(f"Renamed downloaded file to add extension: {rel} -> {new_rel}")
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
