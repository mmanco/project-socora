import json
import os
import re
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Set, List
from scrapy import Item
from .tika_client import extract_with_tika, TikaError

MAX_SLUG_LEN = 80


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)

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

    def open_spider(self, spider):
        settings = spider.settings
        base = settings.get("OUTPUT_BASE_DIR") or os.getenv("SCRAPY_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
        files_store = settings.get("FILES_STORE")
        run_id = settings.get("RUN_ID")
        if not run_id:
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
        self.files_store_root = Path(files_store).resolve() if files_store else None
        self._slug_counts: Dict[str, int] = {}
        self._used_slugs: Set[str] = set()
        # Expose on spider for other components (if needed)
        try:
            setattr(spider, "run_id", run_id)
        except Exception:
            pass
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config(spider, run_id, base, files_store)
        spider.logger.info(f"Writing outputs to: {self.run_dir}")

    def process_item(self, item: Item | Dict[str, Any], spider):
        data: Dict[str, Any] = dict(item)
        files_info = data.get("files") or []
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
        image_meta_entries: List[Dict[str, str]] = []
        seen_image_sources: Set[str] = set()
        if content_list is not None:
            url_to_path: Dict[str, str] = {}
            for file_info in files_info:
                if not isinstance(file_info, dict):
                    continue
                stored_path = (file_info.get("path") or "").strip()
                if not stored_path:
                    continue
                normalized_path = stored_path.replace("\\", "/")
                url_val = (file_info.get("url") or "").strip()
                if url_val:
                    url_to_path[url_val] = normalized_path
                original_url = (file_info.get("original_url") or "").strip()
                if original_url and original_url not in url_to_path:
                    url_to_path[original_url] = normalized_path
            for block in content_list:
                if not isinstance(block, dict):
                    continue
                meta = block.get("meta")
                if not isinstance(meta, dict) or not meta.get("isImage"):
                    continue
                src = (meta.get("src") or "").strip()
                if not src:
                    continue
                stored_path = url_to_path.get(src)
                rel_path_value = None
                if stored_path:
                    meta.setdefault("downloaded_path", stored_path)
                    rel_candidate = stored_path
                    if getattr(self, "files_store_root", None):
                        try:
                            abs_path = self.files_store_root / stored_path
                            rel_candidate = os.path.relpath(abs_path, target_dir)
                        except Exception:
                            rel_candidate = stored_path
                    rel_path_value = str(rel_candidate).replace("\\", "/")
                    meta["downloaded_rel_path"] = rel_path_value
                else:
                    existing_rel = meta.get("downloaded_rel_path") or meta.get("downloaded_path")
                    if existing_rel:
                        rel_path_value = str(existing_rel).replace("\\", "/")
                        meta["downloaded_rel_path"] = rel_path_value
                if src not in seen_image_sources:
                    entry: Dict[str, str] = {"src": src}
                    link_path = rel_path_value or meta.get("downloaded_rel_path") or meta.get("downloaded_path")
                    if link_path:
                        entry["path"] = str(link_path).replace("\\", "/")
                    alt_val = meta.get("alt")
                    if isinstance(alt_val, str) and alt_val.strip():
                        entry["alt"] = alt_val.strip()
                    image_meta_entries.append(entry)
                    seen_image_sources.add(src)
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
        if image_meta_entries:
            metadata["images"] = image_meta_entries
        extra_meta = data.get("_metadata_extra")
        if isinstance(extra_meta, dict):
            for key, value in extra_meta.items():
                if isinstance(value, list) and isinstance(metadata.get(key), list):
                    metadata[key] = list(dict.fromkeys(metadata[key] + value))
                else:
                    metadata[key] = value

        # Reference downloaded files from FilesPipeline
        if files_info:
            # Prefix run_id so paths are relative to output/files/<run_id>
            prefixed = []
            for f in files_info:
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

    def _collect_run_config(self, spider, run_id: str, base_dir: str, files_store: str | None) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "run_id": run_id,
            "spider": getattr(spider, "name", spider.__class__.__name__),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output": {
                "base_dir": str(base_dir),
                "run_dir": str(self.run_dir),
            },
        }
        if files_store:
            config.setdefault("output", {})["files_store"] = str(files_store)
        start_urls: List[str] = []
        start_arg = getattr(spider, "_start_urls_arg", None)
        if isinstance(start_arg, str):
            start_urls.extend([u.strip() for u in start_arg.split(",") if u.strip()])
        elif isinstance(start_arg, (list, tuple, set)):
            start_urls.extend([str(u).strip() for u in start_arg if str(u).strip()])
        start_attr = getattr(spider, "start_urls", None)
        if isinstance(start_attr, (list, tuple, set)):
            for u in start_attr:
                s = str(u).strip()
                if s:
                    start_urls.append(s)
        if start_urls:
            dedup: List[str] = []
            seen = set()
            for url in start_urls:
                if url not in seen:
                    dedup.append(url)
                    seen.add(url)
            config["start_urls"] = dedup
        url_file = getattr(spider, "_url_file", None)
        if isinstance(url_file, str) and url_file.strip():
            config["url_file"] = url_file.strip()
        allowed_list: List[str] = []
        for src in (getattr(spider, "_allowed", None), getattr(spider, "allowed_domains", None)):
            if isinstance(src, (list, tuple, set)):
                for val in src:
                    s = str(val).strip()
                    if s:
                        allowed_list.append(s)
        if allowed_list:
            config["allowed_domains"] = sorted(dict.fromkeys(allowed_list))
        spider_args: Dict[str, Any] = {}
        follow = getattr(spider, "_follow", None)
        if isinstance(follow, bool):
            spider_args["follow_links"] = follow
        max_depth = getattr(spider, "_max_depth", None)
        if isinstance(max_depth, int):
            spider_args["max_depth"] = max_depth
        render_wait = getattr(spider, "_render_wait", None)
        if isinstance(render_wait, str) and render_wait.strip():
            spider_args["render_wait"] = render_wait.strip()
        extractor_specs = getattr(spider, "_extractor_specs", None)
        if extractor_specs:
            spider_args["extractors"] = _json_safe(extractor_specs)
        if spider_args:
            config["spider_args"] = spider_args
        settings_map: Dict[str, Any] = {}
        try:
            nav_timeout = spider.settings.getint("PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT")
            settings_map["playwright_default_navigation_timeout"] = nav_timeout
        except Exception:
            pass
        if settings_map:
            config["settings"] = settings_map
        output_info = config.get("output")
        if isinstance(output_info, dict):
            config["output"] = {k: v for k, v in output_info.items() if v}
        return {k: _json_safe(v) for k, v in config.items() if v not in (None, [], {}, set())}


    def _write_run_config(self, spider, run_id: str, base_dir: str, files_store: str | None) -> None:
        config = None
        try:
            config = self._collect_run_config(spider, run_id, base_dir, files_store)
        except Exception as exc:
            try:
                spider.logger.debug(f"Failed to collect run config: {exc}")
            except Exception:
                pass
        if not config:
            return
        cfg_path = self.run_dir / "run_config.json"
        try:
            with cfg_path.open("w", encoding="utf-8") as fh:
                json.dump(config, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            try:
                spider.logger.warning(f"Failed to write run_config.json: {exc}")
            except Exception:
                pass


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

