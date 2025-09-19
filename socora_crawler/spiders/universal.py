from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import asyncio
import scrapy

from socora_crawler.extractors.framework import ExtractorContext, build_extractors
from scrapy.http import Response, Request
from urllib.parse import urlparse, parse_qs
from twisted.python.failure import Failure
from lxml import html as LH


FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp3", ".wav", ".mp4", ".mov", ".avi",
    ".csv", ".json", ".xml", ".txt",
}


def looks_like_file(url: str) -> bool:
    # Heuristics by extension, path tokens and query params commonly used for downloads
    lower_base = url.split("?", 1)[0].lower()
    for ext in FILE_EXTENSIONS:
        if lower_base.endswith(ext):
            return True

    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith("/file") or path.endswith("/download") or \
       "/download/" in path or "/attachment" in path:
        return True

    q = parse_qs(parsed.query)
    # any query parameter named like download/attachment or with value containing 'download'
    if any(k in q for k in ("download", "attachment", "dl")):
        return True
    if any(any("download" in v.lower() for v in vals) for vals in q.values()):
        return True
    # cms-specific patterns like task=*.download
    for key in ("task", "action", "cmd"):
        vals = [v.lower() for v in q.get(key, [])]
        if any("download" in v for v in vals):
            return True
    # E.g., format=pdf or type=pdf
    for key in ("format", "type", "ext"):
        vals = [v.lower() for v in q.get(key, [])]
        if any(v in ("pdf", "doc", "docx", "xls", "xlsx") for v in vals):
            return True
    return False


class UniversalSpider(scrapy.Spider):
    name = "universal"

    custom_settings = {
        # Make sure file pipeline has a place to store outputs
        # Users can override via -s FILES_STORE=...
    }

    def __init__(
        self,
        start_urls: Optional[str] = None,
        url_file: Optional[str] = None,
        allowed: Optional[str] = None,
        follow_links: str = "true",
        max_depth: int = 2,
        render_wait: str = "networkidle",
        extractors: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._start_urls_arg = start_urls
        self._url_file = url_file
        self._allowed = [d.strip() for d in allowed.split(",")] if allowed else None
        self._follow = follow_links.lower() != "false"
        self._max_depth = int(max_depth)
        self._render_wait = render_wait
        self._extractor_specs: List[Any] = self._parse_extractor_specs(extractors)
        self._extractors: List[Any] = []
        self._max_playwright_retries = 4

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider._apply_settings(crawler.settings)
        spider._init_extractors(crawler)

        return spider

    async def start(self) -> AsyncIterator[Request]:
        for req in self._iter_start_requests():
            yield req

    async def parse_page(self, response: Response, depth: int):
        page = response.meta.get("playwright_page")
        screenshot_bytes = None
        if page is not None:
            try:
                screenshot_bytes = await page.screenshot(full_page=True, type="png")
            except Exception as exc:
                self.logger.warning(f"Screenshot capture failed for {response.url}: {exc}")
            finally:
                try:
                    await page.close()
                except Exception:
                    pass

        html = response.text or ""
        title = self._extract_title(html)
        content_type = response.headers.get("Content-Type", b"").decode("latin-1")

        links = self._extract_links(response)

        text_nodes = self._extract_text_nodes(response)
        embeds = self._extract_iframes(response)

        content_blocks = text_nodes + embeds

        return self._build_page_results(
            response=response,
            depth=depth,
            html=html,
            content_type=content_type or "text/html",
            title=title,
            links=links,
            content_blocks=content_blocks,
            screenshot_bytes=screenshot_bytes,
        )

    def parse_file(self, response: Response, depth: int) -> Any:
        content_type = response.headers.get("Content-Type", b"").decode("latin-1")
        status = response.status

        if self._should_treat_response_as_html(content_type, response.body):
            html = response.text or ""
            if html.strip():
                title = self._extract_title(html)
                links = self._extract_links(response)
                text_nodes = self._extract_text_nodes(response)
                embeds = self._extract_iframes(response)
                self.logger.debug(f"Treating file-like response as HTML page: {response.url}")
                normalized_content_type = content_type if "html" in (content_type or "").lower() else "text/html"
                return self._build_page_results(
                    response=response,
                    depth=depth,
                    html=html,
                    content_type=normalized_content_type,
                    title=title,
                    links=links,
                    content_blocks=text_nodes + embeds,
                    screenshot_bytes=None,
                )

        item = {
            "url": response.request.url,
            "final_url": response.url,
            "status": status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "file_urls": [response.url],
            "note": "Direct file URL"
        }
        return item
    
    def on_request_error(self, failure: Failure):
        request = failure.request
        url = getattr(request, 'url', None) or (request and request.url)
        depth = 0
        try:
            depth = int((request.cb_kwargs or {}).get('depth', 0))
        except Exception:
            pass

        page = request.meta.pop("playwright_page", None) if request.meta else None
        if page is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None
            if loop is not None:
                try:
                    loop.create_task(page.close())
                except Exception:
                    pass

        msg = str(failure.value)
        # If Playwright navigation triggers a download, retry as a direct file request
        if 'Download is starting' in msg or 'download is starting' in msg:
            self.logger.info(f"Retrying as file due to download start: {url}")
            return scrapy.Request(url, callback=self.parse_file, cb_kwargs={'depth': depth}, dont_filter=True)

        if request.meta and request.meta.get('playwright'):
            retry_times = int(request.meta.get('playwright_retry_times', 0))
            max_retries = int(request.meta.get('playwright_max_retries', self._max_playwright_retries))
            if retry_times < max_retries:
                new_meta = dict(request.meta)
                new_meta.pop('playwright_page', None)
                new_meta['playwright_retry_times'] = retry_times + 1
                new_meta.setdefault('playwright_max_retries', max_retries)
                self.logger.warning(
                    f"Playwright request failed (attempt {retry_times + 1}/{max_retries}) for {url}: {msg}. Retrying..."
                )
                return request.replace(meta=new_meta, dont_filter=True)
            else:
                self.logger.error(
                    f"Playwright request exhausted {retry_times} retries for {url}. Last error: {msg}"
                )
        else:
            self.logger.warning(f"Playwright request failed: {msg} for {url}")    

    def _apply_settings(self, settings):
        self._max_playwright_retries = int(settings.getint("RETRY_TIMES", 4))
        self._extractor_specs.extend(self._parse_extractor_specs(settings.get("EXTRACTORS")))
        self._extractor_specs.extend(self._parse_extractor_specs(settings.get("SUPPLEMENTAL_EXTRACTORS")))

    def _ensure_extractor_modules(self, specs: List[Any]) -> List[Any]:
        out: List[Any] = []
        for entry in specs:
            if isinstance(entry, str) and ':' not in entry and '.' in entry:
                try:
                    import importlib
                    importlib.import_module(entry)
                except Exception as exc:
                    self.logger.warning("Could not import extractor module %s: %s", entry, exc)
                    out.append(entry)
                    continue
            out.append(entry)
        return out
    def _parse_extractor_specs(self, value) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [s.strip() for s in value.split(',') if s.strip()]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple, set)):
            specs: List[Any] = []
            for entry in value:
                specs.extend(self._parse_extractor_specs(entry))
            return specs
        return [value]

    def _init_extractors(self, crawler):
        specs = self._extractor_specs or []
        if specs:
            specs = self._ensure_extractor_modules(specs)
            unique: List[Any] = []
            for entry in specs:
                if entry not in unique:
                    unique.append(entry)
            specs = unique
        if not specs:
            self._extractors = []
            return
        instances = build_extractors(specs)
        bound: List[Any] = []
        for extractor in instances:
            try:
                extractor.bind(self)
            except Exception as exc:
                self.logger.warning(
                    "Failed to bind extractor %s: %s",
                    getattr(extractor, 'name', extractor.__class__.__name__),
                    exc,
                )
                continue
            bound.append(extractor)
        self._extractors = bound
        if bound:
            names = ", ".join(getattr(e, 'name', e.__class__.__name__) for e in bound)
            self.logger.info("Loaded extractors: %s", names)

    def _run_extractors(
        self,
        response: Response,
        item: Dict[str, Any],
        context: ExtractorContext,
    ) -> None:
        if not self._extractors:
            return

        for extractor in self._extractors:
            name = getattr(extractor, 'name', extractor.__class__.__name__)
            try:
                if extractor.matches(response, item, context):
                    extractor.apply(response, item, context)
            except Exception as exc:
                self.logger.warning("Extractor %s failed for %s: %s", name, response.url, exc)
        if getattr(context, "item_updates", None):
            try:
                item.update(context.item_updates)
            except Exception:
                pass
        if context.metadata:
            extra = item.setdefault("_metadata_extra", {})
            if not isinstance(extra, dict):
                extra = {}
                item["_metadata_extra"] = extra
            for key, value in context.metadata.items():
                if isinstance(value, list):
                    existing = extra.get(key)
                    if isinstance(existing, list):
                        combined = list(dict.fromkeys(existing + value))
                    else:
                        combined = list(dict.fromkeys(value))
                    extra[key] = combined
                else:
                    extra[key] = value

    def _iter_start_requests(self) -> Iterable[Request]:
        urls: List[str] = []
        if self._start_urls_arg:
            urls.extend([u.strip() for u in self._start_urls_arg.split(",") if u.strip()])
        if self._url_file:
            with open(self._url_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        urls.append(s)
        if not urls:
            raise scrapy.exceptions.CloseSpider("No start URLs provided. Use -a start_urls=... or -a url_file=...")

        # Auto-include seed domains into allowed list (if user provided one)
        seed_domains = set()
        for u in urls:
            try:
                host = urlparse(u).netloc.lower()
                if host:
                    seed_domains.add(host)
                    if host.startswith("www."):
                        seed_domains.add(host[4:])
            except Exception:
                pass
        if self._allowed is not None:
            # Merge without duplicates
            merged = set(d.lower() for d in self._allowed)
            merged.update(seed_domains)
            self._allowed = sorted(merged)

        nav_timeout_ms = int(self.settings.getint("PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT", 30000))
        download_timeout_sec = max(10, int(nav_timeout_ms / 1000) + 5)

        for u in urls:
            if looks_like_file(u):
                yield scrapy.Request(
                    u,
                    callback=self.parse_file,
                    cb_kwargs={"depth": 0},
                    dont_filter=True,
                    meta={"download_timeout": download_timeout_sec},
                )
            else:
                yield scrapy.Request(
                    u,
                    meta={
                        "playwright": True,
                        "playwright_context": "default",
                        "playwright_include_page": True,
                        "playwright_page_goto_kwargs": {"wait_until": self._render_wait, "timeout": nav_timeout_ms},
                        "download_timeout": download_timeout_sec,
                        "playwright_retry_times": 0,
                        "playwright_max_retries": self._max_playwright_retries,
                    },
                    callback=self.parse_page,
                    cb_kwargs={"depth": 0},
                    errback=self.on_request_error,
                    dont_filter=True,
                )

    def _should_treat_response_as_html(self, content_type: str, body: bytes) -> bool:
        ctype = (content_type or "").lower()
        if "text/html" in ctype or "application/xhtml" in ctype:
            return True
        if not body:
            return False
        sample = body[:4096].lstrip()
        if not sample:
            return False
        lowered = sample.lower()
        if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
            return True
        if lowered.startswith(b"<?xml"):
            parts = lowered.split(b">", 1)
            if len(parts) == 2:
                tail = parts[1].lstrip().lower()
                if tail.startswith(b"<!doctype html") or tail.startswith(b"<html"):
                    return True
        if b"<html" in lowered:
            return True
        if lowered.startswith(b"<head") or lowered.startswith(b"<body"):
            return True
        return False 

    def _build_page_results(
        self,
        response: Response,
        depth: int,
        html: str,
        content_type: str,
        title: Optional[str],
        links: List[str],
        content_blocks: List[Any],
        screenshot_bytes: Optional[bytes] = None,
    ) -> List[Any]:
        results: List[Any] = []
        links_list = list(links or [])
        content_list = list(content_blocks or [])
        item = {
            "url": response.request.url,
            "final_url": response.url,
            "status": response.status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type or "text/html",
            "title": title,
            "html": html,
            "links": links_list,
        }
        context = ExtractorContext(
            content=content_list,
            links=links_list,
            item=item,
            metadata={},
            requests=[],
            item_updates={},
        )
        self._run_extractors(response, item, context)
        content_list = list(context.content or [])
        item["content"] = content_list
        links_list = list(dict.fromkeys(context.links))
        item["links"] = links_list
        if screenshot_bytes:
            item["screenshot"] = screenshot_bytes
        results.append(item)

        nav_timeout_ms = int(self.settings.getint("PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT", 30000))
        download_timeout_sec = max(10, int(nav_timeout_ms / 1000) + 5)

        if self._follow and depth < self._max_depth:
            for link in links_list:
                if self._allowed and not self._is_allowed_domain(link):
                    continue
                if looks_like_file(link):
                    results.append(
                        scrapy.Request(
                            link,
                            callback=self.parse_file,
                            cb_kwargs={"depth": depth + 1},
                            meta={"download_timeout": download_timeout_sec},
                        )
                    )
                else:
                    results.append(
                        scrapy.Request(
                            link,
                            meta={
                                "playwright": True,
                                "playwright_context": "default",
                                "playwright_include_page": True,
                                "playwright_page_goto_kwargs": {"wait_until": self._render_wait, "timeout": nav_timeout_ms},
                                "download_timeout": download_timeout_sec,
                                "playwright_retry_times": 0,
                                "playwright_max_retries": self._max_playwright_retries,
                            },
                            callback=self.parse_page,
                            cb_kwargs={"depth": depth + 1},
                            errback=self.on_request_error,
                        )
                    )

        if context.requests:
            results.extend(context.requests)

        return results

    @staticmethod
    def _extract_title(html: str) -> Optional[str]:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        return title or None

    @staticmethod
    def _extract_links(response: Response) -> List[str]:
        hrefs = response.css("a::attr(href)").getall()
        out: List[str] = []
        for h in hrefs:
            if not h:
                continue
            s = h.strip()
            if not s:
                continue
            low = s.lower()
            # Skip non-navigational or unsupported schemes
            if low.startswith((
                "javascript:", "mailto:", "tel:", "sms:", "callto:",
                "about:", "data:", "blob:", "#",
            )):
                continue
            absu = response.urljoin(s).strip()
            if absu.startswith("http://") or absu.startswith("https://"):
                out.append(absu)
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for u in out:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        return uniq

    @staticmethod
    def _extract_text_nodes(response: Response) -> List[dict]:
        # Use lxml to ensure stable node identities and accurate xpaths
        try:
            doc = LH.fromstring(response.text)
        except Exception:
            return []

        bodies = doc.xpath("//body")
        if not bodies:
            return []
        body = bodies[0]

        # Collect text nodes under body, ignoring those inside script/style/etc.
        text_nodes = body.xpath(
            ".//text()[normalize-space() and not(ancestor::script or ancestor::style or ancestor::noscript or "
            "ancestor::svg or ancestor::canvas or ancestor::template or ancestor::code or ancestor::pre)]"
        )

        def looks_like_css(text: str) -> bool:
            t = text.strip()
            if not t:
                return False
            if t.startswith("@media") or "@font-face" in t:
                return True
            if ("{" in t and "}" in t) and re.search(r"[.#][A-Za-z]", t):
                return True
            if re.match(r"^\s*[#.][A-Za-z0-9_-]+\s*\{", t):
                return True
            return False

        out: List[dict] = []
        tree = doc.getroottree()

        def has_class(el, name: str) -> bool:
            cls = el.get("class") or ""
            return f" {name} " in f" {cls} "

        def attr_lower(el, key: str) -> str:
            v = el.get(key)
            return v.lower() if isinstance(v, str) else ""

        for node in text_nodes:
            try:
                s = re.sub(r"\s+", " ", str(node)).strip()
                if not s:
                    continue
                if looks_like_css(s):
                    continue

                parent = node.getparent()
                if parent is None:
                    continue
                parent_path = tree.getpath(parent)
                siblings = parent.xpath("text()")
                pos = 1
                for i, tnode in enumerate(siblings, start=1):
                    if tnode is node:
                        pos = i
                        break
                xpath_str = f"{parent_path}/text()[{pos}]"

                # Ascend ancestors for meta flags
                def has_ancestor(pred) -> bool:
                    el = parent
                    while el is not None:
                        try:
                            if pred(el):
                                return True
                        except Exception:
                            pass
                        el = el.getparent()
                    return False

                is_nav = has_ancestor(lambda el: el.tag.lower() == "nav" or attr_lower(el, "role") == "navigation" or el.tag.lower() == "a")
                is_title = has_ancestor(lambda el: el.tag.lower() in {"h1","h2","h3","h4","h5","h6"} or attr_lower(el, "role") == "heading")
                is_par = has_ancestor(lambda el: el.tag.lower() in {"p","li"})
                # If it's navigation (e.g., anchor text), don't mark as paragraph
                if is_nav:
                    is_par = False
                is_action = has_ancestor(
                    lambda el: el.tag.lower() == "button" or attr_lower(el, "role") == "button" or (
                        el.tag.lower() == "input" and attr_lower(el, "type") in {"button","submit","reset"}
                    ) or has_class(el, "btn")
                )

                meta = {
                    "isNav": is_nav,
                    "isTitle": is_title,
                    "isParagraph": is_par,
                    "isAction": is_action,
                }

                # If navigation text, capture nearest anchor href as absolute URL
                if is_nav:
                    try:
                        el = parent
                        href_val = None
                        while el is not None and href_val is None:
                            if el.tag.lower() == "a":
                                href_val = el.get("href")
                                break
                            el = el.getparent()
                        if href_val:
                            href_str = str(href_val).strip()
                            low = href_str.lower()
                            if not (low.startswith("javascript:") or low.startswith("mailto:") or low.startswith("tel:") or low.startswith("sms:") or low.startswith("callto:") or href_str.startswith("#")):
                                meta["href"] = response.urljoin(href_str)
                    except Exception:
                        pass

                out.append({
                    "xpath": xpath_str,
                    "content": s,
                    "meta": meta,
                })
            except Exception:
                continue
        return out

    def _is_allowed_domain(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
            if not host:
                return False
            for d in self._allowed or []:
                d = d.lower()
                if host == d or host.endswith("." + d):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _extract_iframes(response: Response) -> List[dict]:
        out: List[dict] = []
        try:
            sels = response.xpath("//iframe[@src]")
            for sel in sels:
                try:
                    src = sel.xpath("@src").get() or ""
                    src = src.strip()
                    if not src:
                        continue
                    href = response.urljoin(src)
                    # Detect platform by hostname
                    platform = UniversalSpider._detect_embed_platform(href)
                    # Compute element xpath
                    xpath_str = None
                    try:
                        el = sel.root
                        xpath_str = el.getroottree().getpath(el)
                    except Exception:
                        xpath_str = "//iframe"
                    out.append({
                        "xpath": xpath_str,
                        "content": href,
                        "meta": {
                            "isEmbed": True,
                            "href": href,
                            "platform": platform,
                        },
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return out

    @staticmethod
    def _detect_embed_platform(href: str) -> str:
        try:
            host = urlparse(href).netloc.lower()
        except Exception:
            return "other"
        if any(h in host for h in ("youtube.com", "youtu.be")):
            return "youtube"
        if "instagram.com" in host:
            return "instagram"
        if "pinterest." in host:
            return "pinterest"
        if "tiktok.com" in host:
            return "tiktok"
        if "twitter.com" in host or host.endswith(".x.com") or host == "x.com":
            return "x"
        return "other"




