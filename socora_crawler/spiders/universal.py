from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import scrapy
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

    def start_requests(self) -> Iterable[Request]:
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
                        "playwright_page_goto_kwargs": {"wait_until": self._render_wait, "timeout": nav_timeout_ms},
                        "download_timeout": download_timeout_sec,
                    },
                    callback=self.parse_page,
                    cb_kwargs={"depth": 0},
                    errback=self.on_request_error,
                    dont_filter=True,
                )

    def parse_file(self, response: Response, depth: int) -> Optional[dict]:
        # Hand off to FilesPipeline by populating file_urls
        content_type = response.headers.get("Content-Type", b"").decode("latin-1")
        status = response.status
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

    def parse_page(self, response: Response, depth: int):
        # Extract rendered page content
        html = response.text or ""
        title = self._extract_title(html)
        content_type = response.headers.get("Content-Type", b"").decode("latin-1")

        # Gather links
        links = self._extract_links(response)

        # Timeouts for subsequent requests (avoid NameError by computing locally)
        nav_timeout_ms = int(self.settings.getint("PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT", 30000))
        download_timeout_sec = max(10, int(nav_timeout_ms / 1000) + 5)

        # Extract visible text nodes in DOM order (excluding scripts/styles and similar)
        text_nodes = self._extract_text_nodes(response)

        item = {
            "url": response.request.url,
            "final_url": response.url,
            "status": response.status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type or "text/html",
            "title": title,
            "html": html,
            "links": links,
            "text_nodes": text_nodes,
        }
        yield item

        # Follow links if enabled and within depth
        if self._follow and depth < self._max_depth:
            for link in links:
                if self._allowed and not self._is_allowed_domain(link):
                    continue
                if looks_like_file(link):
                    yield scrapy.Request(
                        link,
                        callback=self.parse_file,
                        cb_kwargs={"depth": depth + 1},
                        meta={"download_timeout": download_timeout_sec},
                    )
                else:
                    yield scrapy.Request(
                        link,
                        meta={
                            "playwright": True,
                            "playwright_context": "default",
                            "playwright_page_goto_kwargs": {"wait_until": self._render_wait, "timeout": nav_timeout_ms},
                            "download_timeout": download_timeout_sec,
                        },
                        callback=self.parse_page,
                        cb_kwargs={"depth": depth + 1},
                        errback=self.on_request_error,
                    )

    def on_request_error(self, failure: Failure):
        request = failure.request
        url = getattr(request, 'url', None) or (request and request.url)
        depth = 0
        try:
            depth = int((request.cb_kwargs or {}).get('depth', 0))
        except Exception:
            pass
        msg = str(failure.value)
        # If Playwright navigation triggers a download, retry as a direct file request
        if 'Download is starting' in msg or 'download is starting' in msg:
            self.logger.info(f"Retrying as file due to download start: {url}")
            return scrapy.Request(url, callback=self.parse_file, cb_kwargs={'depth': depth}, dont_filter=True)
        # Propagate other errors by logging; returning None drops the request
        self.logger.warning(f"Playwright request failed: {msg} for {url}")

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
