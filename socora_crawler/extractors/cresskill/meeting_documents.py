from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import scrapy
from scrapy import Request

from ..framework import BaseExtractor, ExtractorContext, register_extractor, dedupe_list

_TREE_RE = re.compile(r"new\s+Docman\.Tree\.CategoriesSite\([^,]+,\s*(\{.*?\})\s*\);", re.DOTALL)


@register_extractor("extractors.cresskill.meeting_documents")
class MeetingDocumentsExtractor(BaseExtractor):
    """Extractor that expands meeting document directories and pagination for cresskillboro.com.

    The site relies on Docman trees and JS-driven pagination instead of standard <a> navigation,
    so without this extractor those directories and result pages remain undiscovered. It parses
    the embedded tree once per crawl, queues leaf routes, and inspects pagination controls to
    surface every document listing page."""

    _tree_processed_attr = "_cresskill_doc_tree_processed"

    def __init__(self, *args, **kwargs):
        url_patterns = kwargs.pop("url_patterns", None)
        if not url_patterns:
            url_patterns = [r"^https://(?:www\.)?cresskillboro\.com/index\.php/agenda-library(?:/.*)?$"]
        super().__init__(*args, url_patterns=url_patterns, **kwargs)

    _tree_urls_attr = "_cresskill_doc_tree_urls"
    _pagination_attr = "_cresskill_doc_pages"

    def apply(self, response: scrapy.http.Response, item: Dict[str, Any], context: ExtractorContext) -> None:
        self._maybe_emit_tree_requests(response, context)

    # ------------------------------------------------------------------
    def _maybe_emit_tree_requests(self, response: scrapy.http.Response, context: ExtractorContext) -> None:
        spider = self.spider
        if spider is None or getattr(spider, self._tree_processed_attr, False):
            return

        nodes = self._extract_tree(response)
        if not nodes:
            return

        leaves = self._collect_leaf_routes(nodes)
        if not leaves:
            setattr(spider, self._tree_processed_attr, True)
            return

        seen: set[str] = getattr(spider, self._tree_urls_attr, set())
        new_urls: List[str] = []
        for route in leaves:
            url = response.urljoin(route)
            if url in seen:
                continue
            seen.add(url)
            new_urls.append(url)
            context.requests.append(Request(url, callback=spider.parse_page, cb_kwargs={'depth': 0}))
        if new_urls:
            metadata = context.metadata.setdefault("meeting_document_routes", [])
            metadata.extend(new_urls)
            context.metadata["meeting_document_routes"] = dedupe_list(metadata)
        setattr(spider, self._tree_urls_attr, seen)
        setattr(spider, self._tree_processed_attr, True)

    def _extract_tree(self, response: scrapy.http.Response) -> List[Dict[str, Any]]:
        match = _TREE_RE.search(response.text)
        if not match:
            return []
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        data = parsed.get("data")
        if not isinstance(data, list):
            return []
        return [node for node in data if isinstance(node, dict) and isinstance(node.get("route"), str)]

    def _collect_leaf_routes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        child_counts: Dict[Any, int] = {}
        for node in nodes:
            node_id = node.get("id")
            if node_id is None:
                continue
            child_counts.setdefault(node_id, 0)
        for node in nodes:
            parent = node.get("parent")
            if parent is None:
                continue
            child_counts[parent] = child_counts.get(parent, 0) + 1
        leaves: List[str] = []
        for node in nodes:
            node_id = node.get("id")
            if node_id is None:
                continue
            if child_counts.get(node_id, 0) == 0:
                leaves.append(node["route"])
        return leaves





