from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from scrapy import signals
from twisted.internet.task import LoopingCall


class StallWatchdog:
    """
    Closes the spider if no progress (responses or items) is observed for a
    configurable timeout window. Useful to recover from headless browser stalls
    or hanging network operations that keep the process alive indefinitely.

    Settings (can be set via env or -s):
    - STALL_WATCHDOG_TIMEOUT: seconds of inactivity before closing (default 900)
    - STALL_WATCHDOG_INTERVAL: seconds between checks (default 60)
    """

    def __init__(self, crawler):
        self.crawler = crawler
        s = crawler.settings
        self.timeout = int(s.get("STALL_WATCHDOG_TIMEOUT", 900))
        self.interval = int(s.get("STALL_WATCHDOG_INTERVAL", 60))
        self._lc: Optional[LoopingCall] = None
        self._last_progress_time = datetime.now(timezone.utc)
        self._last_counts = {
            "responses": 0,
            "items": 0,
        }

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls(crawler)
        crawler.signals.connect(ext.engine_started, signal=signals.engine_started)
        crawler.signals.connect(ext.engine_stopped, signal=signals.engine_stopped)
        crawler.signals.connect(ext.response_received, signal=signals.response_received)
        crawler.signals.connect(ext.item_scraped, signal=signals.item_scraped)
        return ext

    def engine_started(self):
        self._update_progress()
        self._lc = LoopingCall(self._check)
        self._lc.start(self.interval, now=False)

    def engine_stopped(self):
        if self._lc and self._lc.running:
            try:
                self._lc.stop()
            except Exception:
                pass

    def response_received(self, response, request, spider):
        self._update_progress()

    def item_scraped(self, item, response, spider):
        self._update_progress()

    def _update_progress(self):
        stats = self.crawler.stats
        resp = stats.get_value("response_received_count", 0) or 0
        items = stats.get_value("item_scraped_count", 0) or 0
        if resp != self._last_counts["responses"] or items != self._last_counts["items"]:
            self._last_counts["responses"] = resp
            self._last_counts["items"] = items
            self._last_progress_time = datetime.now(timezone.utc)

    def _check(self):
        idle_for = (datetime.now(timezone.utc) - self._last_progress_time).total_seconds()
        if idle_for >= self.timeout:
            self.crawler.engine.close_spider(self.crawler.spider, reason=f"stalled_{int(idle_for)}s")

