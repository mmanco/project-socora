import os

# Scrapy settings for socora_crawler project

BOT_NAME = "socora_crawler"

SPIDER_MODULES = ["socora_crawler.spiders"]
NEWSPIDER_MODULE = "socora_crawler.spiders"

# Required by scrapy-playwright
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Respect robots.txt by default (can be overridden per run)
ROBOTSTXT_OBEY = True

# Concurrency defaults; tune as needed
CONCURRENT_REQUESTS = int(os.getenv("SCRAPY_CONCURRENT_REQUESTS", 8))

# AutoThrottle to be kind by default
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = float(os.getenv("SCRAPY_AUTOTHROTTLE_START_DELAY", 1.0))
AUTOTHROTTLE_MAX_DELAY = float(os.getenv("SCRAPY_AUTOTHROTTLE_MAX_DELAY", 10.0))

# User-Agent
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
    "User-Agent": os.getenv(
        "SCRAPY_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    ),
}

# Output base directory
OUTPUT_BASE_DIR = os.getenv("SCRAPY_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))

# Enable scrapy-playwright for JS-rendered pages
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

PLAYWRIGHT_BROWSER_TYPE = os.getenv("PLAYWRIGHT_BROWSER_TYPE", "chromium")
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() != "false",
}
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = int(os.getenv("PLAYWRIGHT_NAV_TIMEOUT_MS", 30_000))

# Allow Playwright contexts to accept downloads to avoid failures when
# navigating to endpoints that trigger file downloads.
PLAYWRIGHT_CONTEXTS = {
    "default": {
        # Playwright Python expects snake_case option name
        "accept_downloads": True,
    }
}

# Pipeline order: download files first, then write item outputs
ITEM_PIPELINES = {
    "scrapy.pipelines.files.FilesPipeline": 100,
    "socora_crawler.pipelines.TikaExtractPipeline": 800,
    "socora_crawler.pipelines.OutputWriterPipeline": 900,
}

# Where to store downloaded files
FILES_STORE = os.getenv("FILES_STORE", os.path.join(OUTPUT_BASE_DIR, "files"))

# Allow reasonable response sizes by default
DOWNLOAD_MAXSIZE = int(os.getenv("SCRAPY_DOWNLOAD_MAXSIZE", 1024 * 1024 * 64))  # 64MB

# Logging
LOG_LEVEL = os.getenv("SCRAPY_LOG_LEVEL", "INFO")
