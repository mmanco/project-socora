# Socora Crawler

The Socora crawler is a Scrapy + Playwright project that captures civic websites, downloads linked documents, and prepares structured artifacts for downstream indexing.

## Project Layout
- `scrapy.cfg` — Scrapy entry point used by `scrapy` CLI.
- `socora_crawler/` — project package (settings, pipelines, spiders, helpers).
- `pyproject.toml` / `uv.lock` — dependency definitions managed by uv.
- `.venv/` — optional local virtual environment created by uv (ignored by Git).
- `../output/` — default parent directory where crawl runs and derived files are written.

## Prerequisites
- Python 3.13 (see `.python-version`).
- [uv](https://docs.astral.sh/uv/) for package and environment management.
- Playwright browsers (installed via the setup steps below).

## Setup
```powershell
uv sync
uv run python -m playwright install chromium
```

The first command installs Python dependencies into `.venv`. The second downloads the Chromium bundle used for JavaScript rendering.

## Running Crawls
Use the uv-powered Scrapy entry point from the `socora-crawler` directory:
```powershell
uv run scrapy crawl universal -a start_urls="https://example.com" -s ROBOTSTXT_OBEY=false
```
Common arguments:
- `start_urls` — comma-separated seed URLs.
- `url_file` — path to a file with one URL per line (alternative to `start_urls`).
- `allowed` — restricts link following to listed domains.
- `max_depth` — limits crawl depth.
- `SCRAPY_OUTPUT_DIR` — override the default `../output` target directory.

The crawler writes one folder per page under `../output/run-<timestamp>/`, including `content.html`, `metadata.json`, `content.json`, optional `content.md`, and any downloaded files.

## Post-Processing Helpers
Utility scripts live under `scripts/crawler` at the repository root:
- `extract_links_run.(bat|sh)` — build per-page link files and aggregate them into `links_index.jsonl`.
- `normalize_run.(bat|sh)` — generate Markdown summaries (`content.md`) and optional commonality reports.
- `find_empty_content_md.(bat|sh)` — detect pages that lack normalized content.

Invoke them from the repo root; they will locate the latest run automatically or accept an explicit path, for example:
```powershell
scripts\crawler\normalize_run.bat output\run-20250921-120000 --force-commonalities
```

## Apache Tika (Optional)
Enable richer MIME detection and text extraction for downloaded files by running an Apache Tika server.
1. Start a local Tika container:
   ```powershell
   .containers\tika\run.bat
   ```
2. Set the server URL when launching the crawl:
   ```powershell
   set TIKA_SERVER_URL=http://localhost:9998
   uv run scrapy crawl universal -a start_urls="https://example.com"
   ```

When enabled, the pipeline augments each item with Tika metadata and extracted text, ensuring non-HTML files still produce content artifacts.

## Tips
- Adjust concurrency via `-s CONCURRENT_REQUESTS=16` and Playwright wait behaviour via `-a render_wait=networkidle|load|domcontentloaded`.
- The spider retries direct file downloads when Playwright navigation triggers a download-only response.
- Respect robots.txt in production; pass `-s ROBOTSTXT_OBEY=false` only for testing.