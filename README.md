# Socora.ai

**Technology with society at its core. People-centered, accessible, and built for communities.**

Socora.ai is an open platform for making civic and institutional data accessible, searchable, and useful.  
We focus on transforming the document-heavy, outdated systems of municipalities, schools, and local institutions 
into a modern knowledge base that residents can query directly.

---

## ✨ Mission

Local governments and schools often communicate through fragmented websites and static documents (PDFs, Word files, images).  
Socora.ai makes this information accessible by:

- Crawling and indexing municipal websites, agendas, meeting minutes, and forms.
- Parsing files into structured, searchable data.
- Enabling residents to ask **direct questions** about their town, school district, or community services.
- Providing a unified dashboard for administrators to manage, analyze, and share data.

Our vision extends beyond municipalities: Socora.ai is building the foundation 
for **AI-powered data accessibility and management systems** for towns, schools, and institutions.

## 🌍 Vision

Socora.ai is more than a crawler.  
It is the beginning of a **civic operating system**:  
where residents can access knowledge, administrators can manage with clarity, and AI ensures data is not hidden in PDFs but available to all.

---

## 🛠 Tech Stack

- **Python Scrapy + Playwright** – JS-rendered crawling, file detection/downloads; uv-managed environment.
- **Apache Tika (server)** – MIME detection and robust text/metadata extraction from downloaded files (Docker-compose helper included).
- **Elasticsearch** – Scalable indexing and semantic search across structured and unstructured data.
- **Redis** – Frontier/state scheduling (planned integration for large-scale crawling).
- **AI Agents + RAG** – Retrieval-Augmented Generation layer to answer natural language questions.
- **Chrome Extension (initial client)** – Residents can install and ask direct questions about their community.

---

## 🔄 How It Works

1. **Crawl & Collect**  
   Scrapy + Playwright crawl municipality and school websites, including JS-rendered pages.  
   File links are downloaded; Apache Tika extracts MIME, text, and metadata.

2. **Index & Store**  
   Parsed content is indexed in Elasticsearch with change detection.  
   Redis (planned) will track crawl state and prevent wasteful re-fetching.

3. **Query & Answer**  
   AI agents leverage RAG pipelines to provide precise answers from local data.  
   Example:
   > "When is the next garbage collection in Cresskill?"  
   > "What were the decisions in the last Board of Education meeting?"

4. **Access & Manage**
    - Residents: through the **Chrome extension** or other future clients.
    - Administrators: through the **Dedicated dashboard** for monitoring, indexing health, and insights.

---

## 🚀 Roadmap

- [ ] Core crawler + parsing pipeline (Scrapy + Playwright + Tika).
- [ ] Elasticsearch integration with change-aware indexing.
- [ ] AI RAG agent prototype over indexed data.
- [ ] Chrome extension for end-user Q&A.
- [ ] Dedicated dashboard for monitoring targets and usage.
- [ ] Expand to schools, districts, and community systems.

---

## Scrapy Crawler (JS rendering + file downloads)

This repo includes a Scrapy-based crawler that:

- Renders JavaScript using Playwright to extract post-JS HTML content
- Detects when a URL points to a file and downloads it
- Writes out per-URL folders with `content.html` (or `content.txt`) and `metadata.json`
- Extracts all visible content nodes in DOM order into `content.json` with schema `{ "source": <url>, "content": [ { xpath, content, meta } ] }`. 
  Also captures embedded iframes as items with `meta.isEmbed=true`, `meta.href=<resolved src>`, and `meta.platform` inferred from the embed host (youtube, instagram, pinterest, x, tiktok, other).
- References downloaded files in the metadata using Scrapy's FilesPipeline

### Layout

- `scrapy.cfg`: Scrapy config
- `socora_crawler/`: project package
  - `settings.py`: Scrapy and Playwright settings, FilesPipeline, output paths
  - `pipelines.py`: writes content and metadata for each crawled item
  - `spiders/universal.py`: main spider with JS rendering and file detection
- `pyproject.toml`: Python dependencies and uv configuration

### Install (uv)

Prereqs:

- Python 3.13 (repo has `.python-version`)
- uv installed (see https://docs.astral.sh/uv/)

Install dependencies and browser with uv:

```
uv sync
uv run python -m playwright install chromium
```

### Run

Provide either a comma-separated list of start URLs or a file with one URL per line.

Examples (with uv):

```
# Single page
uv run scrapy crawl universal -a start_urls="https://example.com" -s ROBOTSTXT_OBEY=false

# Multiple pages
uv run scrapy crawl universal -a start_urls="https://example.com,https://example.org/docs" -s ROBOTSTXT_OBEY=false

# From file
uv run scrapy crawl universal -a url_file=seeds.txt -s ROBOTSTXT_OBEY=false

# Restrict following to specific domains, limit depth, and customize output dir
uv run scrapy crawl universal \
  -a start_urls="https://example.com" \
  -a allowed="example.com,static.example.com" \
  -a follow_links=true -a max_depth=2 \
  -s SCRAPY_OUTPUT_DIR=./output -s FILES_STORE=./output/files -s ROBOTSTXT_OBEY=true
```

### Normalize Extracted Text (Markdown)

Convert a page’s `content.json` into clean Markdown using the built‑in normalizer.

Single file:

```
uv run python -m socora_crawler.normalize_text_content output/run-YYYYmmdd-HHMMSS/<page>/content.json > output/run-YYYYmmdd-HHMMSS/<page>/content.md
```

All pages (bash):

```
find output -type f -name content.json -print0 | while IFS= read -r -d '' f; do \
  uv run python -m socora_crawler.normalize_text_content "$f" > "$(dirname "$f")/content.md"; \
done
```

All pages (PowerShell):

```
Get-ChildItem -Recurse -Filter content.json output | ForEach-Object { \
  uv run python -m socora_crawler.normalize_text_content $_.FullName | Out-File -FilePath (Join-Path $_.DirectoryName 'content.md') -Encoding utf8 \
}
```

Content.md format and behavior:

- Front matter: Each `content.md` starts with YAML including `title`, `url`, `run_id`, and `fetched_at` (sourced from `metadata.json`).
- Link rendering: Text nodes with captured hrefs are emitted as Markdown links `[text](url)`; links without visible text are skipped.
- Tables: Data tables are reconstructed as Markdown tables (with header detection), and cell links are clickable.
- Common text filter: Repeated boilerplate across the run is filtered via a per‑run cache.
- Missing text: If a page has only `content.txt` (file extraction), `content.md` contains the front matter and the raw text body.

### Workflow

End-to-end sequence for a typical run:

1) Crawl (with optional Tika)

- Start Tika (optional, improves file MIME/text extraction):
  - Bash: `.containers/tika/run.sh`
  - Windows: `.containers\tika\run.bat`
- Run the crawler (examples):
  - `scripts\run_cresskill.bat` (Windows), or
  - `uv run scrapy crawl universal -a start_urls="https://example.com" -s SCRAPY_OUTPUT_DIR=./output -s FILES_STORE=./output/files -s ROBOTSTXT_OBEY=false`

2) Normalize page text to Markdown

- Single page: `uv run python -m socora_crawler.normalize_text_content output/run-.../<page>/content.json > output/run-.../<page>/content.md`
- Batch (latest run):
  - Bash: `scripts/normalize_run.sh`
  - Windows: `scripts\normalize_run.bat`
- Notes:
  - Creates/uses a cross-page cache at `.output/<run-id>/text_commonalities.json` to suppress repetitive boilerplate.
  - Use `--force-commonalities` to recompute; `--disable-commonalities` to include more plain items; adjust ratio with `--common-threshold` (or env `NORM_COMMON_RATIO`).
  - Utility words: configurable via env `NORM_UTILITY_WORDS="comma,separated,words"` or `.output/<run-id>/normalize_config.json` (keys: `utility_words` to override; `extra_utility_words` to extend). Defaults are minimal (`home`, `search`).

3) Extract and aggregate links

- Batch extract per-page links and aggregate a run-level JSONL:
  - Bash: `scripts/extract_links_run.sh output/run-YYYYmmdd-HHMMSS`
  - Windows: `scripts\extract_links_run.bat output\run-YYYYmmdd-HHMMSS`
- Outputs:
  - Per-page `links.json` next to each `content.json`
  - Run-level `.output/<run-id>/links_index.jsonl`
- Optional per-page link normalization (filters repetitive/common nav):
  - `uv run python -m socora_crawler.normalize_links output/run-.../<page>/links.json --write`
  - Uses `.output/<run-id>/text_commonalities.json` if present or `.output/<run-id>/links_commonalities.json` if computed here; adjust with `--common-threshold` (or env `LINK_COMMON_RATIO`).
  - Notes:
    - For file-backed pages (no `text_content.json`, only `content.txt`), `links.json` is still produced with `{ isFile: true, page_url, page_title, links: [] }` using `metadata.json`.

### Normalization Configuration

- `.output/<run-id>/normalize_config.json` (optional):
  - `utility_words`: array of words treated as utility/nav (replaces defaults)
  - `extra_utility_words`: array to extend defaults
  - Example:
    - `{ "utility_words": ["home", "search", "directory"], "extra_utility_words": ["accessibility"] }`
- Environment variables:
  - `NORM_UTILITY_WORDS`: comma‑separated utility words
  - `NORM_COMMON_RATIO`: default ratio for text commonality filtering (e.g., `0.4`)
  - `LINK_COMMON_RATIO`: default ratio for link commonality filtering

### Run-Level Artifacts

- `output/run-<id>/<page>/`:
  - `content.html`, `metadata.json`, `content.json`, `content.md` (normalized), `links.json` (if extracted)
- `.output/<run-id>/`:
  - `text_commonalities.json` (from text normalization)
  - `links_index.jsonl` (aggregated links across the run)
  - `links_commonalities.json` (from link normalization, if used)

### Indexing (Overview)

Suggested indices for search systems (e.g., Elasticsearch):

- Pages index
  - `url` (keyword): page URL
  - `title` (text): page title
  - `content_md` (text): normalized Markdown (from `content.md`)
  - `run_id` (keyword): run identifier
  - `fetched_at` (date): from `metadata.json`

- Links index
  - `run_id` (keyword)
  - `page_url` (keyword)
  - `page_title` (text)
  - `link_text` (text)
  - `href` (keyword)
  - `heading_path` (keyword, multi-valued): context path of headings
  - `flags` (object): `isNav`, `isAction`, etc.

Query strategy:
- For “Where can I find …” questions, query `link_text` first (boost exact/fuzzy matches), then fall back to `content_md` and `title`. Return `href` with heading context from `heading_path`.

### Optional: Apache Tika Integration

For robust MIME detection and text/metadata extraction from downloaded files, you can run an Apache Tika server and enable the Tika pipeline.

Run Tika server (Docker):

- Using docker-compose helpers in this repo:

```
# Bash (Linux/macOS/WLS)
.containers/tika/run.sh

# Windows
.containers\tika\run.bat
```

- Or via raw Docker:

```
docker run --pull=always --rm -p 9998:9998 apache/tika:2.9.0.0
```

Enable Tika in this project by setting the server URL:

```
# Windows PowerShell example
$env:TIKA_SERVER_URL = "http://localhost:9998"
uv run scrapy crawl universal -a start_urls="https://example.com" -s FILES_STORE=./output/files
```

What it does:

- After FilesPipeline downloads a file, it sends it to Tika for text and metadata.
- Adds `tika` to each item (per-file text + metadata) and, when no page HTML is present, stores extracted text to `text` so `content.txt` is written.
 - If a downloaded file has no extension, the Tika pipeline will rename it to include a suitable extension based on the detected MIME type (e.g., .pdf, .docx, .ics), and update item references accordingly.

Notes:

- If `TIKA_SERVER_URL` is not set, the pipeline defaults to `http://localhost:9998` and will log the fallback.
- You can adjust timeout via `TIKA_TIMEOUT` env var (default 30s).
- To stop the docker-compose Tika server:

```
# Bash
.containers/tika/stop.sh

# Windows
.containers\tika\stop.bat
```

Outputs are written to `output/run-YYYYmmdd-HHMMSS/` with one folder per URL, containing:

- `content.html` (rendered HTML) or `content.txt` for non-HTML text
- `metadata.json` with URL, final URL, status, title, timestamp, content type, links, and any downloaded file references
- `content.json` listing content elements (text nodes and embeds) in order of appearance for the page

Downloaded files are stored under `output/files/<RUN_ID>/` by default (override with `-s FILES_STORE=...`). Metadata includes the `files` array; each `path` is prefixed with `<RUN_ID>/...` so it is relative to `output/files/`.

### Notes and Tips

- JS rendering waits for Playwright load state (default `networkidle`). You can change via `-a render_wait=domcontentloaded|load|networkidle`.
- Under the hood, the spider uses `playwright_page_goto_kwargs` (not page methods) to control the wait behavior; warnings like "expected PageMethod" are resolved with this approach.
- Endpoints that trigger direct downloads (e.g., paths ending with `/file` or `/download`) are detected heuristically and fetched without Playwright to avoid Playwright navigation errors. The browser context also has `acceptDownloads` enabled as a safeguard.
 - If a Playwright navigation unexpectedly starts a download and fails, the spider retries the same URL as a direct file request automatically.
- To be respectful, `ROBOTSTXT_OBEY` is true in settings by default; pass `-s ROBOTSTXT_OBEY=false` for testing.
- You can tune concurrency with `-s CONCURRENT_REQUESTS=16` and AutoThrottle settings.
- The spider checks URL extensions to detect files. If a server serves files without common extensions, you can extend `FILE_EXTENSIONS` in `socora_crawler/spiders/universal.py`.
 - Stability: A stall watchdog closes the spider if no progress (responses/items) is observed for a configurable period (defaults: 15 min). Configure via `STALL_WATCHDOG_TIMEOUT` and `STALL_WATCHDOG_INTERVAL`. Global `DOWNLOAD_TIMEOUT` and Playwright navigation `timeout` are applied per-request to reduce hangs.
