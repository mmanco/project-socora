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

Downloaded files are stored under `output/files/` by default (override with `-s FILES_STORE=...`). Metadata includes the `files` array generated by the FilesPipeline.

### Notes and Tips

- JS rendering waits for Playwright load state (default `networkidle`). You can change via `-a render_wait=domcontentloaded|load|networkidle`.
- Under the hood, the spider uses `playwright_page_goto_kwargs` (not page methods) to control the wait behavior; warnings like "expected PageMethod" are resolved with this approach.
- Endpoints that trigger direct downloads (e.g., paths ending with `/file` or `/download`) are detected heuristically and fetched without Playwright to avoid Playwright navigation errors. The browser context also has `acceptDownloads` enabled as a safeguard.
 - If a Playwright navigation unexpectedly starts a download and fails, the spider retries the same URL as a direct file request automatically.
- To be respectful, `ROBOTSTXT_OBEY` is true in settings by default; pass `-s ROBOTSTXT_OBEY=false` for testing.
- You can tune concurrency with `-s CONCURRENT_REQUESTS=16` and AutoThrottle settings.
- The spider checks URL extensions to detect files. If a server serves files without common extensions, you can extend `FILE_EXTENSIONS` in `socora_crawler/spiders/universal.py`.
