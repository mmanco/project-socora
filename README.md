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


# Socora Platform

Socora is a modular platform for collecting, organizing, and activating civic knowledge. The codebase now hosts multiple services that work together to ingest public information, normalize it, and make it queryable for residents and administrators.

## Modules
- **socora-crawler** � Scrapy + Playwright pipeline that fetches civic websites, captures HTML, downloads linked files, and normalizes content for downstream use.
- **socora-indexer** � LlamaIndex-based workers that transform crawler artifacts into retrieval-friendly indexes and RAG-ready knowledge stores.
- _Coming soon_ � additional orchestration and delivery components that build on these shared outputs.

## Repository Layout
- `socora-crawler/` � crawler package, uv project, and Playwright helpers.
- `socora-indexer/` � indexer package and experiments with LlamaIndex.
- `scripts/` � cross-module automation; the `crawler/` folder targets crawler runs.
- `output/` � default location for crawler run artifacts (shared by the indexer during prototyping).
- `.containers/` � helper scripts for local infrastructure such as Apache Tika.

## Getting Started
1. Install Python 3.13 (or the version pinned by `.python-version`) and [uv](https://docs.astral.sh/uv/).
2. Bootstrap each module independently; see their README files for environment sync, Playwright browser installation, and module-specific workflows.
3. Use the scripts in `scripts/crawler` for common post-processing tasks like link extraction and content normalization.

## Running The Modules
- **Crawler**: instructions, arguments, and troubleshooting live in `socora-crawler/README.md`.
- **Indexer**: setup notes, data contracts, and sample pipelines live in `socora-indexer/README.md`.

## Roadmap Themes
- Expand crawling coverage to additional municipalities, districts, and agencies.
- Harden indexing pipelines with scheduled builds, validation, and semantic retrieval evaluation.
- Deliver end-user experiences (dashboards, assistants, extensions) powered by the shared knowledge graph.
- Capture operational telemetry across modules for observability and governance.

## Contributing
PRs and issues are welcome. Please keep documentation in sync when moving modules or changing shared contracts between components.