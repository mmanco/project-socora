from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

import types

import yaml
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser.text import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from tqdm.auto import tqdm
from socora_indexer.opensearch_utils import ensure_opensearch_index

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("opensearch").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _close_opensearch_client(vector_client: OpensearchVectorClient) -> None:
    client = getattr(vector_client, "_os_client", None)
    if client is not None:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close OpenSearch client", exc_info=True)
        else:
            transport = getattr(client, "transport", None)
            close = getattr(transport, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close OpenSearch transport", exc_info=True)
    async_client = getattr(vector_client, "_os_async_client", None)
    if async_client is not None:
        transport = getattr(async_client, "transport", None)
        if transport is not None:
            close = getattr(transport, "close", None)
            if callable(close):
                try:
                    maybe = close()
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:
                    logger.debug("Failed to close async OpenSearch transport", exc_info=True)
        close = getattr(async_client, "close", None)
        if callable(close):
            try:
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:
                logger.debug("Failed to close async OpenSearch client", exc_info=True)


def _apply_embedding_prefix(
    embed_model: OllamaEmbedding,
    *,
    document_prefix: str,
    query_prefix: str,
) -> OllamaEmbedding:
    """Ensure inputs to the embedding model include the required task prefixes."""

    def _wrap(method, prefix: str):
        original = method.__func__

        def _prefixed(self, value: str):
            formatted = original(self, value)
            formatted = formatted.strip()
            if formatted.startswith(prefix):
                return formatted
            return f"{prefix}{formatted}"

        return types.MethodType(_prefixed, embed_model)

    if document_prefix:
        embed_model._format_text = _wrap(embed_model._format_text, document_prefix)
    if query_prefix:
        embed_model._format_query = _wrap(embed_model._format_query, query_prefix)
    return embed_model


def build_content_index(
    run_dir: str | Path,
    *,
    opensearch_endpoint: str,
    opensearch_index: str,
    opensearch_username: Optional[str] = None,
    opensearch_password: Optional[str] = None,
    opensearch_client_kwargs: Optional[dict[str, Any]] = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text",
    embed_batch_size: int = 32,
    semantic_buffer_size: int = 2,
    semantic_breakpoint_percentile: int = 92,
    embedding_dimension: int = 768,
    additional_metadata: Optional[dict[str, Any]] = None,
    skip_missing_content: bool = True,
    pipeline_workers: Optional[int] = None,
) -> VectorStoreIndex:
    """Build a content index for a crawler run and push it into OpenSearch."""

    kwargs = dict(
        opensearch_endpoint=opensearch_endpoint,
        opensearch_index=opensearch_index,
        opensearch_username=opensearch_username,
        opensearch_password=opensearch_password,
        opensearch_client_kwargs=opensearch_client_kwargs,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        embed_batch_size=embed_batch_size,
        semantic_buffer_size=semantic_buffer_size,
        semantic_breakpoint_percentile=semantic_breakpoint_percentile,
        embedding_dimension=embedding_dimension,
        additional_metadata=additional_metadata,
        skip_missing_content=skip_missing_content,
        pipeline_workers=pipeline_workers,
    )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build_content_index_async(run_dir, **kwargs))

    raise RuntimeError(
        "build_content_index cannot be invoked from an active event loop. "
        "Use 'await build_content_index_async(...)' instead."
    )


async def build_content_index_async(
    run_dir: str | Path,
    *,
    opensearch_endpoint: str,
    opensearch_index: str,
    opensearch_username: Optional[str] = None,
    opensearch_password: Optional[str] = None,
    opensearch_client_kwargs: Optional[dict[str, Any]] = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text",
    embed_batch_size: int = 32,
    semantic_buffer_size: int = 2,
    semantic_breakpoint_percentile: int = 92,
    embedding_dimension: int = 768,
    additional_metadata: Optional[dict[str, Any]] = None,
    skip_missing_content: bool = True,
    pipeline_workers: Optional[int] = None,
) -> VectorStoreIndex:
    """Async variant of build_content_index using a streaming backpressure pipeline."""

    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    total_content_files = sum(1 for _ in _iter_content_markdown(run_path))
    if total_content_files == 0:
        raise ValueError(f"No content.md files found in run directory: {run_path}")

    embed_model = OllamaEmbedding(
        model_name=ollama_model,
        base_url=ollama_base_url,
        embed_batch_size=embed_batch_size,
    )
    embed_model = _apply_embedding_prefix(
        embed_model,
        document_prefix='search_document: ',
        query_prefix='',
    )

    node_parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=semantic_buffer_size,
        breakpoint_percentile_threshold=semantic_breakpoint_percentile,
    )

    client_kwargs = dict(opensearch_client_kwargs or {})
    if opensearch_username or opensearch_password:
        client_kwargs.setdefault(
            "http_auth",
            (
                opensearch_username or "",
                opensearch_password or "",
            ),
        )

    os_client = ensure_opensearch_index(
        endpoint=opensearch_endpoint,
        index_name=opensearch_index,
        dimension=embedding_dimension,
        client_kwargs=client_kwargs,
    )

    vector_client = OpensearchVectorClient(
        endpoint=opensearch_endpoint,
        index=opensearch_index,
        dim=embedding_dimension,
        os_client=os_client,
        **client_kwargs,
    )
    vector_store = OpensearchVectorStore(vector_client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )

    chunk_workers = pipeline_workers if pipeline_workers is not None else 1
    if chunk_workers <= 0:
        raise ValueError("pipeline_workers must be a positive integer")

    progress = tqdm(
        total=total_content_files,
        unit="page",
        desc="Indexing pages",
    )

    async def load_document(content_path: Path) -> Optional[Document]:
        try:
            return await asyncio.to_thread(
                _document_from_markdown,
                content_path,
                run_path=run_path,
                additional_metadata=additional_metadata,
            )
        except ValueError as exc:
            progress.update(1)
            if skip_missing_content:
                logger.warning("Skipping %s: %s", content_path, exc)
                return None
            raise

    pipeline = _BackpressureIndexingPipeline(
        node_parser=node_parser,
        index=index,
        progress=progress,
        chunk_workers=chunk_workers,
        queue_size=max(2, chunk_workers * 2),
    )

    try:
        pages_indexed, nodes_indexed = await pipeline.run(
            document_paths=_iter_content_markdown(run_path),
            load_document=load_document,
        )
    finally:
        progress.close()
        await _close_ollama_embedding(embed_model)
        await _close_opensearch_client(vector_client)

    worker_note = f" using {chunk_workers} workers" if chunk_workers > 1 else ""
    logger.info(
        "Indexed %d pages as %d nodes into OpenSearch index '%s'%s",
        pages_indexed,
        nodes_indexed,
        opensearch_index,
        worker_note,
    )

    return index


class _BackpressureIndexingPipeline:
    """Co-ordinates streaming document processing with bounded queues and workers."""

    def __init__(
        self,
        *,
        node_parser: SemanticSplitterNodeParser,
        index: VectorStoreIndex,
        progress: tqdm,
        chunk_workers: int,
        queue_size: int,
    ) -> None:
        self._node_parser = node_parser
        self._index = index
        self._progress = progress
        self._chunk_workers = max(1, chunk_workers)
        self._queue_size = max(1, queue_size)
        self._pages_indexed = 0
        self._nodes_indexed = 0

    async def run(
        self,
        *,
        document_paths: Iterable[Path],
        load_document: Callable[[Path], Awaitable[Optional[Document]]],
    ) -> tuple[int, int]:
        doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=self._queue_size)
        node_queue: asyncio.Queue[Optional[Sequence[Any]]] = asyncio.Queue(maxsize=self._queue_size)

        async def producer() -> None:
            try:
                for content_path in document_paths:
                    document = await load_document(content_path)
                    if document is None:
                        continue
                    await doc_queue.put(document)
            finally:
                for _ in range(self._chunk_workers):
                    await doc_queue.put(None)

        async def chunk_worker() -> None:
            while True:
                document = await doc_queue.get()
                if document is None:
                    await node_queue.put(None)
                    return
                nodes = await asyncio.to_thread(
                    self._node_parser.get_nodes_from_documents,
                    [document],
                )
                self._progress.update(1)
                if nodes:
                    await node_queue.put(nodes)

        async def insert_worker() -> None:
            finished = 0
            while finished < self._chunk_workers:
                batch = await node_queue.get()
                if batch is None:
                    finished += 1
                    continue
                await _insert_nodes_async(self._index, batch)
                self._pages_indexed += 1
                self._nodes_indexed += len(batch)
                self._progress.set_postfix(nodes=self._nodes_indexed)

        tasks = [
            asyncio.create_task(producer()),
            *(asyncio.create_task(chunk_worker()) for _ in range(self._chunk_workers)),
            asyncio.create_task(insert_worker()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return self._pages_indexed, self._nodes_indexed


def _document_from_markdown(
    content_path: Path,
    *,
    run_path: Path,
    additional_metadata: Optional[dict[str, Any]] = None,
) -> Document:
    front_matter, body = _load_markdown_with_front_matter(content_path)
    metadata: dict[str, Any] = front_matter.copy()
    if additional_metadata:
        metadata.update(additional_metadata)

    relative_path = content_path.relative_to(run_path)
    metadata.setdefault("source_path", relative_path.as_posix())
    metadata.setdefault("run_dir", run_path.name)

    doc_id = str(
        metadata.get("url")
        or metadata.get("id")
        or relative_path.as_posix()
    )
    return Document(text=body, metadata=metadata, doc_id=doc_id)


async def _insert_nodes_async(index: VectorStoreIndex, nodes: Sequence[Any]) -> None:
    if not nodes:
        return
    ainsert = getattr(index, "ainsert_nodes", None)
    if callable(ainsert):
        await ainsert(nodes)
        return
    await asyncio.to_thread(index.insert_nodes, nodes)


def _iter_content_markdown(run_path: Path) -> Iterable[Path]:
    for content_path in run_path.rglob("content.md"):
        if "files" in content_path.parts:
            continue
        yield content_path


def _load_markdown_with_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    raw_text = path.read_text(encoding="utf-8")
    normalized = raw_text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError("Front matter missing")

    end_marker = "\n---\n"
    end_index = normalized.find(end_marker, len("---\n"))
    if end_index == -1:
        raise ValueError("Front matter terminator not found")

    front_matter_block = normalized[len("---\n") : end_index]
    body = normalized[end_index + len(end_marker) :]

    try:
        metadata = yaml.safe_load(front_matter_block) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid front matter: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Front matter must resolve to a mapping")

    return metadata, body.lstrip("\n")



