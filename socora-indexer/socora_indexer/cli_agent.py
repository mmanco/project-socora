from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)

from socora_indexer.opensearch_utils import ensure_opensearch_index

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class _HybridBM25Retriever(BaseRetriever):
    """Combine embedding and lexical (BM25) retrieval with weighted reranking."""

    def __init__(
        self,
        *,
        index: VectorStoreIndex,
        similarity_top_k: int,
        bm25_top_k: int,
        alpha: float,
    ) -> None:
        self._index = index
        self._vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        self._vector_store = index.vector_store
        self._bm25_top_k = max(1, bm25_top_k)
        self._alpha = max(0.0, min(alpha, 1.0))
        self._target_top_k = similarity_top_k
        dim = getattr(self._vector_store, '_dim', 0) or 0
        self._dummy_embedding = [0.0] * dim
        callback_manager = getattr(self._vector_retriever, 'callback_manager', None)
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_search(query_bundle.query_str)
        return self._combine(vector_nodes, bm25_nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = await self._vector_retriever.aretrieve(query_bundle)
        bm25_nodes = await self._abm25_search(query_bundle.query_str)
        return self._combine(vector_nodes, bm25_nodes)

    def _bm25_search(self, query_str: Optional[str]) -> List[NodeWithScore]:
        if not query_str:
            return []
        result = self._vector_store.query(
            VectorStoreQueryMode.TEXT_SEARCH,
            query_str,
            self._dummy_embedding,
            self._bm25_top_k,
        )
        return self._nodes_from_result(result)

    async def _abm25_search(self, query_str: Optional[str]) -> List[NodeWithScore]:
        if not query_str:
            return []
        result = await self._vector_store.aquery(
            VectorStoreQueryMode.TEXT_SEARCH,
            query_str,
            self._dummy_embedding,
            self._bm25_top_k,
        )
        return self._nodes_from_result(result)

    def _nodes_from_result(self, result) -> List[NodeWithScore]:
        if not result or not result.nodes:
            return []
        scores = result.similarities or []
        pairs = list(zip(result.nodes, scores or [None] * len(result.nodes)))
        return [NodeWithScore(node=node, score=score) for node, score in pairs]

    def _combine(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        combined: Dict[str, NodeWithScore] = {}
        node_lookup: Dict[str, NodeWithScore] = {}
        for node in vector_nodes + bm25_nodes:
            node_lookup[node.node_id] = node

        embed_scores = {n.node_id: n.score or 0.0 for n in vector_nodes}
        bm25_scores = {n.node_id: n.score or 0.0 for n in bm25_nodes}
        embed_norm = self._normalize(embed_scores)
        bm25_norm = self._normalize(bm25_scores)

        for node_id, base in node_lookup.items():
            combined_score = 0.0
            if node_id in embed_norm:
                combined_score += self._alpha * embed_norm[node_id]
            if node_id in bm25_norm:
                combined_score += (1.0 - self._alpha) * bm25_norm[node_id]
            combined[node_id] = NodeWithScore(node=base.node, score=combined_score)

        ranked = sorted(
            combined.values(), key=lambda n: n.score or 0.0, reverse=True
        )
        return ranked[: self._target_top_k]

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        max_val = max(values)
        min_val = min(values)
        if max_val - min_val < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

DEFAULT_SYSTEM_PROMPT = (
    "You are Socora's civic knowledge assistant. Today's date is: 2025-09-21.\n"
    "Socora helps municipalities, schools, and local institutions make public records accessible and searchable for the community.\n"
    "Socora.ai makes this information accessible enabling residents to ask direct questions about their town, school district, or community services.\n"
    "When you rely on tools, specify precise inputs so they reflect the latest context and time-sensitive needs (for example, use explicit dates, deadlines, and durations).\n"
    "Answer questions using the indexed civic content. Highlight reliable information that helps residents, administrators, and community stakeholders understand local services, governance, and programs."
)


def _parse_opensearch_options(options: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw in options:
        if not raw:
            continue
        key, sep, value = raw.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid OpenSearch option '{raw}'. Expected key=value format.")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid OpenSearch option '{raw}'. Key cannot be empty.")
        parsed[key] = _coerce_value(value.strip())
    return parsed



def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _build_index(
    *,
    opensearch_endpoint: str,
    opensearch_index: str,
    embedding_dimension: int,
    opensearch_username: Optional[str],
    opensearch_password: Optional[str],
    opensearch_options: Dict[str, Any],
    ollama_base_url: str,
    ollama_embed_model: str,
    embed_batch_size: int,
) -> VectorStoreIndex:
    embed_model = OllamaEmbedding(
        model_name=ollama_embed_model,
        base_url=ollama_base_url,
        embed_batch_size=embed_batch_size,
    )

    client_kwargs = dict(opensearch_options)
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

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=False,
    )


def _create_function_agent(
    index: VectorStoreIndex,
    *,
    ollama_base_url: str,
    ollama_llm_model: str,
    temperature: float,
    request_timeout: float,
    similarity_top_k: int,
    system_prompt: str,
    verbose: bool,
    bm25_top_k: Optional[int] = None,
    hybrid_alpha: float = 0.6,
) -> FunctionAgent:
    llm = Ollama(
        model=ollama_llm_model,
        base_url=ollama_base_url,
        temperature=temperature,
        request_timeout=request_timeout,
    )
    hybrid_retriever = _HybridBM25Retriever(
        index=index,
        similarity_top_k=similarity_top_k,
        bm25_top_k=bm25_top_k or similarity_top_k,
        alpha=hybrid_alpha,
    )
    query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="knowledge_base",
        description="Answer questions using the indexed OpenSearch content.",
    )
    return FunctionAgent(
        tools=[tool],
        llm=llm,
        system_prompt=system_prompt,
        verbose=verbose,
        streaming=False,
    )


async def _interactive_chat(agent: FunctionAgent, *, show_sources: bool, max_sources: int) -> None:
    print("Type your questions to chat with the indexed content. Type 'exit' to quit.\n")

    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = (await asyncio.to_thread(input, "You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        chat_history.append({"role": "user", "content": user_input})

        try:
            final_output, tool_events = await _execute_agent_turn(
                agent=agent,
                user_input=user_input,
                chat_history=list(chat_history[:-1]),
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error("Agent failed to start: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            print("Assistant: Sorry, I could not process that question.")
            chat_history.pop()
            continue

        answer = ""
        fallback_sources: List[Tuple[str, Optional[float]]] = []
        if final_output is not None:
            answer = (final_output.response.content or "").strip()

        if not answer:
            fallback_answer, fallback_sources = await _fallback_tool_answer(
                agent=agent,
                user_input=user_input,
                max_sources=max_sources,
            )
            answer = fallback_answer.strip()

        if not answer:
            print("Assistant: I wasn't able to produce an answer for that prompt.")
            chat_history.pop()
            continue

        print(f"\nAssistant: {answer}\n")
        chat_history.append({"role": "assistant", "content": answer})

        if show_sources:
            sources = _extract_sources(tool_events, max_sources)
            if not sources and fallback_sources:
                sources = fallback_sources
            if sources:
                print("Sources:")
                for source, score in sources:
                    score_info = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                    print(f"  - {source}{score_info}")
                print()


async def _execute_agent_turn(
    *,
    agent: FunctionAgent,
    user_input: str,
    chat_history: List[Dict[str, str]],
    max_iterations: Optional[int] = None,
) -> Tuple[Optional[AgentOutput], List[ToolCallResult]]:
    handler = agent.run(
        user_msg=user_input,
        chat_history=chat_history,
        max_iterations=max_iterations,
    )
    tool_events: List[ToolCallResult] = []
    final_output: Optional[AgentOutput] = None

    try:
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                tool_events.append(event)
            if isinstance(event, AgentOutput):
                final_output = event
    except Exception as exc:
        if isinstance(exc, ValueError) and "empty message" in str(exc).lower():
            return None, tool_events
        raise

    try:
        if final_output is None:
            final_output = await handler
    except Exception as exc:
        if isinstance(exc, ValueError) and "empty message" in str(exc).lower():
            return None, tool_events
        raise

    return final_output, tool_events
async def _fallback_tool_answer(
    *,
    agent: FunctionAgent,
    user_input: str,
    max_sources: int,
) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
    tools = getattr(agent, "tools", None) or []
    for tool in tools:
        acall = getattr(tool, "acall", None)
        call = getattr(tool, "call", None)
        try:
            if callable(acall):
                try:
                    tool_output = await acall(input=user_input)
                except TypeError:
                    tool_output = await acall(user_input)
            elif callable(call):
                try:
                    tool_output = await asyncio.to_thread(call, input=user_input)
                except TypeError:
                    tool_output = await asyncio.to_thread(call, user_input)
            else:
                continue
        except Exception:
            continue

        if tool_output is None:
            continue

        answer = str(getattr(tool_output, "content", "") or "")
        sources = _extract_sources_from_tool_output(tool_output, max_sources)
        return answer, sources

    return "", []


def _extract_sources_from_tool_output(
    tool_output: Any,
    max_sources: int,
) -> List[Tuple[str, Optional[float]]]:
    raw_output = getattr(tool_output, "raw_output", None)
    if raw_output is None:
        return []
    source_nodes = getattr(raw_output, "source_nodes", None)
    if not source_nodes:
        return []

    collected: List[Tuple[str, Optional[float]]] = []
    seen: set[str] = set()

    for node in source_nodes:
        metadata = getattr(node, "metadata", {}) or {}
        source = metadata.get("source_path") or metadata.get("url") or getattr(node, "node_id", None)
        if not source or source in seen:
            continue
        score = getattr(node, "score", None)
        collected.append((str(source), score if isinstance(score, (int, float)) else None))
        seen.add(str(source))
        if len(collected) >= max_sources:
            break

    return collected

def _extract_sources(events: List[ToolCallResult], max_sources: int) -> List[Tuple[str, Optional[float]]]:
    collected: List[Tuple[str, Optional[float]]] = []
    seen: set[str] = set()

    for event in events:
        tool_output = getattr(event, "tool_output", None)
        if tool_output is None:
            continue
        raw_output = getattr(tool_output, "raw_output", None)
        if raw_output is None:
            continue
        source_nodes = getattr(raw_output, "source_nodes", None)
        if not source_nodes:
            continue
        for node in source_nodes:
            metadata = getattr(node, "metadata", {}) or {}
            source = metadata.get("url") or metadata.get("source_path") or getattr(node, "node_id", None)
            if not source or source in seen:
                continue
            score = getattr(node, "score", None)
            collected.append((str(source), score if isinstance(score, (int, float)) else None))
            seen.add(str(source))
            if len(collected) >= max_sources:
                return collected

    return collected

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI chat agent backed by OpenSearch.")
    parser.add_argument("--opensearch-endpoint", required=True, help="OpenSearch endpoint URL")
    parser.add_argument("--opensearch-index", required=True, help="OpenSearch index name")
    parser.add_argument("--opensearch-username", help="OpenSearch basic auth username")
    parser.add_argument("--opensearch-password", help="OpenSearch basic auth password")
    parser.add_argument(
        "--opensearch-option",
        action="append",
        default=[],
        help="Additional OpenSearch client option as key=value (may repeat).",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Embedding dimensionality expected by the OpenSearch index.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama server.",
    )
    parser.add_argument(
        "--ollama-embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model id used during indexing.",
    )
    parser.add_argument(
        "--ollama-llm-model",
        default="llama3",
        help="Ollama chat model id used for answer generation.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Batch size to use when embedding queries (defaults to 16).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the chat model.",
    )
    parser.add_argument(
        "--ollama-request-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Ollama chat requests.",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=50,
        help="Number of top nodes to retrieve for each question.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional override for the civic assistant system prompt.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Display source metadata for each response.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="Maximum number of sources to display when --show-sources is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )

    args = parser.parse_args()
    try:
        args.opensearch_options = _parse_opensearch_options(args.opensearch_option)
    except ValueError as exc:
        parser.error(str(exc))
    args.max_sources = max(1, args.max_sources)
    return args


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    index = _build_index(
        opensearch_endpoint=args.opensearch_endpoint,
        opensearch_index=args.opensearch_index,
        embedding_dimension=args.embedding_dimension,
        opensearch_username=args.opensearch_username,
        opensearch_password=args.opensearch_password,
        opensearch_options=args.opensearch_options,
        ollama_base_url=args.ollama_base_url,
        ollama_embed_model=args.ollama_embed_model,
        embed_batch_size=args.embed_batch_size,
    )

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    agent = _create_function_agent(
        index,
        ollama_base_url=args.ollama_base_url,
        ollama_llm_model=args.ollama_llm_model,
        temperature=args.temperature,
        request_timeout=args.ollama_request_timeout,
        similarity_top_k=args.similarity_top_k,
        system_prompt=system_prompt,
        verbose=args.verbose,
    )

    asyncio.run(
        _interactive_chat(
            agent,
            show_sources=args.show_sources,
            max_sources=args.max_sources,
        )
    )


if __name__ == "__main__":
    main()