from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import os
import logging
import types
from typing import Any, Dict, Iterable, List, Optional, Tuple
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from socora_indexer.opensearch_utils import ensure_opensearch_index

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are Socora's civic knowledge assistant.\n"
    "Socora helps municipalities, schools, and local institutions make public records accessible and searchable for the community.\n"
    "Socora.ai makes this information accessible enabling residents to ask direct questions about their town, school district, or community services.\n"
    "Today's date is: 2025-09-21.\n"
    "Institution: Cresskill, NJ. Homepage: https://cresskillboro.com\n"
    "When you rely on tools, specify precise inputs so they reflect the latest context and time-sensitive needs (for example, use explicit dates, deadlines, and durations).\n"
    "Before calling the knowledge_base tool, translate the request into multiple predictive search fragments that mirror how the content is likely written.\n"
    "Follow these query principles:\n"
    "- Do not restate the user's words verbatim; infer document-style phrases that express the same mission.\n"
    "- Emphasize department names, program titles, governing bodies, dates, fiscal periods, and key actions that could appear in headings.\n"
    "- Provide several varied guesses (e.g., different dates, statuses, or roles) so the retrievers can triangulate the right material.\n"
    "Answer questions using the indexed civic content. Highlight reliable information that helps users, and community stakeholders understand local services, governance, and programs."
)

def _apply_embedding_prefix(
    embed_model: OllamaEmbedding,
    *,
    document_prefix: str,
    query_prefix: str,
) -> OllamaEmbedding:
    """Ensure embedding inputs include the required task prefixes."""

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


class _OpenSearchBM25Retriever(BaseRetriever):
    """Lightweight lexical retriever backed by OpenSearch BM25."""

    def __init__(
        self,
        *,
        vector_store: OpensearchVectorStore,
        similarity_top_k: int,
    ) -> None:
        callback_manager = getattr(vector_store, "callback_manager", None)
        super().__init__(callback_manager=callback_manager)
        self._vector_store = vector_store
        self._vector_client = vector_store.client
        self._similarity_top_k = max(1, similarity_top_k)
        dim = getattr(self._vector_client, "_dim", 0) or 0
        self._dummy_embedding = [0.0] * dim

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not query_bundle.query_str:
            return []
        result = self._vector_client.query(
            VectorStoreQueryMode.TEXT_SEARCH,
            query_bundle.query_str,
            self._dummy_embedding,
            self._similarity_top_k,
        )
        return self._nodes_from_result(result)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not query_bundle.query_str:
            return []
        try:
            result = await self._vector_client.aquery(
                VectorStoreQueryMode.TEXT_SEARCH,
                query_bundle.query_str,
                self._dummy_embedding,
                self._similarity_top_k,
            )
        except asyncio.TimeoutError:
            logger.warning("OpenSearch BM25 async query timed out; retrying synchronously.")
            result = self._vector_client.query(
                VectorStoreQueryMode.TEXT_SEARCH,
                query_bundle.query_str,
                self._dummy_embedding,
                self._similarity_top_k,
            )
        except Exception as exc:
            logger.warning("OpenSearch BM25 async query failed (%s); falling back to sync query.", exc)
            result = self._vector_client.query(
                VectorStoreQueryMode.TEXT_SEARCH,
                query_bundle.query_str,
                self._dummy_embedding,
                self._similarity_top_k,
            )
        return self._nodes_from_result(result)

    @staticmethod
    def _nodes_from_result(result) -> List[NodeWithScore]:
        if not result or not getattr(result, "nodes", None):
            return []
        scores = result.similarities or []
        nodes_with_scores: List[NodeWithScore] = []
        for node, score in zip(result.nodes, scores or [None] * len(result.nodes)):
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores

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
    embed_model = _apply_embedding_prefix(
        embed_model,
        document_prefix='',
        query_prefix='search_query: ',
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
    chat_provider: str,
    llm_model: Optional[str],
    ollama_base_url: str,
    openai_api_key: Optional[str],
    temperature: float,
    request_timeout: float,
    similarity_top_k: int,
    system_prompt: str,
    verbose: bool,
    bm25_top_k: Optional[int] = None,
    hybrid_alpha: float = 0.6,
) -> FunctionAgent:
    provider = (chat_provider or "ollama").strip().lower()
    default_llm_model = "gpt-4o-mini" if provider == "openai" else "llama3"
    model_name = llm_model or default_llm_model
    if provider == "openai":
        resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI chat provider selected but no API key supplied via --openai-api-key or OPENAI_API_KEY.")
        try:
            from llama_index.llms.openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAI chat provider requires the 'llama-index-llms-openai' package. Install it to continue.") from exc
        llm = OpenAI(
            model=model_name,
            api_key=resolved_api_key,
            temperature=temperature,
            timeout=request_timeout,
        )
    elif provider == "ollama":
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    else:
        raise ValueError(f"Unsupported chat provider: {chat_provider!r}. Expected 'ollama' or 'openai'.")

    dense_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    bm25_target_top_k = bm25_top_k if bm25_top_k is not None else similarity_top_k
    bm25_results_top_k = max(1, bm25_target_top_k)
    vector_store = getattr(index, "vector_store", None)
    bm25_retriever: Optional[BaseRetriever] = None
    if isinstance(vector_store, OpensearchVectorStore):
        try:
            bm25_retriever = _OpenSearchBM25Retriever(
                vector_store=vector_store,
                similarity_top_k=bm25_results_top_k,
            )
        except Exception as exc:
            logger.warning("Failed to initialise OpenSearch BM25 retriever: %s", exc)
    else:
        logger.warning("Vector store %s is not OpensearchVectorStore; defaulting to dense retrieval only.", type(vector_store).__name__)

    if bm25_retriever is not None:
        fusion_query_prompt = (
            "You rewrite civic knowledge-base searches by predicting {num_queries} document-style queries.\n"
            "Original request: {query}\n"
            "Rules:\n"
            "- Produce exactly {num_queries} distinct queries after the 'Queries:' header; no bullets or numbering.\n"
            "- Do not repeat the user's wording; infer how matching records are titled or summarised.\n"
            "- Mix department names, program keywords, dates, and statuses that could realistically appear in civic documents.\n"
            "- Keep each line concise (<=8 words) and vary phrasing to cover nearby concepts.\n"
            "Example:\nQueries:\ncapital improvement plan fy2025\nplanning board meeting minutes april 2025\npublic works maintenance schedule 2025\nmunicipal budget hearing recap 2025\n---\n"
            "Now respond in the same format.\n"
            "Queries:\n"
        )
        fusion_weights = [1.0 - hybrid_alpha, hybrid_alpha]
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            retriever_weights=fusion_weights,
            similarity_top_k=similarity_top_k,
            num_queries=4,
            mode=FUSION_MODES.RELATIVE_SCORE,
            query_gen_prompt=fusion_query_prompt,
            use_async=False,
            verbose=verbose,
            llm=llm,
        )
    else:
        hybrid_retriever = dense_retriever

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        streaming=False,
    )
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="knowledge_base",
        description="Predictive civic search. Provide multiple short document-style queries that mirror record titles; dense and BM25 signals are fused automatically.",
    )
    return FunctionAgent(
        tools=[tool],
        llm=llm,
        system_prompt=system_prompt,
        verbose=verbose,
        streaming=False,
    )



def _format_iteration_prefix(iteration_index: int, max_iterations: Optional[int]) -> str:
    total = "?" if not isinstance(max_iterations, int) or max_iterations <= 0 else str(max_iterations)
    return f"[Iteration {iteration_index}/{total}]"


def _extract_thought_from_output(event: AgentOutput) -> Optional[str]:
    response = getattr(event, "response", None)
    if response is None:
        return None
    extras = getattr(response, "additional_kwargs", {}) or {}
    thought = (
        extras.get("thinking")
        or extras.get("thought")
        or extras.get("thinking_delta")
        or response.content
    )
    if not thought:
        return None
    return " ".join(str(thought).split())

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
    iteration_index = 0
    display_limit = max_iterations or getattr(agent, "max_iterations", None)

    try:
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput):
                if getattr(event, "tool_calls", None):
                    iteration_index += 1
                    thought = _extract_thought_from_output(event)
                    if thought:
                        prefix = _format_iteration_prefix(iteration_index, display_limit)
                        print(f"{prefix} Thought: {thought}")
                final_output = event
            if isinstance(event, ToolCallResult):
                tool_events.append(event)
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
        "--chat-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat model provider to use (ollama or openai).",
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
        "--llm-model",
        default=None,
        help="Chat model id (defaults to llama3 for Ollama or gpt-4o-mini for OpenAI).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (optional; falls back to OPENAI_API_KEY environment variable).",
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
        "--bm25-top-k",
        type=int,
        default=50,
        help="Optional override for the number of keyword/BM25 results to consider before reranking.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.6,
        help="Weight (0-1) for embedding similarity when combining with BM25 scores.",
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
    logging.getLogger("opensearch").setLevel(logging.WARNING)
    logging.getLogger("opensearchpy").setLevel(logging.WARNING)

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
    bm25_top_k = args.bm25_top_k or args.similarity_top_k
    hybrid_alpha = max(0.0, min(args.hybrid_alpha, 1.0))
    llm_model = args.llm_model
    if not llm_model:
        llm_model = "gpt-4o-mini" if args.chat_provider == "openai" else "llama3"

    agent = _create_function_agent(
        index,
        chat_provider=args.chat_provider,
        llm_model=llm_model,
        ollama_base_url=args.ollama_base_url,
        openai_api_key=args.openai_api_key,
        temperature=args.temperature,
        request_timeout=args.ollama_request_timeout,
        similarity_top_k=args.similarity_top_k,
        system_prompt=system_prompt,
        verbose=args.verbose,
        bm25_top_k=bm25_top_k,
        hybrid_alpha=hybrid_alpha,
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

