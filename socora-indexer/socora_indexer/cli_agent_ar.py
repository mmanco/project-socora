from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.opensearch import OpensearchVectorStore

try:
    from llama_index.llms.openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from socora_indexer.cli_agent import (
    DEFAULT_SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
    _OpenSearchBM25Retriever,
    _build_index,
    _parse_opensearch_options,
)

logger = logging.getLogger(__name__)


AGENTIC_REASONING_SYSTEM_PROMPT_TEMPLATE = (
    "{base_prompt}\n\n"
    "You orchestrate Socora's civic retrieval workflow using an iterative ReAct-style loop."
    " For each turn you must either issue a fresh knowledge_base search or finalize the answer."
    " Think step by step about what the user needs, what evidence you already have,"
    " and whether another retrieval would improve confidence."
    " During search iterations, broaden then narrow your civic phrasing to test multiple document styles"
    " (e.g., 'cresskill municipal leadership', 'cresskill mayor', 'cresskill mayor swearing-in $year')."
    " Each iteration must explore a distinct angle such as roles, events, ordinances, or meeting artefacts"
    " so the retriever covers varied phrasing."
    " Always respond with STRICT JSON using keys: thought, action, query, answer."
    "- action must be either 'search' or 'respond'."
    "- thought explains why you chose that action (1-2 sentences)."
    "- When action is 'search', craft query as predictive civic phrasing (<=120 chars)"
    "   and set answer to an empty string."
    "- When action is 'respond', craft the final assistant answer with inline [S#] citations"
    "   that reference collected evidence labels; set query to an empty string."
    " Do not hallucinate evidence and do not return text outside the JSON object."
)

AGENTIC_FINAL_SYSTEM_PROMPT_TEMPLATE = (
    "{base_prompt}\n\n"
    "You are preparing the final response after completing the iterative retrieval loop."
    " Use only the provided evidence snippets labelled [S#]."
    " Cite supporting statements with the matching [S#] markers."
    " If the evidence is insufficient or missing, state that clearly and suggest next steps."
)


@dataclass
class IterationState:
    iteration: int
    thought: str
    action: str
    query: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class SourceRegistry:
    label_to_id: Dict[str, str] = field(default_factory=dict)
    id_to_label: Dict[str, str] = field(default_factory=dict)
    id_to_score: Dict[str, Optional[float]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def register(self, node: NodeWithScore) -> str:
        label = _resolve_source_label(node)
        if label not in self.label_to_id:
            source_id = f"S{len(self.order) + 1}"
            self.label_to_id[label] = source_id
            self.id_to_label[source_id] = label
            self.order.append(source_id)
        else:
            source_id = self.label_to_id[label]

        score = node.score
        if isinstance(score, (int, float)):
            previous = self.id_to_score.get(source_id)
            if previous is None or score > previous:
                self.id_to_score[source_id] = float(score)
        elif source_id not in self.id_to_score:
            self.id_to_score[source_id] = None
        return source_id

    def display_sources(self, max_sources: int) -> List[Tuple[str, str, Optional[float]]]:
        items: List[Tuple[str, str, Optional[float]]] = []
        for source_id in self.order[:max_sources]:
            label = self.id_to_label.get(source_id, source_id)
            score = self.id_to_score.get(source_id)
            items.append((source_id, label, score))
        return items


@dataclass
class ReasonerDecision:
    thought: str
    action: str
    query: str = ""
    answer: str = ""


def _compose_system_prompts(user_prompt: Optional[str], max_iterations: int) -> Tuple[str, str]:
    base = (user_prompt or BASE_SYSTEM_PROMPT).strip()
    reasoning_prompt = AGENTIC_REASONING_SYSTEM_PROMPT_TEMPLATE.format(
        base_prompt=base,
        max_iterations=max_iterations,
    )
    final_prompt = AGENTIC_FINAL_SYSTEM_PROMPT_TEMPLATE.format(
        base_prompt=base,
        max_iterations=max_iterations,
    )
    return reasoning_prompt, final_prompt


def _resolve_source_label(node: NodeWithScore) -> str:
    metadata = node.metadata or {}
    for key in ("title", "source_path", "url", "file_name", "display_name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return getattr(node, "node_id", "unknown_source")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _summarize_nodes_for_prompt(
    nodes: Sequence[NodeWithScore],
    registry: SourceRegistry,
    *,
    max_nodes: int,
    snippet_chars: int = 280,
) -> Tuple[str, List[str]]:
    lines: List[str] = []
    used_ids: List[str] = []
    for node in list(nodes)[:max_nodes]:
        source_id = registry.register(node)
        used_ids.append(source_id)
        text = getattr(node, "text", "") or getattr(getattr(node, "node", None), "text", "")
        snippet = _normalize_whitespace(text)
        if len(snippet) > snippet_chars:
            snippet = f"{snippet[: snippet_chars - 3]}..."
        if not snippet:
            snippet = "(no preview available)"
        lines.append(f"{source_id}: {snippet}")
    if not lines:
        return "No new evidence captured.", used_ids
    return "\n".join(lines), used_ids


def _format_console_detail(
    total_candidates: int,
    used_ids: Sequence[str],
    registry: SourceRegistry,
) -> str:
    if total_candidates <= 0:
        return "no candidates retrieved"
    if used_ids:
        top_id = used_ids[0]
    elif registry.order:
        top_id = registry.order[-1]
    else:
        return f"retrieved {total_candidates} candidates"
    label = registry.id_to_label.get(top_id, top_id)
    score = registry.id_to_score.get(top_id)
    if isinstance(score, float):
        return f"retrieved {total_candidates} candidates; top {top_id} {label} (score {score:.3f})"
    return f"retrieved {total_candidates} candidates; top {top_id} {label}"


def _format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "None"
    lines: List[str] = []
    for turn in chat_history:
        role = turn.get("role", "user").capitalize()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _format_iteration_history(states: Sequence[IterationState]) -> str:
    if not states:
        return "None"
    blocks: List[str] = []
    for state in states:
        header = f"Iteration {state.iteration} ({state.action})"
        details = [f"Thought: {state.thought}"]
        if state.query:
            details.append(f"Query: {state.query}")
        if state.evidence:
            details.append("Findings:\n" + textwrap.indent(state.evidence, "    "))
        blocks.append("\n".join([header, *details]))
    return "\n\n".join(blocks)


def _format_source_catalog(registry: SourceRegistry) -> str:
    if not registry.order:
        return "No sources collected yet."
    lines: List[str] = []
    for source_id in registry.order:
        label = registry.id_to_label.get(source_id, source_id)
        lines.append(f"{source_id}: {label}")
    return "\n".join(lines)


def _build_reasoning_user_prompt(
    *,
    user_input: str,
    chat_history: List[Dict[str, str]],
    states: Sequence[IterationState],
    registry: SourceRegistry,
    remaining_iterations: int,
) -> str:
    history_text = _format_chat_history(chat_history)
    iteration_text = _format_iteration_history(states)
    source_catalog = _format_source_catalog(registry)
    prompt = f"""
You are deciding the next action for the civic retrieval agent.

User question:
{user_input}

Conversation so far:
{history_text}

Iteration log:
{iteration_text}

Collected source labels:
{source_catalog}

Remaining iterations (including this decision): {remaining_iterations}

Instructions:
- If more evidence is needed, set action to "search" with a fresh predictive civic query (<=120 chars).
- Progress from broad civic identifiers (e.g., government leadership) toward focused events, job titles, filings, or timelines so each query probes a new phrasing of the topic.
- Avoid repeating identical queries unless you refine them to target a different document angle.
- If the evidence is sufficient, set action to "respond" and draft the final answer with inline [S#] citations.
- Do not fabricate citations or information not present in the collected evidence.

Respond ONLY with JSON matching this schema:
{{
  "thought": "<reasoning>",
  "action": "search" | "respond",
  "query": "<query when action=search else empty>",
  "answer": "<final answer when action=respond else empty>"
}}
"""
    return textwrap.dedent(prompt).strip()


def _safe_parse_decision(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    text = raw_text.strip()
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    fence_match = fence_pattern.match(text)
    if fence_match:
        text = fence_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _parse_reasoner_output(raw_text: str, fallback_query: str) -> ReasonerDecision:
    data = _safe_parse_decision(raw_text) or {}
    thought = _normalize_whitespace(str(data.get("thought", ""))) or "Planning next step."
    action = str(data.get("action", "search")).strip().lower()
    if action not in {"search", "respond"}:
        action = "search"
    query = _normalize_whitespace(str(data.get("query", "")))
    answer_value = data.get("answer", "")
    answer = str(answer_value) if answer_value is not None else ""
    if action == "search" and not query:
        query = fallback_query
    return ReasonerDecision(thought=thought, action=action, query=query, answer=answer)


async def _reason_about_next_step(
    *,
    llm: Any,
    reasoning_system_prompt: str,
    user_input: str,
    chat_history: List[Dict[str, str]],
    states: Sequence[IterationState],
    registry: SourceRegistry,
    remaining_iterations: int,
    fallback_query: str,
) -> ReasonerDecision:
    messages = [
        ChatMessage.from_str(reasoning_system_prompt, role=MessageRole.SYSTEM),
        ChatMessage.from_str(
            _build_reasoning_user_prompt(
                user_input=user_input,
                chat_history=chat_history,
                states=states,
                registry=registry,
                remaining_iterations=remaining_iterations,
            ),
            role=MessageRole.USER,
        ),
    ]
    response = await llm.achat(messages)  # type: ignore[attr-defined]
    raw_text = response.message.content or ""
    return _parse_reasoner_output(raw_text, fallback_query=fallback_query)


async def _perform_retrieval(
    *,
    retriever: BaseRetriever,
    query: str,
    registry: SourceRegistry,
    evidence_snippet_limit: int,
) -> Tuple[List[NodeWithScore], str, str]:
    if not query:
        return [], "Query was empty; no retrieval executed.", "no retrieval executed"

    bundle = QueryBundle(query_str=query)
    try:
        nodes = await retriever.aretrieve(bundle)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.warning("Async retrieval failed (%s); retrying synchronously.", exc)
        try:
            nodes = retriever.retrieve(bundle)
        except Exception as inner_exc:  # pragma: no cover - runtime safeguard
            logger.error("Retrieval failed: %s", inner_exc)
            return [], "Retrieval failed; no evidence captured.", "retrieval failed"

    summary, used_ids = _summarize_nodes_for_prompt(
        nodes,
        registry,
        max_nodes=max(evidence_snippet_limit, 1),
    )
    detail = _format_console_detail(len(nodes), used_ids, registry)
    return list(nodes), summary, detail


async def _synthesize_final_answer(
    *,
    llm: Any,
    final_system_prompt: str,
    user_input: str,
    chat_history: List[Dict[str, str]],
    states: Sequence[IterationState],
    registry: SourceRegistry,
    iteration_limit_hit: bool,
) -> str:
    history_text = _format_chat_history(chat_history)
    evidence_blocks: List[str] = []
    for state in states:
        if state.action == "search" and state.evidence:
            header = f"Iteration {state.iteration} query '{state.query or ''}':"
            evidence_blocks.append(f"{header}\n{textwrap.indent(state.evidence, '  ')}")
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence was retrieved."
    source_catalog = _format_source_catalog(registry)
    limit_note = (
        "Iteration limit was reached; rely on the collected evidence."
        if iteration_limit_hit
        else ""
    )

    user_prompt = f"""
Prepare the final civic answer.

User question:
{user_input}

Conversation so far:
{history_text}

Evidence summary:
{evidence_text}

Source catalog (use these labels for citations):
{source_catalog}

{limit_note}

Guidelines:
- Respond in a helpful civic tone.
- Reference supporting statements with inline [S#] citations that match the catalog above.
- If evidence is insufficient, say so explicitly and suggest where the user might look next.
"""
    messages = [
        ChatMessage.from_str(final_system_prompt, role=MessageRole.SYSTEM),
        ChatMessage.from_str(textwrap.dedent(user_prompt).strip(), role=MessageRole.USER),
    ]
    response = await llm.achat(messages)  # type: ignore[attr-defined]
    return (response.message.content or "").strip()


async def _fallback_query_engine_answer(
    query_engine: RetrieverQueryEngine,
    user_input: str,
) -> Tuple[str, List[NodeWithScore]]:
    try:
        response = await query_engine.aquery(user_input)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Fallback query engine failed: %s", exc)
        return "", []

    answer = (
        getattr(response, "response", None)
        or getattr(response, "response_text", "")
        or ""
    ).strip()
    source_nodes = getattr(response, "source_nodes", None) or []
    return answer, list(source_nodes)


def _create_llm_and_retriever(
    index,
    *,
    chat_provider: str,
    llm_model: Optional[str],
    ollama_base_url: str,
    openai_api_key: Optional[str],
    temperature: float,
    request_timeout: float,
    similarity_top_k: int,
    verbose: bool,
    bm25_top_k: Optional[int],
    hybrid_alpha: float,
) -> Tuple[Any, BaseRetriever, RetrieverQueryEngine]:
    provider = (chat_provider or "ollama").strip().lower()
    default_llm_model = "gpt-4o-mini" if provider == "openai" else "llama3"
    model_name = llm_model or default_llm_model

    if provider == "openai":
        if OpenAI is None:
            raise ImportError(
                "OpenAI chat provider requires the 'llama-index-llms-openai' package."
            )
        resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI chat provider selected but no API key supplied via --openai-api-key or OPENAI_API_KEY."
            )
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
        raise ValueError(
            f"Unsupported chat provider: {chat_provider!r}. Expected 'ollama' or 'openai'."
        )

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
        if verbose:
            logger.warning(
                "Vector store %s is not OpensearchVectorStore; defaulting to dense retrieval only.",
                type(vector_store).__name__,
            )

    if bm25_retriever is not None:
        fusion_weights = [1.0 - hybrid_alpha, hybrid_alpha]
        hybrid_retriever: BaseRetriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            retriever_weights=fusion_weights,
            similarity_top_k=similarity_top_k,
            num_queries=4,
            mode=FUSION_MODES.RELATIVE_SCORE,
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
    return llm, hybrid_retriever, query_engine


class AgenticResponder:
    def __init__(
        self,
        *,
        llm: Any,
        retriever: BaseRetriever,
        query_engine: RetrieverQueryEngine,
        reasoning_system_prompt: str,
        final_system_prompt: str,
        max_iterations: int,
        max_sources: int,
        evidence_snippet_limit: int,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.query_engine = query_engine
        self.reasoning_system_prompt = reasoning_system_prompt
        self.final_system_prompt = final_system_prompt
        self.max_iterations = max(1, max_iterations)
        self.max_sources = max(1, max_sources)
        self.evidence_snippet_limit = max(1, evidence_snippet_limit)

    async def answer(
        self,
        *,
        user_input: str,
        chat_history: List[Dict[str, str]],
    ) -> Tuple[str, List[Tuple[str, str, Optional[float]]]]:
        states: List[IterationState] = []
        registry = SourceRegistry()
        fallback_query = _normalize_whitespace(user_input)
        final_answer = ""
        iteration_limit_hit = False

        for iteration in range(1, self.max_iterations + 1):
            remaining = self.max_iterations - len(states)
            decision = await _reason_about_next_step(
                llm=self.llm,
                reasoning_system_prompt=self.reasoning_system_prompt,
                user_input=user_input,
                chat_history=chat_history,
                states=states,
                registry=registry,
                remaining_iterations=remaining,
                fallback_query=fallback_query,
            )

            state = IterationState(
                iteration=iteration,
                thought=decision.thought,
                action=decision.action,
            )
            print(f"[Iteration {iteration}/{self.max_iterations}] Thought: {decision.thought}")

            if decision.action == "respond":
                final_answer = decision.answer.strip()
                states.append(state)
                print(
                    f"[Iteration {iteration}/{self.max_iterations}] Finalizing without additional search."
                )
                break

            query = decision.query.strip() or fallback_query
            state.action = "search"
            state.query = query
            print(f"[Iteration {iteration}/{self.max_iterations}] knowledge_base query: {query}")

            nodes, evidence_summary, detail = await _perform_retrieval(
                retriever=self.retriever,
                query=query,
                registry=registry,
                evidence_snippet_limit=self.evidence_snippet_limit,
            )
            state.evidence = evidence_summary
            states.append(state)
            print(f"    -> {detail}")

            fallback_query = query

            if iteration == self.max_iterations:
                iteration_limit_hit = True
                print(
                    f"[Iteration {iteration}/{self.max_iterations}] Iteration limit reached; synthesizing best available answer."
                )
        else:
            iteration_limit_hit = True

        if not final_answer:
            final_answer = await _synthesize_final_answer(
                llm=self.llm,
                final_system_prompt=self.final_system_prompt,
                user_input=user_input,
                chat_history=chat_history,
                states=states,
                registry=registry,
                iteration_limit_hit=iteration_limit_hit,
            )

        if not final_answer.strip():
            fallback_text, fallback_nodes = await _fallback_query_engine_answer(
                self.query_engine,
                user_input,
            )
            if fallback_text:
                final_answer = fallback_text
                for node in fallback_nodes:
                    registry.register(node)

        if not final_answer.strip():
            final_answer = "I wasn't able to locate enough indexed evidence to answer that confidently."

        sources = registry.display_sources(self.max_sources)
        return final_answer.strip(), sources


async def _interactive_chat(
    responder: AgenticResponder,
    *,
    show_sources: bool,
) -> None:
    print("Type your questions to trigger agentic retrieval. Type 'exit' to quit.\n")

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
            answer, sources = await responder.answer(
                user_input=user_input,
                chat_history=list(chat_history[:-1]),
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error(
                "Agentic responder failed: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            print("Assistant: Sorry, I could not process that question.")
            chat_history.pop()
            continue

        print(f"\nAssistant: {answer}\n")
        chat_history.append({"role": "assistant", "content": answer})

        if show_sources and sources:
            print("Sources:")
            for source_id, label, score in sources:
                score_info = f" (score: {score:.3f})" if isinstance(score, float) else ""
                print(f"  - {source_id}: {label}{score_info}")
            print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI chat agent with iterative (agentic) retrieval."
    )
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
        help="Optional override for the civic assistant system prompt (base instructions).",
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
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of agentic retrieval iterations allowed per user question.",
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
    args.max_iterations = max(1, args.max_iterations)
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

    reasoning_prompt, final_prompt = _compose_system_prompts(
        args.system_prompt, args.max_iterations
    )
    bm25_top_k = args.bm25_top_k or args.similarity_top_k
    hybrid_alpha = max(0.0, min(args.hybrid_alpha, 1.0))

    llm, retriever, query_engine = _create_llm_and_retriever(
        index,
        chat_provider=args.chat_provider,
        llm_model=args.llm_model,
        ollama_base_url=args.ollama_base_url,
        openai_api_key=args.openai_api_key,
        temperature=args.temperature,
        request_timeout=args.ollama_request_timeout,
        similarity_top_k=args.similarity_top_k,
        verbose=args.verbose,
        bm25_top_k=bm25_top_k,
        hybrid_alpha=hybrid_alpha,
    )

    responder = AgenticResponder(
        llm=llm,
        retriever=retriever,
        query_engine=query_engine,
        reasoning_system_prompt=reasoning_prompt,
        final_system_prompt=final_prompt,
        max_iterations=args.max_iterations,
        max_sources=args.max_sources,
        evidence_snippet_limit=min(args.max_sources, 5),
    )

    asyncio.run(
        _interactive_chat(
            responder,
            show_sources=args.show_sources,
        )
    )


if __name__ == "__main__":
    main()
