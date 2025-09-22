from __future__ import annotations

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from socora_indexer.content_indexer import build_content_index_async


def _coerce_option_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw.strip()


def _parse_key_value_pairs(pairs: list[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid key-value pair '{item}'. Expected format key=value."
            )
        key, value = item.split("=", 1)
        parsed[key.strip()] = _coerce_option_value(value)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and push Socora content indexes into OpenSearch.",
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Crawler run directory with page folders.")
    parser.add_argument("--opensearch-endpoint", required=True, help="OpenSearch endpoint URL (e.g. https://host:9200)")
    parser.add_argument("--opensearch-index", required=True, help="OpenSearch index name to write embeddings into.")
    parser.add_argument("--opensearch-username", help="Optional OpenSearch basic-auth username.")
    parser.add_argument("--opensearch-password", help="Optional OpenSearch basic-auth password.")
    parser.add_argument(
        "--opensearch-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra opensearch client options (e.g. verify_certs=false).",
    )
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama host for embeddings.")
    parser.add_argument("--ollama-model", default="nomic-embed-text", help="Ollama embedding model id.")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="Batch size for embedding calls.")
    parser.add_argument(
        "--semantic-buffer-size",
        type=int,
        default=2,
        help="Neighbouring sentence window when computing semantic chunks.",
    )
    parser.add_argument(
        "--semantic-breakpoint-percentile",
        type=int,
        default=92,
        help="Percentile threshold controlling semantic split sensitivity.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Embedding dimensionality expected by the OpenSearch index.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata to attach to every node.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Console log level.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    opensearch_options = _parse_key_value_pairs(args.opensearch_option)
    additional_metadata = _parse_key_value_pairs(args.metadata)

    await build_content_index_async(
        run_dir=args.run_dir,
        opensearch_endpoint=args.opensearch_endpoint,
        opensearch_index=args.opensearch_index,
        opensearch_username=args.opensearch_username,
        opensearch_password=args.opensearch_password,
        opensearch_client_kwargs=opensearch_options or None,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        embed_batch_size=args.embed_batch_size,
        semantic_buffer_size=args.semantic_buffer_size,
        semantic_breakpoint_percentile=args.semantic_breakpoint_percentile,
        embedding_dimension=args.embedding_dimension,
        additional_metadata=additional_metadata or None,
    )

if __name__ == "__main__":
    asyncio.run(main())
