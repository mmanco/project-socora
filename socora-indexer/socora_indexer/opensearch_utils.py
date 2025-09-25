from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import RequestError

DEFAULT_VECTOR_METHOD = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "lucene",
    "parameters": {"ef_construction": 256, "m": 48},
}
DEFAULT_INDEX_SETTINGS = {"index": {"knn": True, "knn.algo_param.ef_search": 100}}

logger = logging.getLogger(__name__)


def ensure_opensearch_index(
    endpoint: str,
    index_name: str,
    *,
    dimension: int,
    embedding_field: str = "embedding",
    text_field: str = "content",
    metadata_field: str = "metadata",
    vector_method: Optional[Dict[str, Any]] = None,
    index_settings: Optional[Dict[str, Any]] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
    recreate: bool = False,
) -> OpenSearch:
    """Ensure that an OpenSearch index with vector search capabilities exists."""

    kwargs = dict(client_kwargs or {})
    client = OpenSearch(endpoint, **kwargs)

    existing = bool(client.indices.exists(index=index_name))
    if existing and recreate:
        logger.info("Recreating OpenSearch index '%s'", index_name)
        client.indices.delete(index=index_name, ignore_unavailable=True)
        existing = False

    if existing:
        logger.debug("OpenSearch index '%s' already exists", index_name)
        return client

    method = dict(vector_method or DEFAULT_VECTOR_METHOD)
    settings = index_settings or DEFAULT_INDEX_SETTINGS

    def build_index_body(method_config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(method_config)
        return {
            "settings": settings,
            "mappings": {
                "dynamic": True,
                "dynamic_templates": [
                    {
                        "metadata_strings": {
                            "path_match": f"{metadata_field}.*",
                            "match_mapping_type": "string",
                            "mapping": {"type": "keyword"},
                        }
                    }
                ],
                "properties": {
                    "id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    embedding_field: {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": cfg,
                    },
                    text_field: {"type": "text"},
                    metadata_field: {"type": "object", "dynamic": True},
                },
            },
        }

    index_body = build_index_body(method)

    try:
        client.indices.create(index=index_name, body=index_body)
    except RequestError as exc:
        if method.get("engine") != "lucene" and _is_nmslib_deprecation_error(exc):
            fallback_method = dict(method)
            fallback_method["engine"] = "lucene"
            logger.info(
                "Retrying creation of OpenSearch index '%s' with vector engine 'lucene'",
                index_name,
            )
            fallback_body = build_index_body(fallback_method)
            client.indices.create(index=index_name, body=fallback_body)
            method = fallback_method
        else:
            raise

    client.indices.refresh(index=index_name)
    logger.info(
        "Created OpenSearch index '%s' with dimension %d using engine '%s'",
        index_name,
        dimension,
        method.get("engine", "unknown"),
    )

    return client


def _is_nmslib_deprecation_error(exc: RequestError) -> bool:
    return "nmslib engine is deprecated" in str(exc).lower()
