from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx


class TikaError(RuntimeError):
    pass


def extract_with_tika(
    file_path: Path,
    server_url: str,
    timeout: float = 30.0,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Send the file to an Apache Tika server and return (text, metadata_dict).

    - server_url should be like "http://localhost:9998".
    - Returns (None, None) on non-fatal issues (e.g., 415 unsupported type).
    - Raises TikaError for connectivity issues.
    """
    base = server_url.rstrip("/")
    text: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    try:
        with httpx.Client(timeout=timeout) as client:
            with file_path.open("rb") as f:
                r = client.put(f"{base}/tika", content=f, headers={"Accept": "text/plain"})
            if r.status_code == 200:
                text = r.text
            elif r.status_code in (204, 415):
                text = None
            else:
                raise TikaError(f"Tika text extract failed: {r.status_code} {r.text[:200]}")

            with file_path.open("rb") as f:
                r2 = client.put(f"{base}/meta", content=f, headers={"Accept": "application/json"})
            if r2.status_code == 200:
                try:
                    meta = r2.json()
                except json.JSONDecodeError:
                    meta = None
            elif r2.status_code in (204, 415):
                meta = None
            else:
                raise TikaError(f"Tika metadata failed: {r2.status_code} {r2.text[:200]}")
    except httpx.RequestError as e:
        raise TikaError(f"Could not reach Tika server at {server_url}: {e}") from e

    return text, meta

