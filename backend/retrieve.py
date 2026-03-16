"""
Retrieval module for JLR Technology Intelligence Assistant.
Top-K semantic search with metadata filtering; caps chunks per source to reduce redundancy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.load_config import get_project_root, load_config

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> "OpenAI":
    if not HAS_OPENAI:
        raise RuntimeError("openai is required. Install with: pip install openai")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def load_index_and_metadata(project_root: Path, config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    if not HAS_FAISS:
        raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu")
    paths = config.get("paths", {})
    index_dir = project_root / paths.get("index_dir", "index")
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata = []
    with open(index_dir / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return index, metadata


def embed_query(
    query: str,
    model: str = "text-embedding-3-large",
    client: "OpenAI | None" = None,
    api_key: str | None = None,
) -> list[float]:
    if not client:
        client = get_openai_client(api_key=api_key)
    resp = client.embeddings.create(input=[query], model=model)
    return resp.data[0].embedding


def apply_metadata_filter(
    metadata: list[dict[str, Any]],
    technology_L1: str | None = None,
    technology_L2: str | None = None,
    lifecycle_stage: str | None = None,
) -> set[int]:
    """Return set of indices that pass the filter. Empty filter = all pass."""
    allowed = set(range(len(metadata)))
    if technology_L1:
        allowed &= {i for i, m in enumerate(metadata) if (m.get("technology_L1") or "").lower() == technology_L1.lower()}
    if technology_L2:
        allowed &= {i for i, m in enumerate(metadata) if (m.get("technology_L2") or "").lower() == technology_L2.lower()}
    if lifecycle_stage:
        allowed &= {i for i, m in enumerate(metadata) if (m.get("lifecycle_stage") or "").lower() == lifecycle_stage.lower()}
    return allowed


def retrieve(
    query: str,
    config: dict[str, Any] | None = None,
    *,
    top_k: int | None = None,
    technology_L1: str | None = None,
    technology_L2: str | None = None,
    lifecycle_stage: str | None = None,
    max_chunks_per_source: int | None = None,
    min_similarity_threshold: float | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run Top-K retrieval with optional metadata filter and per-source cap.
    Returns list of chunk dicts (with 'text' and metadata), ordered by relevance.
    """
    if not HAS_FAISS or not HAS_OPENAI:
        raise RuntimeError("faiss-cpu and openai are required for retrieval.")
    config = config or load_config()
    project_root = get_project_root()
    ret_cfg = config.get("retrieval", {})
    emb_cfg = config.get("embedding", {})

    k = top_k if top_k is not None else ret_cfg.get("top_k", 10)
    max_per_source = max_chunks_per_source if max_chunks_per_source is not None else ret_cfg.get("max_chunks_per_source", 3)
    threshold = min_similarity_threshold if min_similarity_threshold is not None else ret_cfg.get("min_similarity_threshold", 0.0)

    index, metadata = load_index_and_metadata(project_root, config)
    allowed_indices = apply_metadata_filter(metadata, technology_L1, technology_L2, lifecycle_stage)
    if not allowed_indices:
        return []

    base_url = config.get("llm", {}).get("base_url") or None
    client = get_openai_client(api_key=api_key, base_url=base_url)
    model = emb_cfg.get("model", "text-embedding-3-large")
    qvec = embed_query(query, model=model, client=client, api_key=api_key)
    q = np.array([qvec], dtype="float32")
    faiss.normalize_L2(q)

    # Search more than k to allow for per-source capping
    search_k = min(len(metadata), max(k * 3, 50))
    scores, indices = index.search(q, search_k)

    # Filter by allowed indices and threshold; cap per source
    source_count: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx not in allowed_indices:
            continue
        if threshold and float(score) < threshold:
            continue
        meta = metadata[idx].copy()
        source = meta.get("source", "")
        if source_count.get(source, 0) >= max_per_source:
            continue
        source_count[source] = source_count.get(source, 0) + 1
        meta["_score"] = float(score)
        results.append(meta)
        if len(results) >= k:
            break

    return results


if __name__ == "__main__":
    import os
    from config.load_config import get_api_key
    config = load_config()
    out = retrieve("What are adoption barriers for digital twins in AEC?", config=config, api_key=get_api_key())
    for r in out:
        print(r.get("chunk_id"), r.get("_score"), r.get("title"))
