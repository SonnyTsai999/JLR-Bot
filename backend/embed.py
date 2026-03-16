"""
Embedding and FAISS index build for JLR Technology Intelligence Assistant.
Reads processed chunks, calls OpenAI embeddings, builds and saves FAISS index.
Resumable: set embedding.max_chunks_per_run (e.g. 500) to limit cost per run; re-run to continue.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.load_config import get_project_root, load_config, get_api_key

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    np = None

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


def load_chunks(project_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config.get("paths", {})
    chunks_path = project_root / paths.get("processed_chunks", "data/processed_chunks") / "chunks.jsonl"
    if not chunks_path.exists():
        return []
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-large",
    client: "OpenAI | None" = None,
    api_key: str | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """Get OpenAI embeddings for a list of texts."""
    if not client:
        client = get_openai_client(api_key=api_key)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for e in resp.data:
            all_embeddings.append(e.embedding)
    return all_embeddings


def build_index(
    chunks: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """
    Build FAISS index from chunks. Returns (faiss_index, metadata_list).
    metadata_list[i] corresponds to the i-th vector (chunk metadata for retrieval).
    """
    if not HAS_FAISS or np is None:
        raise RuntimeError("faiss-cpu and numpy are required. Install with: pip install faiss-cpu numpy")
    config = config or load_config()
    emb_cfg = config.get("embedding", {})
    model = emb_cfg.get("model", "text-embedding-3-large")
    batch_size = max(1, int(emb_cfg.get("batch_size", 100)))

    texts = [c["text"] for c in chunks]
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url") or None
    client = get_openai_client(api_key=api_key, base_url=base_url)
    vectors = embed_texts(texts, model=model, client=client, api_key=api_key, batch_size=batch_size)

    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    faiss.normalize_L2(matrix)
    index.add(matrix)

    return index, chunks


def get_index_dir(project_root: Path, config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    index_dir = project_root / paths.get("index_dir", "index")
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def load_existing_index(index_dir: Path) -> tuple[Any, list[dict[str, Any]]] | None:
    """Load existing FAISS index and metadata if present. Returns None if not found."""
    if not HAS_FAISS:
        return None
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        return None
    index = faiss.read_index(str(idx_path))
    metadata = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return index, metadata


def save_index(index: Any, metadata: list[dict[str, Any]], project_root: Path, config: dict[str, Any]) -> None:
    index_dir = get_index_dir(project_root, config)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def load_index(project_root: Path, config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
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


def run_embed(config: dict[str, Any] | None = None, api_key: str | None = None) -> None:
    """
    Build or resume FAISS index. Saves after every batch — safe to stop (Ctrl+C); re-run to resume.
    If embedding.max_chunks_per_run is set in config, only that many new chunks are embedded per run.
    """
    config = config or load_config()
    key = api_key or get_api_key()
    if not key or not key.strip():
        raise SystemExit(
            "API key not set. Set OPENAI_API_KEY in the environment or llm.api_key in config/settings.yaml."
        )
    project_root = get_project_root()
    all_chunks = load_chunks(project_root, config)
    if not all_chunks:
        raise SystemExit("No chunks found. Run ingest first (backend/ingest.py).")

    emb_cfg = config.get("embedding", {})
    batch_size = max(1, int(emb_cfg.get("batch_size", 50)))
    max_per_run = int(emb_cfg.get("max_chunks_per_run", 0))
    model = emb_cfg.get("model", "text-embedding-3-large")
    index_dir = get_index_dir(project_root, config)

    existing = load_existing_index(index_dir)
    if existing:
        index, metadata = existing
        embedded_ids = {m.get("chunk_id") for m in metadata if m.get("chunk_id")}
        pending = [c for c in all_chunks if c.get("chunk_id") not in embedded_ids]
        print(f"Resuming: {len(metadata)} chunks already in index, {len(pending)} pending.")
    else:
        index = None
        metadata = []
        pending = list(all_chunks)
        print(f"New index: {len(pending)} chunks to embed.")

    if not pending:
        print("All chunks already embedded. Nothing to do.")
        return

    if max_per_run > 0:
        pending = pending[:max_per_run]
        print(f"Limiting to {max_per_run} chunks this run ({len(pending)} to process).")

    total_chunks = len(all_chunks)
    num_batches = (len(pending) + batch_size - 1) // batch_size
    pct0 = 100 * len(metadata) / total_chunks if total_chunks else 0
    print(f"Progress: {len(metadata)}/{total_chunks} ({pct0:.1f}%) — batches of {batch_size}.")
    print()

    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url") or None
    client = get_openai_client(api_key=key, base_url=base_url)

    processed = 0
    for batch_num, i in enumerate(range(0, len(pending), batch_size), 1):
        batch_chunks = pending[i : i + batch_size]
        texts = [c["text"] for c in batch_chunks]
        vectors = embed_texts(texts, model=model, client=client, api_key=key, batch_size=batch_size)
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)

        if index is None:
            index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        metadata.extend(batch_chunks)
        save_index(index, metadata, project_root, config)
        processed += len(batch_chunks)
        pct = 100 * len(metadata) / total_chunks
        print(f"  Batch {batch_num}/{num_batches}: {processed}/{len(pending)} this run — {len(metadata)}/{total_chunks} total ({pct:.1f}%) — saved.")

    print()
    if max_per_run > 0 and len(pending) >= max_per_run:
        print(f"Done. Index has {len(metadata)} chunks. Re-run to continue embedding more.")
    else:
        print(f"Done. Index has {len(metadata)} chunks.")


if __name__ == "__main__":
    run_embed(api_key=get_api_key())
