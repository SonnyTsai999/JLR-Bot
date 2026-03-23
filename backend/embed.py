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


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 300.0,
) -> "OpenAI":
    if not HAS_OPENAI:
        raise RuntimeError("openai is required. Install with: pip install openai")
    kwargs: dict = {"api_key": api_key, "timeout": timeout}
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
    model: str = "text-embedding-3-small",
    client: "OpenAI | None" = None,
    api_key: str | None = None,
    batch_size: int = 100,
    timeout: float = 300.0,
) -> list[list[float]]:
    """Get OpenAI embeddings for a list of texts. Uses timeout to avoid hanging."""
    if not client:
        client = get_openai_client(api_key=api_key, timeout=timeout)
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
    model = emb_cfg.get("model", "text-embedding-3-small")
    batch_size = max(1, int(emb_cfg.get("batch_size", 1)))

    texts = [c["text"] for c in chunks]
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url") or None
    timeout = float(emb_cfg.get("timeout_seconds", 300))
    client = get_openai_client(api_key=api_key, base_url=base_url, timeout=timeout)
    vectors = embed_texts(texts, model=model, client=client, api_key=api_key, batch_size=batch_size, timeout=timeout)

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


def check_index_health(
    project_root: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Verify FAISS index and metadata are consistent and resumable.
    Returns dict with: ok (bool), message (str), index_vectors, metadata_rows, chunks_total, pending.
    """
    config = config or load_config()
    project_root = project_root or get_project_root()
    index_dir = get_index_dir(project_root, config)
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"

    out: dict[str, Any] = {
        "ok": False,
        "message": "",
        "index_vectors": 0,
        "metadata_rows": 0,
        "chunks_total": 0,
        "pending": 0,
    }

    if not idx_path.exists():
        out["message"] = f"Index file missing: {idx_path}"
        return out
    if not meta_path.exists():
        out["message"] = f"Metadata file missing: {meta_path}"
        return out

    all_chunks = load_chunks(project_root, config)
    out["chunks_total"] = len(all_chunks)

    if not HAS_FAISS:
        out["message"] = "faiss-cpu not installed; cannot verify vector count."
        return out

    try:
        index = faiss.read_index(str(idx_path))
        out["index_vectors"] = index.ntotal
    except Exception as e:
        out["message"] = f"FAISS index unreadable: {e}"
        return out

    metadata = []
    try:
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
    except Exception as e:
        out["message"] = f"Metadata file unreadable: {e}"
        return out

    out["metadata_rows"] = len(metadata)

    if out["index_vectors"] != out["metadata_rows"]:
        out["message"] = (
            f"Index inconsistent: faiss.index has {out['index_vectors']} vectors but metadata.jsonl has {out['metadata_rows']} rows. "
            "Re-build the index or fix files manually."
        )
        return out

    embedded_ids = {m.get("chunk_id") for m in metadata if m.get("chunk_id")}
    pending = [c for c in all_chunks if c.get("chunk_id") not in embedded_ids]
    out["pending"] = len(pending)

    out["ok"] = True
    out["message"] = (
        f"Index healthy: {out['index_vectors']} vectors, {out['metadata_rows']} metadata rows, "
        f"{out['chunks_total']} total chunks in source, {out['pending']} pending."
    )
    return out


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


def run_embed(
    config: dict[str, Any] | None = None,
    api_key: str | None = None,
    skip_health_check: bool = False,
) -> None:
    """
    Build or resume FAISS index. Saves after every chunk — safe to stop (Ctrl+C); re-run to resume.
    If embedding.max_chunks_per_run is set in config, only that many new chunks are embedded per run.
    """
    config = config or load_config()
    project_root = get_project_root()

    existing_index_dir = get_index_dir(project_root, config)
    idx_path = existing_index_dir / "faiss.index"
    meta_path = existing_index_dir / "metadata.jsonl"

    if idx_path.exists() or meta_path.exists():
        health = check_index_health(project_root, config)
        if not health["ok"]:
            print("Index health check failed:", health["message"])
            print("Fix the index before proceeding (e.g. remove index/faiss.index and index/metadata.jsonl to start fresh).")
            raise SystemExit("Check the health of the index; cannot proceed now.")
        if not skip_health_check:
            print("Index health:", health["message"])

    key = api_key or get_api_key()
    if not key or not key.strip():
        raise SystemExit(
            "API key not set. Set OPENAI_API_KEY in the environment or llm.api_key in config/settings.yaml."
        )
    all_chunks = load_chunks(project_root, config)
    if not all_chunks:
        raise SystemExit("No chunks found. Run ingest first (backend/ingest.py).")

    emb_cfg = config.get("embedding", {})
    max_per_run = int(emb_cfg.get("max_chunks_per_run", 0))
    max_chunk_chars = int(emb_cfg.get("max_chunk_chars", 20000))
    # If True: any PDF with at least one oversized chunk drops ALL its chunks (legacy; very aggressive).
    skip_abnormal_sources = bool(emb_cfg.get("skip_abnormal_sources", False))
    # When not skipping whole sources: "truncate" = embed first max_chunk_chars (default); "skip" = drop that chunk only.
    on_oversized_chunk = str(emb_cfg.get("on_oversized_chunk", "truncate")).strip().lower()
    model = emb_cfg.get("model", "text-embedding-3-small")
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

    skipped_abnormal: list[dict[str, Any]] = []
    truncated_count = 0

    if skip_abnormal_sources:
        abnormal_sources = {
            c.get("source")
            for c in pending
            if c.get("source") and len(c.get("text", "")) > max_chunk_chars
        }
        if abnormal_sources:
            kept = []
            for c in pending:
                if c.get("source") in abnormal_sources:
                    skipped_abnormal.append(c)
                else:
                    kept.append(c)
            pending = kept
            print(
                f"Skipping {len(skipped_abnormal)} chunks from {len(abnormal_sources)} abnormal source PDFs "
                f"(chunk length > {max_chunk_chars} chars). "
                f"Tip: set embedding.skip_abnormal_sources: false and use on_oversized_chunk: truncate instead."
            )
            sample = list(sorted(abnormal_sources))[:5]
            for s in sample:
                print(f"  - skipped source: {s}")
            if len(abnormal_sources) > len(sample):
                print(f"  ... and {len(abnormal_sources) - len(sample)} more source(s).")
    else:
        kept = []
        for c in pending:
            text = c.get("text") or ""
            if len(text) <= max_chunk_chars:
                kept.append(c)
                continue
            if on_oversized_chunk == "skip":
                skipped_abnormal.append(c)
            else:
                # Default and "truncate": cap length so embedding APIs stay within limits.
                nc = dict(c)
                nc["text"] = text[:max_chunk_chars]
                nc["_embed_text_truncated"] = True
                nc["_embed_text_orig_len"] = len(text)
                kept.append(nc)
                truncated_count += 1
        pending = kept
        if skipped_abnormal:
            print(
                f"Skipping {len(skipped_abnormal)} oversized chunks "
                f"(each > {max_chunk_chars} chars; on_oversized_chunk=skip)."
            )
        if truncated_count:
            print(
                f"Truncated {truncated_count} chunk(s) to {max_chunk_chars} chars for embedding "
                f"(metadata stores truncated text; re-chunk PDFs in ingest for full coverage)."
            )

    if not pending:
        print("No embeddable chunks remain after abnormal-PDF filtering.")
        return

    if max_per_run > 0:
        pending = pending[:max_per_run]
        print(f"Limiting to {max_per_run} chunks this run ({len(pending)} to process).")

    total_chunks = len(all_chunks)
    pct0 = 100 * len(metadata) / total_chunks if total_chunks else 0
    print(f"Progress: {len(metadata)}/{total_chunks} ({pct0:.1f}%) — processing one chunk at a time.")
    print()

    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url") or None
    timeout = float(emb_cfg.get("timeout_seconds", 300))
    client = get_openai_client(api_key=key, base_url=base_url, timeout=timeout)

    total_this_run = len(pending)
    failures: list[tuple[str, str, str]] = []
    for processed, chunk in enumerate(pending, 1):
        chunk_id = chunk.get("chunk_id") or f"chunk_{processed}"
        print(f"  Chunk {processed}/{total_this_run} this run: calling API for {chunk_id}...", flush=True)
        try:
            vectors = embed_texts(
                [chunk["text"]], model=model, client=client, api_key=key, batch_size=1, timeout=timeout
            )
        except Exception as e:
            source = chunk.get("source") or ""
            failures.append((chunk_id, source, str(e)))
            print(f"  Failed on {chunk_id}: {e} (skipping and continuing)", flush=True)
            continue

        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)

        if index is None:
            index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        metadata.append(chunk)
        save_index(index, metadata, project_root, config)

        n = len(metadata)
        pct = 100 * n / total_chunks
        print(f"  Chunk {n}/{total_chunks} ({pct:.1f}%) — saved.", flush=True)

    print()
    if max_per_run > 0 and total_this_run >= max_per_run:
        print(f"Done. Index has {len(metadata)} chunks. Re-run to continue embedding more.")
    else:
        print(f"Done. Index has {len(metadata)} chunks.")
    if failures:
        print(f"Skipped {len(failures)} failed chunk(s) this run.")
        for cid, src, err in failures[:8]:
            label = f"{cid}" if not src else f"{cid} [{src}]"
            print(f"  - {label}: {err}")
        if len(failures) > 8:
            print(f"  ... and {len(failures) - 8} more failure(s).")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build or resume FAISS index from processed chunks.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only run index health check and exit (exit 0 if healthy, 1 otherwise).",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip printing health check when resuming (embedding only).",
    )
    args = parser.parse_args()

    if args.check:
        config = load_config()
        project_root = get_project_root()
        health = check_index_health(project_root, config)
        print(health["message"])
        raise SystemExit(0 if health["ok"] else 1)

    run_embed(api_key=get_api_key(), skip_health_check=args.skip_health_check)


if __name__ == "__main__":
    main()
