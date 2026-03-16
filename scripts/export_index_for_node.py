"""
Export FAISS index + metadata to index/node_index.json for the Node.js / Vercel app.
Run from project root with Python env that has faiss-cpu and access to config:
  python scripts/export_index_for_node.py

Output: index/node_index.json with shape { "chunks": [ { "e": [...], "text", "source", ... } ] }
Vercel deploy: if the file is > ~50MB, upload to Vercel Blob (or similar) and set INDEX_URL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import faiss
    import numpy as np
except ImportError:
    print("Install faiss-cpu and numpy: pip install faiss-cpu numpy")
    sys.exit(1)

from config.load_config import get_project_root, load_config

INDEX_DIR = get_project_root() / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.jsonl"
OUT_PATH = INDEX_DIR / "node_index.json"


def main():
    if not FAISS_PATH.exists():
        print(f"FAISS index not found: {FAISS_PATH}")
        print("Run the Python embed pipeline first to create the index.")
        sys.exit(1)
    if not META_PATH.exists():
        print(f"Metadata not found: {META_PATH}")
        sys.exit(1)

    index = faiss.read_index(str(FAISS_PATH))
    n, d = index.ntotal, index.d
    print(f"Index: {n} vectors, dim {d}")

    metadata = []
    with open(META_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if len(metadata) != n:
        print(f"Warning: metadata count {len(metadata)} != index size {n}")

    vectors = np.zeros((n, d), dtype="float32")
    for i in range(n):
        vectors[i] = index.reconstruct(i)

    chunks = []
    for i in range(n):
        meta = metadata[i] if i < len(metadata) else {}
        row = {"e": vectors[i].tolist()}
        for k, v in meta.items():
            if k not in ("e", "embedding"):
                row[k] = v
        chunks.append(row)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"dim": d, "chunks": chunks}, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")
    if size_mb > 45:
        print("Note: Vercel deployment limit is ~50MB. Consider setting INDEX_URL to a hosted copy (e.g. Vercel Blob).")


if __name__ == "__main__":
    main()
