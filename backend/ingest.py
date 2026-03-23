"""
PDF ingestion pipeline for JLR Technology Intelligence Assistant.
Extracts text, cleans (no references/appendices), chunks with overlap, attaches metadata.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config.load_config import get_project_root, load_config

# Optional: use pypdf for extraction; fallback to minimal extractor if not installed
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Approximate tokens (4 chars per token for English)
CHARS_PER_TOKEN = 4


def _ensure_dirs(project_root: Path, config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    (project_root / paths.get("raw_pdfs", "data/raw_pdfs")).mkdir(parents=True, exist_ok=True)
    (project_root / paths.get("processed_chunks", "data/processed_chunks")).mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF. Requires pypdf."""
    if not HAS_PYPDF:
        raise RuntimeError("pypdf is required for PDF extraction. Install with: pip install pypdf")
    if pdf_path.stat().st_size == 0:
        raise ValueError(f"Empty file: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def clean_text(raw: str) -> str:
    """
    Remove reference sections, appendices, normalize whitespace.
    Preserve paragraph boundaries.
    """
    text = raw
    # Remove common reference section headers and everything after
    ref_patterns = [
        r"\n\s*References?\s*\n.*",
        r"\n\s*Bibliography\s*\n.*",
        r"\n\s*REFERENCES\s*\n.*",
        r"\n\s*Appendix\s+[A-Za-z0-9]*\s*\n.*",
        r"\n\s*Appendices\s*\n.*",
    ]
    for pat in ref_patterns:
        text = re.sub(pat, "\n", text, flags=re.DOTALL | re.IGNORECASE)
    # Normalize whitespace: collapse runs, keep single newlines between paragraphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_estimate(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    token_target_min: int = 500,
    token_target_max: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Split text into chunks of 500–800 tokens with 100-token overlap.
    Does not break mid-sentence when possible.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_target = overlap_tokens

    for para in paragraphs:
        para_tokens = token_estimate(para)
        if current_tokens + para_tokens > token_target_max and current:
            chunk_text_val = "\n\n".join(current)
            chunks.append(chunk_text_val)
            # Overlap: keep last paragraphs that fit in overlap_tokens
            overlap_so_far = 0
            keep: list[str] = []
            for p in reversed(current):
                t = token_estimate(p)
                if overlap_so_far + t <= overlap_target:
                    keep.append(p)
                    overlap_so_far += t
                else:
                    break
            current = list(reversed(keep))
            current_tokens = sum(token_estimate(p) for p in current)
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def load_document_metadata(pdf_path: Path, project_root: Path) -> dict[str, Any]:
    """
    Load per-document metadata from a JSON sidecar next to the PDF.
    Example: paper.pdf -> paper.metadata.json
    Keys: title, year, technology_L1, technology_L2, lifecycle_stage, apa_citation, doi
    """
    sidecar = pdf_path.with_suffix(".metadata.json")
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            return json.load(f)
    # Defaults from filename (technology_* and lifecycle_stage not required)
    stem = pdf_path.stem
    return {
        "title": stem.replace("_", " ").replace("-", " "),
        "year": None,
        "authors": "",
        "apa_citation": "",
        "doi": "",
    }


def ingest_pdf(
    pdf_path: Path,
    metadata_override: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Process one PDF into chunks with metadata.
    Returns list of chunk dicts with keys: text, source, title, year, technology_L1,
    technology_L2, lifecycle_stage, apa_citation, doi, chunk_id.
    """
    config = config or load_config()
    project_root = get_project_root()
    meta = load_document_metadata(pdf_path, project_root)
    if metadata_override:
        meta.update(metadata_override)

    raw = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw)
    chunking_cfg = config.get("chunking", {})
    text_chunks = chunk_text(
        cleaned,
        token_target_min=chunking_cfg.get("token_target_min", 500),
        token_target_max=chunking_cfg.get("token_target_max", 800),
        overlap_tokens=chunking_cfg.get("overlap_tokens", 100),
    )

    result = []
    for i, t in enumerate(text_chunks):
        result.append({
            "text": t,
            "source": pdf_path.name,
            "title": meta.get("title") or pdf_path.stem,
            "year": meta.get("year"),
            "authors": meta.get("authors") or "",
            "apa_citation": meta.get("apa_citation") or "",
            "doi": meta.get("doi") or "",
            "chunk_id": f"{pdf_path.stem}_{i}",
        })
    return result


def _load_existing_chunks(out_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing chunks.jsonl if present. Returns (chunks list, set of source filenames)."""
    out_path = out_dir / "chunks.jsonl"
    if not out_path.exists():
        return [], set()
    chunks = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    ingested_sources = {c.get("source", "") for c in chunks if c.get("source")}
    return chunks, ingested_sources


def _pdf_paths_in_dir(raw_dir: Path) -> list[Path]:
    """All .pdf files (case-insensitive on Windows)."""
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(raw_dir.iterdir()) if raw_dir.is_dir() else []:
        if not p.is_file():
            continue
        if p.suffix.lower() != ".pdf":
            continue
        key = p.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def run_ingest(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Run ingestion on all PDFs under paths.raw_pdfs (see config/settings.yaml).
    Saves processed chunks to data/processed_chunks as JSONL (one JSON object per line).
    Resumable: loads existing chunks and skips PDFs already ingested (by filename).

    Important: copying PDFs into the folder is not enough — you must run this script
    before `python -m backend.embed`. Embed only reads chunks.jsonl, not raw_pdfs.
    """
    config = config or load_config()
    project_root = get_project_root()
    _ensure_dirs(project_root, config)
    paths = config.get("paths", {})
    raw_dir = project_root / paths.get("raw_pdfs", "data/raw_pdfs")
    out_dir = project_root / paths.get("processed_chunks", "data/processed_chunks")

    print(f"Configured PDF folder: {raw_dir.resolve()}")
    alt_root_raw = project_root / "raw_pdfs"
    if alt_root_raw.is_dir() and alt_root_raw.resolve() != raw_dir.resolve():
        alt_pdfs = _pdf_paths_in_dir(alt_root_raw)
        if alt_pdfs:
            print(
                f"NOTE: Found {len(alt_pdfs)} PDF(s) in {alt_root_raw.resolve()} — "
                f"that path is NOT used. Either move them to the folder above, or set "
                f"paths.raw_pdfs: \"raw_pdfs\" in config/settings.yaml."
            )

    # Resume: keep existing chunks and skip already-ingested sources
    all_chunks, ingested_sources = _load_existing_chunks(out_dir)
    if ingested_sources:
        print(f"Resuming: {len(all_chunks)} existing chunks from {len(ingested_sources)} ingested PDF(s).")

    pdf_paths = _pdf_paths_in_dir(raw_dir)
    print(f"Found {len(pdf_paths)} PDF file(s) in configured folder.")
    to_ingest = [p for p in pdf_paths if p.name not in ingested_sources]
    if to_ingest:
        print(f"Will ingest {len(to_ingest)} new PDF(s); skipping {len(pdf_paths) - len(to_ingest)} already in chunks.jsonl.")
    elif pdf_paths:
        print("No new PDF filenames — all are already in chunks.jsonl. To replace a PDF, remove its lines from chunks.jsonl or rename the file.")

    for pdf_path in pdf_paths:
        if pdf_path.name in ingested_sources:
            print(f"Skipping (already ingested): {pdf_path.name}")
            continue
        try:
            chunks = ingest_pdf(pdf_path, config=config)
            all_chunks.extend(chunks)
            ingested_sources.add(pdf_path.name)
        except Exception as e:
            print(f"Skipping {pdf_path.name}: {e}")
            continue

    out_path = out_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in all_chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Ingest complete: {len(all_chunks)} chunks from {len(ingested_sources)} PDF(s).")
    return all_chunks


if __name__ == "__main__":
    run_ingest()
