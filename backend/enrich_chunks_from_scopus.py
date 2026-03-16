"""
Enrich chunks.jsonl with metadata from a Scopus export CSV.
Matches each chunk's source PDF to a Scopus row (by title/year or DOI), then fills
authors, title, year, doi, and a proper APA citation. Does not add technology_L1,
technology_L2, or lifecycle_stage (left empty).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from config.load_config import get_project_root, load_config


def _normalize_for_match(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces for fuzzy matching."""
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stem_to_title_like(stem: str) -> str:
    """Convert PDF stem like '100_Modular Construction_ A Comprehensive Review_2025' to title-like."""
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"_\d{4}$", "", stem)
    return stem.replace("_", " ").strip()


def load_scopus_records(csv_path: Path) -> list[dict[str, Any]]:
    """Load Scopus export CSV into list of dicts."""
    records = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def build_scopus_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build lookup by normalized title+year and by DOI."""
    by_key: dict[str, dict[str, Any]] = {}
    by_doi: dict[str, dict[str, Any]] = {}
    for r in records:
        title = (r.get("Title") or "").strip()
        year = (r.get("Year") or "").strip()
        doi = (r.get("DOI") or "").strip()
        if title:
            key = _normalize_for_match(title) + " " + year
            by_key[key] = r
        if doi:
            by_doi[_normalize_for_match(doi)] = r
            by_doi[doi.lower().replace(" ", "")] = r
    return {"by_title_year": by_key, "by_doi": by_doi}


def format_apa(scopus: dict[str, Any]) -> str:
    """Format a single Scopus record as APA 7 journal article."""
    authors = (scopus.get("Authors") or "").strip()
    year = (scopus.get("Year") or "").strip()
    title = (scopus.get("Title") or "").strip()
    source = (scopus.get("Source title") or "").strip()
    vol = (scopus.get("Volume") or "").strip()
    issue = (scopus.get("Issue") or "").strip()
    art_no = (scopus.get("Art. No.") or "").strip()
    page_start = (scopus.get("Page start") or "").strip()
    page_end = (scopus.get("Page end") or "").strip()
    doi = (scopus.get("DOI") or "").strip()

    parts = []
    if authors:
        parts.append(authors.rstrip(".") + ".")
    if year:
        parts.append(f"({year}).")
    if title:
        parts.append(title + ".")
    if source:
        vol_issue = vol
        if vol and issue:
            vol_issue = f"{vol}({issue})"
        elif not vol_issue:
            vol_issue = ""
        page_range = ""
        if page_start:
            page_range = f"{page_start}-{page_end}" if page_end else page_start
        elif art_no:
            page_range = art_no
        if vol_issue and page_range:
            parts.append(f"{source}, {vol_issue}, {page_range}.")
        elif vol_issue:
            parts.append(f"{source}, {vol_issue}.")
        else:
            parts.append(f"{source}.")
    if doi:
        if not doi.startswith("http"):
            doi = re.sub(r"^https?://doi\.org/", "", doi).strip()
            parts.append(f"https://doi.org/{doi}")
        else:
            parts.append(doi)
    return " ".join(parts)


def extract_doi_from_text(text: str) -> str | None:
    """Extract DOI from chunk text."""
    if not text:
        return None
    m = re.search(r"https?://doi\.org/([^\s\]\)]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".,")
    m = re.search(r"\b10\.\d{4,}/[^\s\]\)]+", text)
    if m:
        return m.group(0).rstrip(".,")
    return None


def match_chunk_to_scopus(
    chunk: dict[str, Any],
    by_title_year: dict[str, dict[str, Any]],
    by_doi: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a Scopus record for this chunk by title/year or DOI in text."""
    source = chunk.get("source") or ""
    stem = Path(source).stem
    title_like = _stem_to_title_like(stem)
    norm_title = _normalize_for_match(title_like)
    year = chunk.get("year") or ""
    if not year:
        y = re.search(r"_20\d{2}", stem)
        if y:
            year = y.group(0).lstrip("_")

    key = norm_title + " " + str(year)
    if key in by_title_year:
        return by_title_year[key]
    if len(norm_title) >= 15:
        for k, rec in by_title_year.items():
            if norm_title[:25] in k or k[:25] in norm_title:
                return rec

    doi = extract_doi_from_text(chunk.get("text") or "")
    if doi:
        doi_norm = _normalize_for_match(doi)
        if doi_norm in by_doi:
            return by_doi[doi_norm]
        for d, rec in by_doi.items():
            if doi in d or d in doi:
                return rec

    return None


def run_enrich(
    scopus_csv_path: Path | str | None = None,
    chunks_path: Path | None = None,
    config: dict[str, Any] | None = None,
    out_path: Path | None = None,
    keep_unmatched: bool = False,
) -> int:
    """
    Enrich chunks with Scopus metadata (authors, title, year, doi, apa_citation).
    If keep_unmatched is False (default), chunks that did not match any Scopus record are removed.
    Re-run embed after enriching with keep_unmatched=False to rebuild the index.
    """
    config = config or load_config()
    project_root = get_project_root()
    paths_cfg = config.get("paths", {})
    chunks_dir = project_root / paths_cfg.get("processed_chunks", "data/processed_chunks")
    default_chunks = chunks_dir / "chunks.jsonl"

    chunks_path = chunks_path or default_chunks
    chunks_path = Path(chunks_path)
    out_path = Path(out_path) if out_path else chunks_path

    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}")
        return 0

    scopus_path = None
    if scopus_csv_path and Path(scopus_csv_path).exists():
        scopus_path = Path(scopus_csv_path)
    if not scopus_path and paths_cfg.get("scopus_export_csv"):
        p = Path(paths_cfg["scopus_export_csv"])
        if p.exists():
            scopus_path = p
    if not scopus_path:
        scopus_path = project_root.parent / "Screening" / "scopus_export_2015-2025_901.csv"
    if not scopus_path.exists():
        print(f"Scopus CSV not found: {scopus_path}")
        return 0

    print(f"Loading Scopus from {scopus_path} ...")
    records = load_scopus_records(scopus_path)
    lookup = build_scopus_lookup(records)
    print(f"Loaded {len(records)} Scopus records.")

    print(f"Loading chunks from {chunks_path} ...")
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks.")

    updated = 0
    for c in chunks:
        scopus = match_chunk_to_scopus(c, lookup["by_title_year"], lookup["by_doi"])
        if scopus:
            c["authors"] = (scopus.get("Authors") or "").strip()
            c["title"] = (scopus.get("Title") or "").strip()
            c["year"] = (scopus.get("Year") or "").strip() or None
            c["doi"] = (scopus.get("DOI") or "").strip()
            c["apa_citation"] = format_apa(scopus)
            updated += 1
        c.setdefault("technology_L1", "")
        c.setdefault("technology_L2", "")
        c.setdefault("lifecycle_stage", "")
        c.setdefault("authors", "")

    if not keep_unmatched:
        original_count = len(chunks)
        chunks = [c for c in chunks if (c.get("apa_citation") or "").strip()]
        removed = original_count - len(chunks)
        if removed:
            print(f"Removed {removed} chunks with no Scopus match (keeping {len(chunks)} enriched chunks).")

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Update index metadata only if we kept all chunks (same count); otherwise user must re-embed
    index_meta = project_root / paths_cfg.get("index_dir", "index") / "metadata.jsonl"
    if index_meta.exists():
        with open(index_meta, encoding="utf-8") as f:
            index_count = sum(1 for _ in f)
        if index_count == len(chunks):
            with open(index_meta, "w", encoding="utf-8") as f:
                for obj in chunks:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"Updated {index_meta} with enriched metadata (no re-embed needed).")
        else:
            print(f"Index has {index_count} entries, chunks now {len(chunks)}. Re-run embed to rebuild index.")

    print(f"Enriched {updated} chunks total. Wrote {len(chunks)} chunks to {out_path}.")
    return updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enrich chunks from Scopus export; remove unmatched by default.")
    parser.add_argument("scopus_csv", nargs="?", help="Path to Scopus export CSV")
    parser.add_argument("--keep-unmatched", action="store_true", help="Keep chunks that did not match any Scopus record")
    args = parser.parse_args()
    run_enrich(scopus_csv_path=args.scopus_csv, keep_unmatched=args.keep_unmatched)
