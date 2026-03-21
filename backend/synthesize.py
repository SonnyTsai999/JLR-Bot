"""
LLM synthesis for JLR Technology Intelligence Assistant.
Uses only retrieved evidence; supports dynamic content blocks (adaptive RAG) or legacy fixed template.
"""
from __future__ import annotations

from typing import Any

from config.load_config import load_config

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from backend.adaptive_rag import CONTENT_BLOCKS

LEGACY_TEMPLATE = """You are a {system_identity}

Answer the user's question using ONLY the provided evidence excerpts. Do not invent citations or add information not supported by the evidence. If evidence is insufficient, say so explicitly.

Output your response in the following structure:

## Executive Summary
(2–4 sentences.)

## Key Findings
(Bullet points grounded in the evidence. Cite the source in parentheses after each point, e.g. (Author et al., Year).)

## Adoption Barriers
(Barriers mentioned in the literature; if none found, state "No adoption barriers identified in the retrieved evidence.")

## Technology Maturity Signal
(Emerging / Growth / Mature / Decline, with brief justification from evidence.)

## Strategic Implications for JLR
(1–3 concrete, actionable implications. Prefer specific capabilities, named risk areas, or types of partnerships; avoid generic statements without detail.)

## APA References
(One line per source. APA 7 format: Author, A. A., Author, B. B., & Author, C. C. (Year). Title. Journal, Volume(Issue), pages. https://doi.org/... Do not repeat the author list or add a short form after the citation.)

Rules:
- Use only retrieved evidence. Do not invent citations.
- Cite each source you use at least once in the narrative.
- No single excerpt longer than ~500 words in your discussion.
- Do not reproduce full sections verbatim."""


def build_dynamic_prompt(block_ids: list[str], system_identity: str) -> str:
    """Build prompt from selected content block IDs. Always include sources."""
    ids = block_ids if block_ids else ["summary", "key_findings", "sources"]
    if "sources" not in ids:
        ids = list(ids) + ["sources"]
    sections = []
    for bid in ids:
        b = CONTENT_BLOCKS.get(bid)
        if b:
            sections.append(f"## {b['heading']}\n{b['instruction']}")
    body = "\n\n".join(sections)
    return f"""You are a {system_identity}

Answer the user's question using ONLY the provided evidence excerpts. Do not invent citations or add information not supported by the evidence. If evidence is insufficient for a section, say so briefly or omit that section.

Output your response in the following structure (include only these sections):

{body}

Rules:
- Use only retrieved evidence. Do not invent citations.
- Cite each source you use at least once (e.g. (Author et al., Year)).
- When multiple sources support or contrast a point, cite more than one.
- For Strategic Implications: be specific (e.g. which capability, which risk); avoid generic advice.
- No single excerpt longer than ~500 words in your discussion.
- Do not reproduce full sections verbatim."""


def chunk_to_apa7(c: dict[str, Any]) -> str:
    """Build APA 7 citation from chunk fields (avoids duplicated/stored malformed apa_citation)."""
    import re
    raw = (c.get("authors") or "").strip()
    parts = [p.strip() for p in re.split(r"[;,]+", raw) if p.strip()] if raw else []
    if not parts:
        authors_str = "Unknown"
    elif len(parts) == 1:
        authors_str = parts[0]
    else:
        authors_str = ", ".join(parts[:-1]) + ", & " + parts[-1]
    y = (c.get("year") or "").strip()
    year = f" ({y})." if y else "."
    title = (c.get("title") or "").strip() or (c.get("source") or "Untitled")
    doi = (c.get("doi") or "").strip()
    if doi and doi.lower().startswith("http"):
        doi = re.sub(r"^https?://doi\.org/", "", doi, flags=re.I)
    url = " https://doi.org/" + doi if doi else ""
    return authors_str + year + " " + title + "." + url


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> "OpenAI":
    if not HAS_OPENAI:
        raise RuntimeError("openai is required. Install with: pip install openai")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def format_evidence(chunks: list[dict[str, Any]], max_excerpt_chars: int = 8000) -> str:
    """Format retrieved chunks for the prompt. Use clean APA 7 citation per chunk."""
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        text = (c.get("text") or "")[:2000]
        if total + len(text) > max_excerpt_chars:
            break
        total += len(text)
        citation = chunk_to_apa7(c)
        header = f"[Source {i+1}: {citation}]\n"
        parts.append(header + text)
    return "\n\n---\n\n".join(parts) if parts else "(No evidence provided.)"


def synthesize(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    blocks: list[str] | None = None,
) -> str:
    """
    Generate structured analytical response from query and retrieved chunks.
    If blocks is provided, uses dynamic prompt with only those sections; else legacy template.
    """
    if not HAS_OPENAI:
        raise RuntimeError("openai is required. Install with: pip install openai")
    config = config or load_config()
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model", "gpt-4o")
    temperature = llm_cfg.get("temperature", 0.3)
    max_tokens = llm_cfg.get("max_tokens", 2048)
    system_identity = config.get("system_identity", "Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.")

    if not retrieved_chunks:
        return (
            "## Executive Summary\n"
            "Insufficient evidence was retrieved for this query. No analytical response can be generated without source material.\n\n"
            "## APA References\n"
            "None."
        )

    evidence = format_evidence(retrieved_chunks)
    if blocks:
        system_msg = build_dynamic_prompt(blocks, system_identity)
    else:
        system_msg = LEGACY_TEMPLATE.format(system_identity=system_identity)
    user_msg = f"Evidence:\n\n{evidence}\n\n---\n\nUser question: {query}"

    base_url = llm_cfg.get("base_url") or None
    client = get_openai_client(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.retrieve import retrieve
    from config.load_config import get_api_key

    config = load_config()
    chunks = retrieve("What are adoption barriers for BIM in construction?", config=config, api_key=get_api_key())
    answer = synthesize("What are adoption barriers for BIM in construction?", chunks, config=config, api_key=get_api_key())
    print(answer)
