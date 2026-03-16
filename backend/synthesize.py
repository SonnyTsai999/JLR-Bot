"""
LLM synthesis for JLR Technology Intelligence Assistant.
Uses only retrieved evidence; outputs structured template with APA references.
"""
from __future__ import annotations

from typing import Any

from config.load_config import load_config

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


STRUCTURED_TEMPLATE = """You are a {system_identity}

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
(Concrete, actionable implications for JLR's technology strategy based on the evidence—e.g. specific capabilities to develop, risks to mitigate, or partnerships to consider. Do not be generic.)

## APA References
(List every source you cited in the response, in APA 7 format. Use the exact APA citation provided for each source in the evidence. Include DOI when available. Do not list sources you did not cite.)

Rules:
- Use only retrieved evidence. Do not invent citations.
- Cite each source you use at least once in the narrative (Key Findings, Barriers, or Maturity).
- No single excerpt longer than ~500 words in your discussion.
- Do not reproduce full sections verbatim."""


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> "OpenAI":
    if not HAS_OPENAI:
        raise RuntimeError("openai is required. Install with: pip install openai")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def format_evidence(chunks: list[dict[str, Any]], max_excerpt_chars: int = 8000) -> str:
    """Format retrieved chunks for the prompt. Include APA citation when available."""
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        title = c.get("title") or c.get("source", "")
        authors = c.get("authors", "")
        year = c.get("year")
        doi = c.get("doi", "")
        apa = c.get("apa_citation", "")
        text = (c.get("text") or "")[:2000]
        if total + len(text) > max_excerpt_chars:
            break
        total += len(text)
        header = f"[Source {i+1}:"
        if apa:
            header += f" {apa}"
        else:
            header += f" {title}"
            if authors:
                header += f"; {authors}"
            if year:
                header += f", {year}"
            if doi:
                header += f", DOI: {doi}"
        header += "]\n"
        parts.append(header + text)
    return "\n\n---\n\n".join(parts) if parts else "(No evidence provided.)"


def synthesize(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
) -> str:
    """
    Generate structured analytical response from query and retrieved chunks.
    Uses system identity and structured template; evidence-only, no hallucinated citations.
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
    system_msg = STRUCTURED_TEMPLATE.format(system_identity=system_identity)
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
