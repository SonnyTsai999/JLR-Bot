"""
Adaptive RAG: block registry, intent classification, evidence confidence.
Pipeline: query → intent (which blocks?) → retrieve → confidence filter → synthesize with selected blocks.
"""
from __future__ import annotations

import re
from typing import Any

CONTENT_BLOCKS: dict[str, dict[str, str]] = {
    "summary": {
        "heading": "Executive Summary",
        "instruction": "(2–4 sentences.)",
    },
    "key_findings": {
        "heading": "Key Findings",
        "instruction": "(Bullet points grounded in the evidence. Cite the source in parentheses after each point, e.g. (Author et al., Year).)",
    },
    "barriers": {
        "heading": "Adoption Barriers",
        "instruction": '(Barriers mentioned in the literature; if none found, state "No adoption barriers identified in the retrieved evidence.")',
    },
    "roadmap": {
        "heading": "Technology Maturity / Roadmap",
        "instruction": "(Stages, timeline, or maturity: Emerging / Growth / Mature / Decline. When evidence supports it, give a concise adoption roadmap or evolution path. Brief justification from evidence.)",
    },
    "comparison": {
        "heading": "Comparison",
        "instruction": "(Compare or contrast technologies, approaches, or findings from the evidence when relevant.)",
    },
    "implications": {
        "heading": "Strategic Implications",
        "instruction": "(1–3 concrete, actionable implications. Prefer specific capabilities, named risk areas, or types of partnerships; avoid generic statements like 'invest in' or 'establish partnerships' without detail.)",
    },
    "evidence": {
        "heading": "Evidence Summary",
        "instruction": "(Short summary of the most relevant evidence excerpts; cite sources.)",
    },
    "sources": {
        "heading": "APA References",
        "instruction": "(One line per source. APA 7: Author, A. A., Author, B. B., & Author, C. C. (Year). Title. Journal, Volume(Issue), pages. https://doi.org/... Do not repeat authors or add a short form.)",
    },
}

BLOCK_IDS = list(CONTENT_BLOCKS.keys())
DEFAULT_BLOCKS = ["summary", "key_findings", "implications", "sources"]


def classify_intent(
    query: str,
    config: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    use_llm: bool = True,
) -> list[str]:
    """Classify query intent → which content blocks are relevant. Uses LLM or keyword fallback."""
    from config.load_config import load_config

    config = config or load_config()
    if not use_llm or not api_key:
        return _select_blocks_by_keywords(query, DEFAULT_BLOCKS)

    try:
        from openai import OpenAI

        llm_cfg = config.get("llm", {})
        base_url = llm_cfg.get("base_url") or None
        client = OpenAI(api_key=api_key, base_url=base_url)
        block_list = ", ".join(b for b in BLOCK_IDS if b != "sources")
        system_msg = (
            f"You are a classifier. Given a user question about technology or AEC, output a comma-separated list of content block IDs that are relevant. "
            f"Only use: {block_list}. Always include summary and key_findings. "
            "Use barriers for adoption/challenges. Use roadmap for maturity/readiness. Use comparison for compare/versus. Use implications for strategy/recommendations. "
            "Reply with nothing else."
        )
        user_msg = f"Question: {query}\nRelevant block IDs:"
        resp = client.chat.completions.create(
            model=llm_cfg.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip()
        ids = [s.strip().lower() for s in re.split(r"[,;\s]+", text) if s.strip()]
        ids = [i for i in ids if i in BLOCK_IDS]
        if not ids:
            return DEFAULT_BLOCKS
        if "sources" not in ids:
            ids.append("sources")
        return list(dict.fromkeys(ids))
    except Exception:
        return _select_blocks_by_keywords(query, DEFAULT_BLOCKS)


def _select_blocks_by_keywords(query: str, default_blocks: list[str]) -> list[str]:
    q = query.lower()
    chosen = list(dict.fromkeys(default_blocks))
    if re.search(r"\b(barrier|challenge|adoption|obstacle|limit)\b", q) and "barriers" not in chosen:
        chosen.append("barriers")
    if re.search(r"\b(maturity|mature|emerging|readiness|trend|roadmap)\b", q) and "roadmap" not in chosen:
        chosen.append("roadmap")
    if re.search(r"\b(compare|versus|vs\.?|difference|contrast)\b", q) and "comparison" not in chosen:
        chosen.append("comparison")
    if re.search(r"\b(implication|strateg|recommend|action|partnership)\b", q) and "implications" not in chosen:
        chosen.append("implications")
    if re.search(r"\b(evidence|excerpt|source|cite)\b", q) and "evidence" not in chosen:
        chosen.append("evidence")
    if "sources" not in chosen:
        chosen.append("sources")
    return chosen


def compute_evidence_confidence(
    chunks: list[dict[str, Any]],
    min_confidence: float = 0.25,
) -> tuple[float, bool]:
    """Returns (mean_score, ok). chunks may have _score."""
    if not chunks:
        return 0.0, False
    scores = [c.get("_score") for c in chunks if isinstance(c.get("_score"), (int, float))]
    if not scores:
        return 0.0, False
    mean_score = sum(scores) / len(scores)
    return mean_score, mean_score >= min_confidence


def select_blocks_with_confidence(
    intent_blocks: list[str],
    chunks: list[dict[str, Any]],
    min_confidence: float = 0.25,
    core_blocks: list[str] | None = None,
) -> list[str]:
    """Apply confidence filter: if evidence weak, return only core blocks."""
    core_blocks = core_blocks or ["summary", "key_findings", "sources"]
    _, ok = compute_evidence_confidence(chunks, min_confidence)
    if ok:
        if "sources" not in intent_blocks:
            return intent_blocks + ["sources"]
        return intent_blocks
    return core_blocks
