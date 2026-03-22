/**
 * Narrow / direct questions: criteria lists, definitions, named methods (VIKOR, TOPSIS, …).
 * These need a "Direct answer" first and less topic drift than broad technology scans.
 */

const NARROW_PATTERNS =
  /\b(what are (the )?|list\s|enumerate\s|^define\s|^what is\b|criteria\b|indicators?\b|dimensions?\b|factors?\b|components?\b|steps\b|commonly used\b|typical\s|standard\s+(practice|criteria)|which\s+(criteria|factors|dimensions))\b/i;

const METHOD_TERMS =
  /\b(VIKOR|TOPSIS|AHP|ANP|ELECTRE|PROMETHEE|MACBETH|MCDM|multi[\s-]?criteria|SAW|WASPAS|EDAS|CODAS)\b/i;

export function isNarrowFactualQuery(query) {
  const s = (query || '').trim();
  if (!s) return false;
  if (NARROW_PATTERNS.test(s)) return true;
  if (METHOD_TERMS.test(s) && /\b(criteria|indicator|dimension|factor|define|what|which|list)\b/i.test(s)) return true;
  return false;
}

/** Prepended to system prompts for synthesis / deep research in narrow mode. */
export function narrowQueryFocusBlock() {
  return `QUERY MODE — NARROW / DIRECT (the user asked a specific list, definition, or methodological question):
1. The FIRST section must be "## Direct answer": one line restating the question, then bullets or a numbered list that answers ONLY that question from the evidence (each point cited).
2. Do not expand into unrelated AEC topics (generic BIM, AR/VR, sustainability surveys) unless an excerpt explicitly ties them to the exact question (e.g. VIKOR criteria).
3. If the question names a method (VIKOR, TOPSIS, …), prioritize criteria, attributes, weights, or steps for that method. If the corpus does not contain them, say clearly that evidence is insufficient — do not invent.
4. Keep "## Executive Summary" to 1–2 sentences summarizing that direct answer, not a broad literature review.
5. "## Key Findings" must only include points that support the direct answer; omit tangential papers.
6. Omit "## Adoption Barriers" and "## Technology Maturity / Roadmap" (or equivalent) unless the user asked for them or the evidence speaks only to those topics for the same question.`;
}

/**
 * For adaptive RAG block lists: drop roadmap/barriers on narrow queries unless the user asked.
 */
export function trimBlocksForNarrowQuery(blocks, query) {
  if (!isNarrowFactualQuery(query) || !Array.isArray(blocks)) return blocks;
  const wantsMaturity = /\b(maturity|roadmap|emerging|growth|decline|timeline|evolution|readiness|future of)\b/i.test(query);
  const wantsBarriers = /\b(barrier|challenge|adoption obstacle|obstacle|hurdle)\b/i.test(query);
  let b = [...blocks];
  if (!wantsMaturity) b = b.filter((id) => id !== 'roadmap');
  if (!wantsBarriers) b = b.filter((id) => id !== 'barriers');
  if (!b.includes('sources')) b.push('sources');
  return b;
}
