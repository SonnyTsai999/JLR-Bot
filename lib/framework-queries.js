/**
 * Named academic / adoption frameworks (often asked by acronym).
 * Strict mode: only answer from excerpts that explicitly name the framework or its standard full name.
 */

export const FRAMEWORK_DEFS = [
  {
    id: 'toe',
    acronyms: [/\bTOE\b/i],
    fullNameRegex: /technology[-\s–—]+organization[-\s–—]+environment/i,
    retrievalBoosts: [
      'TOE Technology Organization Environment framework adoption',
      'TOE framework AEC construction building industry',
      'technology organization environment TOE innovation adoption',
    ],
  },
  {
    id: 'tam',
    acronyms: [/\bTAM\b/i],
    fullNameRegex: /technology\s+acceptance\s+model/i,
    retrievalBoosts: [
      'Technology Acceptance Model TAM AEC construction',
      'TAM perceived usefulness perceived ease of use BIM',
    ],
  },
  {
    id: 'utaut',
    acronyms: [/\bUTAUT\b/i],
    fullNameRegex: /unified\s+theory\s+of\s+acceptance\s+and\s+use/i,
    retrievalBoosts: [
      'UTAUT unified theory acceptance use technology construction',
      'UTAUT BIM building information modeling',
    ],
  },
];

export function isFrameworkStrictQuery(query) {
  const q = query || '';
  return FRAMEWORK_DEFS.some((def) => def.acronyms.some((re) => re.test(q)));
}

export function getFrameworkRetrievalBoosts(query) {
  const q = query || '';
  const boosts = [];
  for (const def of FRAMEWORK_DEFS) {
    if (def.acronyms.some((re) => re.test(q))) boosts.push(...def.retrievalBoosts);
  }
  return [...new Set(boosts)];
}

/** Human-readable labels for prompts */
export function labelsForMatchingFrameworks(query) {
  const q = query || '';
  const labels = [];
  if (/\bTOE\b/i.test(q)) labels.push('TOE (Technology–Organization–Environment)');
  if (/\bTAM\b/i.test(q)) labels.push('TAM (Technology Acceptance Model)');
  if (/\bUTAUT\b/i.test(q)) labels.push('UTAUT (Unified Theory of Acceptance and Use of Technology)');
  return labels.length ? labels.join('; ') : 'the named framework';
}

export function frameworkStrictPromptFragment(labels) {
  return `NAMED-FRAMEWORK STRICT MODE — ${labels}:
- You may list criteria, dimensions, constructs, or factors ONLY if an excerpt explicitly names the acronym (e.g. TOE) OR spells out the usual academic full name (e.g. Technology–Organization–Environment).
- FORBIDDEN: Do not "map" unrelated models (LEED, BREEAM, UB-EIA, EIA, generic sustainability pillars, MCDM methods) onto TOE/TAM/UTAUT dimensions unless the same sentence or paragraph in the evidence explicitly states that link.
- If no excerpt qualifies, your FIRST heading must be "## Direct answer" and you must state clearly that the retrieved corpus does not explicitly discuss ${labels}; you cannot list framework-specific criteria from this evidence. Suggest widening the literature search — do not invent criteria.
- Omit Executive Summary / Key Findings that pretend the question was answered when it was not.`;
}

/** Rough check: does chunk text explicitly reference the framework? */
export function chunkExplicitlyMentionsFramework(query, text) {
  const t = text || '';
  const q = query || '';
  for (const def of FRAMEWORK_DEFS) {
    if (!def.acronyms.some((re) => re.test(q))) continue;
    if (def.acronyms.some((re) => re.test(t))) return true;
    if (def.fullNameRegex.test(t)) return true;
  }
  return false;
}

export function countChunksWithFrameworkMention(query, chunks) {
  if (!isFrameworkStrictQuery(query) || !Array.isArray(chunks)) return { total: 0, withMention: 0 };
  let withMention = 0;
  for (const c of chunks) {
    const txt = `${c.title || ''} ${c.text || ''}`;
    if (chunkExplicitlyMentionsFramework(query, txt)) withMention++;
  }
  return { total: chunks.length, withMention };
}
