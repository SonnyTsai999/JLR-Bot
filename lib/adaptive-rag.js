/**
 * Adaptive RAG: block registry, intent classification, evidence confidence.
 * Pipeline: query → intent (which blocks?) → retrieve → confidence filter → synthesize with selected blocks.
 */
import { getConfig, getApiKey } from './config.js';
import { isNarrowFactualQuery } from './query-modes.js';
import OpenAI from 'openai';

/** Content blocks: id, heading, instruction for the LLM. */
export const CONTENT_BLOCKS = {
  direct_answer: {
    heading: 'Direct answer',
    instruction:
      '(Required for list/criteria/definition questions. One line restating the user question, then bullets or a numbered list that answers ONLY that question using the evidence. Cite (Author et al., Year) after each item. If the evidence does not contain the answer, say so explicitly — do not invent.)',
  },
  summary: {
    heading: 'Executive Summary',
    instruction: '(2–4 sentences.)',
  },
  key_findings: {
    heading: 'Key Findings',
    instruction: '(Bullet points grounded in the evidence. Cite the source in parentheses after each point, e.g. (Author et al., Year).)',
  },
  barriers: {
    heading: 'Adoption Barriers',
    instruction: '(Barriers mentioned in the literature; if none found, state "No adoption barriers identified in the retrieved evidence.")',
  },
  roadmap: {
    heading: 'Technology Maturity / Roadmap',
    instruction: '(Stages, timeline, or maturity: Emerging / Growth / Mature / Decline. When evidence supports it, give a concise adoption roadmap or evolution path. Brief justification from evidence.)',
  },
  comparison: {
    heading: 'Comparison',
    instruction: '(Compare or contrast technologies, approaches, or findings from the evidence when relevant.)',
  },
  implications: {
    heading: 'Strategic Implications',
    instruction: '(1–3 concrete, actionable implications. Prefer specific capabilities, named risk areas, or types of partnerships; avoid generic statements like "invest in" or "establish partnerships" without detail.)',
  },
  evidence: {
    heading: 'Evidence Summary',
    instruction: '(Short summary of the most relevant evidence excerpts; cite sources.)',
  },
  sources: {
    heading: 'References',
    instruction: '(One line per source. APA 7: Author, A. A., Author, B. B., & Author, C. C. (Year). Title. Journal, Volume(Issue), pages. https://doi.org/... Do not repeat authors or add a short form.)',
  },
};

const BLOCK_IDS = Object.keys(CONTENT_BLOCKS);

/** Default blocks when intent cannot be determined or evidence is weak. */
const DEFAULT_BLOCKS = ['summary', 'key_findings', 'implications', 'sources'];

/**
 * Classify query intent → which content blocks are relevant.
 * Uses a short LLM call. On failure or if disabled, returns default blocks.
 */
export async function classifyIntent(query, options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  const useLlm = options.use_intent_classification !== false;

  if (!useLlm || !apiKey) {
    return selectBlocksByKeywords(query, DEFAULT_BLOCKS);
  }

  const llm = config.llm || {};
  const client = new OpenAI({
    apiKey,
    ...(llm.base_url ? { baseURL: llm.base_url } : {}),
  });

  const blockList = BLOCK_IDS.filter((id) => id !== 'sources').join(', ');
  const systemMsg = `You are a classifier. Given a user question about technology or AEC, output a comma-separated list of content block IDs that are relevant to answer it. Only use these IDs: ${blockList}. Always include "summary" and "key_findings". For questions that ask for criteria, lists, definitions, "what are the…", "commonly used…", or name a decision method (VIKOR, TOPSIS, AHP, MCDM), include "direct_answer" FIRST in your list. Use "barriers" for adoption/challenges/barriers. Use "roadmap" for maturity/readiness/trends. Use "comparison" for compare/versus/difference. Use "implications" for strategy/recommendations/actions. Use "evidence" only if the user asks for evidence or excerpts. Reply with nothing else.`;
  const userMsg = `Question: ${query}\nRelevant block IDs:`;

  try {
    const resp = await client.chat.completions.create({
      model: llm.model || 'gpt-4o',
      messages: [
        { role: 'system', content: systemMsg },
        { role: 'user', content: userMsg },
      ],
      temperature: 0,
      max_tokens: 80,
    });
    const text = (resp.choices[0]?.message?.content ?? '').trim();
    const ids = text
      .split(/[,;\s]+/)
      .map((s) => s.trim().toLowerCase())
      .filter((id) => BLOCK_IDS.includes(id));
    const unique = [...new Set(ids)];
    if (unique.length === 0) return DEFAULT_BLOCKS;
    if (!unique.includes('sources')) unique.push('sources');
    return unique;
  } catch {
    return selectBlocksByKeywords(query, DEFAULT_BLOCKS);
  }
}

/** Keyword-based fallback for block selection. */
function selectBlocksByKeywords(query, defaultBlocks) {
  const q = query.toLowerCase();
  let chosen = [...new Set(defaultBlocks)];

  if (isNarrowFactualQuery(query)) {
    chosen = ['direct_answer', 'summary', 'key_findings', 'implications', 'sources'].filter(
      (id, i, a) => a.indexOf(id) === i
    );
  }

  if (/\b(barrier|challenge|adoption|obstacle|limit)\b/.test(q) && !chosen.includes('barriers')) chosen.push('barriers');
  if (/\b(maturity|mature|emerging|readiness|trend|roadmap)\b/.test(q) && !chosen.includes('roadmap')) chosen.push('roadmap');
  if (/\b(compare|versus|vs\.?|difference|contrast)\b/.test(q) && !chosen.includes('comparison')) chosen.push('comparison');
  if (/\b(implication|strateg|recommend|action|partnership)\b/.test(q) && !chosen.includes('implications')) chosen.push('implications');
  if (/\b(evidence|excerpt|source|cite)\b/.test(q) && !chosen.includes('evidence')) chosen.push('evidence');

  if (!chosen.includes('sources')) chosen.push('sources');
  return chosen;
}

/**
 * Compute evidence confidence from retrieval scores.
 * chunks may have _score (cosine similarity). Returns { meanScore, ok }.
 */
export function computeEvidenceConfidence(chunks, options = {}) {
  const minConfidence = options.min_confidence ?? 0.25;
  if (!chunks || chunks.length === 0) {
    return { meanScore: 0, ok: false };
  }
  const scores = chunks.map((c) => (typeof c._score === 'number' ? c._score : 0)).filter((s) => s > 0);
  const meanScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  return { meanScore, ok: meanScore >= minConfidence };
}

/**
 * Select final blocks: apply confidence filter. If evidence is weak, use only core blocks.
 */
export function selectBlocksWithConfidence(intentBlocks, chunks, options = {}) {
  const narrow = options.narrow_query === true;
  const defaultCore = narrow
    ? ['direct_answer', 'summary', 'key_findings', 'sources']
    : ['summary', 'key_findings', 'sources'];
  const { min_confidence = 0.25, core_blocks: coreBlocksOpt } = options;
  const core_blocks = coreBlocksOpt ?? defaultCore;
  const { meanScore, ok } = computeEvidenceConfidence(chunks, { min_confidence });

  if (ok) {
    return intentBlocks.includes('sources') ? intentBlocks : [...intentBlocks, 'sources'];
  }
  return core_blocks.includes('sources') ? core_blocks : [...core_blocks, 'sources'];
}
