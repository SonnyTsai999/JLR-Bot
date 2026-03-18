/**
 * LLM synthesis: format evidence, call OpenAI chat, return structured response.
 * Supports dynamic content blocks (adaptive RAG) or legacy fixed template.
 */
import { getConfig, getApiKey } from './config.js';
import { CONTENT_BLOCKS } from './adaptive-rag.js';
import OpenAI from 'openai';

const LEGACY_TEMPLATE = `You are a {system_identity}

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
- When evidence from multiple sources supports or contrasts a point, cite more than one (e.g. (Author A, Year; Author B, Year)).
- Use APA in-text citations only: (Author, Year) or (Author et al., Year); do not use "Source N" labels.
- For two authors use "&" in parentheses, e.g. (Smith & Lee, 2024); for three or more use "et al.".
- If author/year is unavailable, use a short title and n.d., e.g. ("Untitled", n.d.).
- No single excerpt longer than ~500 words in your discussion.
- Do not reproduce full sections verbatim.`;

/** Build dynamic prompt from selected content block IDs. Always include sources. */
export function buildDynamicPrompt(blockIds, system_identity) {
  const ids = Array.isArray(blockIds) && blockIds.length ? blockIds : ['summary', 'key_findings', 'sources'];
  const withSources = ids.includes('sources') ? ids : [...ids, 'sources'];
  const sections = withSources
    .filter((id) => CONTENT_BLOCKS[id])
    .map((id) => {
      const b = CONTENT_BLOCKS[id];
      return `## ${b.heading}\n${b.instruction}`;
    })
    .join('\n\n');

  return `You are a ${system_identity}

Answer the user's question using ONLY the provided evidence excerpts. Do not invent citations or add information not supported by the evidence. If evidence is insufficient for a section, say so briefly or omit that section.

Output your response in the following structure (include only these sections):

${sections}

Rules:
- Use only retrieved evidence. Do not invent citations.
- Cite each source you use at least once (e.g. (Author et al., Year)).
- When multiple sources support or contrast a point, cite more than one.
- Use APA in-text citations only: (Author, Year) or (Author et al., Year); do not use "Source N" labels.
- For two authors use "&" in parentheses, e.g. (Smith & Lee, 2024); for three or more use "et al.".
- If author/year is unavailable, use a short title and n.d., e.g. ("Untitled", n.d.).
- For Strategic Implications: be specific (e.g. which capability, which risk); avoid generic advice.
- No single excerpt longer than ~500 words in your discussion.
- Do not reproduce full sections verbatim.`;
}

/** Build APA 7 citation from chunk fields (avoids duplicated/stored malformed apa_citation). */
function chunkToAPA7(c) {
  const raw = (c.authors || '').trim();
  const parts = raw ? raw.split(/[;,]+/).map((p) => p.trim()).filter(Boolean) : [];
  const authorsStr = parts.length === 0 ? 'Unknown' : parts.length === 1 ? parts[0] : parts.slice(0, -1).join(', ') + ', & ' + parts[parts.length - 1];
  const year = (c.year || '').trim() ? ` (${(c.year || '').trim()}).` : '.';
  const title = (c.title || '').trim() || (c.source || 'Untitled');
  const doi = (c.doi || '').trim().replace(/^https?:\/\/doi\.org\//i, '');
  const url = doi ? ` https://doi.org/${doi}` : '';
  return authorsStr + year + ' ' + title + '.' + url;
}

function formatEvidence(chunks, maxExcerptChars = 8000) {
  const parts = [];
  let total = 0;
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const text = (c.text || '').slice(0, 2000);
    if (total + text.length > maxExcerptChars) break;
    total += text.length;
    const citation = chunkToAPA7(c);
    const header = `[Source ${i + 1}: ${citation}]\n`;
    parts.push(header + text);
  }
  return parts.length ? parts.join('\n\n---\n\n') : '(No evidence provided.)';
}

export async function synthesize(query, retrievedChunks, options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');

  const llm = config.llm || {};
  const model = options.model || llm.model || 'gpt-4o';
  const temperature = llm.temperature ?? 0.3;
  const max_tokens = llm.max_tokens ?? 2048;
  const system_identity = config.system_identity || 'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.';

  if (!retrievedChunks || retrievedChunks.length === 0) {
    return (
      '## Executive Summary\n' +
      'Insufficient evidence was retrieved for this query. No analytical response can be generated without source material.\n\n' +
      '## APA References\n' +
      'None.'
    );
  }

  const evidence = formatEvidence(retrievedChunks);
  const systemMsg =
    options.blocks && options.blocks.length
      ? buildDynamicPrompt(options.blocks, system_identity)
      : LEGACY_TEMPLATE.replace('{system_identity}', system_identity);
  const userMsg = `Evidence:\n\n${evidence}\n\n---\n\nUser question: ${query}`;

  const client = new OpenAI({
    apiKey,
    ...(llm.base_url ? { baseURL: llm.base_url } : {}),
  });

  const resp = await client.chat.completions.create({
    model,
    messages: [
      { role: 'system', content: systemMsg },
      { role: 'user', content: userMsg },
    ],
    temperature,
    max_tokens,
  });

  return resp.choices[0]?.message?.content ?? '';
}
