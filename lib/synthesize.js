/**
 * LLM synthesis for standard Q&A mode.
 * Uses a simple, direct-answer format — no fixed multi-section template.
 */
import { getConfig, getApiKey } from './config.js';
import { resolveModel } from './chat-models.js';
import { chatCompletionsPayload } from './llm-chat-params.js';
import { splitAuthorField, joinAuthorsApaStyle } from './author-format.js';
import { sanitizeChunkTextForLlm } from './strip-running-headers.js';
import {
  isFrameworkStrictQuery,
  frameworkStrictPromptFragment,
  labelsForMatchingFrameworks,
  countChunksWithFrameworkMention,
} from './framework-queries.js';
import OpenAI from 'openai';

const SIMPLE_TEMPLATE = `You are a {identity}.

OUTPUT FORMAT (normal Q&A — not a report):
1. Start immediately with the substantive answer. Optional single heading: "## Answer" then your text; or no heading and go straight into prose/bullets.
2. Do NOT use report-style sections. Forbidden headings (do not write these at all): "Executive Summary", "Key Findings", "Evidence Summary", "Introduction", "Overview", "Discussion", "Strategic Implications", "Technology Maturity", "Adoption Barriers", or any similar multi-section report blocks.
3. If evidence is insufficient, say so in plain sentences inside the answer — still no Executive Summary / Key Findings structure.
4. Answer using ONLY the evidence excerpts. Do not invent facts or citations.

Format inside the answer only:
- Criteria / lists / steps → numbered or bullet list, each point cited where relevant
- Definitions or short factual questions → short direct prose
- Comparisons → brief analysis with inline citations

Then end with exactly one block (mandatory):

## References
One APA 7 line per source you cited (Author, A. A. (Year). Title. https://doi.org/...).

In-text citations: author surnames only — (Surname et al., Year) or (Surname & Surname, Year). Never use journal names or article titles as if they were authors. If no author: ("Short Title", n.d.).`;

function chunkToAPA7(c) {
  const parts = splitAuthorField(c.authors || '');
  const authorsStr = joinAuthorsApaStyle(parts);
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
    const text = sanitizeChunkTextForLlm(c.text, 2000);
    if (total + text.length > maxExcerptChars) break;
    total += text.length;
    const citation = chunkToAPA7(c);
    parts.push(`[Source ${i + 1}: ${citation}]\n` + text);
  }
  return parts.length ? parts.join('\n\n---\n\n') : '(No evidence provided.)';
}

export async function synthesize(query, retrievedChunks, options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');

  const llm = config.llm || {};
  // API handler already resolves; re-resolve only when synthesize is called without model (e.g. tests).
  const rawPick =
    typeof options.model === 'string' && options.model.trim() ? options.model.trim() : undefined;
  const model = resolveModel(rawPick, llm.model);
  const system_identity =
    config.system_identity ||
    'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.';

  if (!retrievedChunks || retrievedChunks.length === 0) {
    return '## References\nNone.\n\nInsufficient evidence was retrieved for this query.';
  }

  const evidence = formatEvidence(retrievedChunks);
  let systemMsg = SIMPLE_TEMPLATE.replace('{identity}', system_identity);
  if (isFrameworkStrictQuery(query)) {
    const stats = countChunksWithFrameworkMention(query, retrievedChunks);
    const fw = frameworkStrictPromptFragment(labelsForMatchingFrameworks(query));
    systemMsg = `${fw}\n\nExcerpt check: ${stats.withMention} of ${stats.total} retrieved chunks explicitly name the framework acronym or its standard full name.\n\n${systemMsg}`;
  }
  const userMsg = `Evidence:\n\n${evidence}\n\n---\n\nQuestion: ${query}`;

  const client = new OpenAI({
    apiKey,
    ...(llm.base_url ? { baseURL: llm.base_url } : {}),
  });

  const payload = chatCompletionsPayload(model, llm, [
    { role: 'system', content: systemMsg },
    { role: 'user', content: userMsg },
  ]);
  const resp = await client.chat.completions.create(payload);

  return resp.choices[0]?.message?.content ?? '';
}
