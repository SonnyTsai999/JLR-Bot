/**
 * LLM synthesis for standard Q&A mode.
 * Uses a simple, direct-answer format — no fixed multi-section template.
 */
import { getConfig, getApiKey } from './config.js';
import { resolveModel } from './chat-models.js';
import { splitAuthorField, joinAuthorsApaStyle } from './author-format.js';
import { sanitizeChunkTextForLlm } from './strip-running-headers.js';
import OpenAI from 'openai';

const SIMPLE_TEMPLATE = `You are a {identity}.

Answer the user's question directly and concisely using ONLY the evidence excerpts provided. Do not add information not present in the evidence. If the evidence is insufficient, say so explicitly.

Choose the most natural format for the answer:
- "What are the criteria / steps / components / indicators?" → numbered or bullet list, each point cited
- Definitions or factual questions → direct prose
- Comparative or analytical questions → concise analysis with inline citations
- Do NOT impose a rigid multi-section report structure (no forced Executive Summary, Key Findings sections)
- Keep the response focused on the specific question; do not drift into unrelated topics

Always end your response with:

## References
One APA 7 citation per source you used (Author, A. A. (Year). Title. https://doi.org/...).

In-text citation rules:
- Use author surnames only: (Surname et al., Year) or (Surname & Surname, Year)
- Never cite a journal or periodical name as an author (e.g. not "(Buildings, 2025)" or "(Mathematics, 2025)")
- Never cite an article title as an in-text reference
- If no author is available: ("Short Title", n.d.)`;

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
  const model = resolveModel(options.model, llm.model);
  const temperature = llm.temperature ?? 0.3;
  const max_tokens = llm.max_tokens ?? 2048;
  const system_identity =
    config.system_identity ||
    'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.';

  if (!retrievedChunks || retrievedChunks.length === 0) {
    return '## References\nNone.\n\nInsufficient evidence was retrieved for this query.';
  }

  const evidence = formatEvidence(retrievedChunks);
  const systemMsg = SIMPLE_TEMPLATE.replace('{identity}', system_identity);
  const userMsg = `Evidence:\n\n${evidence}\n\n---\n\nQuestion: ${query}`;

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
