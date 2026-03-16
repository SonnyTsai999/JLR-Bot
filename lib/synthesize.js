/**
 * LLM synthesis: format evidence, call OpenAI chat, return structured response.
 */
import { getConfig, getApiKey } from './config.js';
import OpenAI from 'openai';

const STRUCTURED_TEMPLATE = `You are a {system_identity}

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
- Do not reproduce full sections verbatim.`;

function formatEvidence(chunks, maxExcerptChars = 8000) {
  const parts = [];
  let total = 0;
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const title = c.title || c.source || '';
    const authors = c.authors || '';
    const year = c.year;
    const doi = c.doi || '';
    const apa = c.apa_citation || '';
    const text = (c.text || '').slice(0, 2000);
    if (total + text.length > maxExcerptChars) break;
    total += text.length;
    let header = `[Source ${i + 1}:`;
    if (apa) header += ` ${apa}`;
    else header += ` ${title}${authors ? `; ${authors}` : ''}${year ? `, ${year}` : ''}${doi ? `, DOI: ${doi}` : ''}`;
    header += ']\n';
    parts.push(header + text);
  }
  return parts.length ? parts.join('\n\n---\n\n') : '(No evidence provided.)';
}

export async function synthesize(query, retrievedChunks, options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');

  const llm = config.llm || {};
  const model = llm.model || 'gpt-4o';
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
  const systemMsg = STRUCTURED_TEMPLATE.replace('{system_identity}', system_identity);
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
