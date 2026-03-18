/**
 * Deep Research mode: multi-agent pipeline.
 * Flow: User Query → Research Planner → Search Strategy → Retrieval R1 → Evidence Screening →
 *       Retrieval R2 (targeted) → Evidence Extraction → Theme Synthesis → Report Generation.
 */
import { getConfig, getApiKey } from './config.js';
import { retrieve } from './retrieve.js';
import OpenAI from 'openai';

const DEFAULT_TOP_K_R1 = 20;
const DEFAULT_TOP_K_R2 = 5;
const MAX_TARGETED_QUERIES = 3;
const MAX_EVIDENCE_CHARS = 20000;
const DEFAULT_REPORT_MAX_TOKENS = 8192;

function getClient(options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  const llm = config.llm || {};
  const deepCfg = config.deep_research || {};
  return {
    client: new OpenAI({
      apiKey: apiKey || llm.api_key,
      ...(llm.base_url ? { baseURL: llm.base_url } : {}),
    }),
    model: options.model || llm.model || 'gpt-4o',
    temperature: llm.temperature ?? 0.3,
    max_tokens: llm.max_tokens ?? 2048,
    report_max_tokens: deepCfg.report_max_tokens ?? DEFAULT_REPORT_MAX_TOKENS,
  };
}

async function llmCall(systemPrompt, userMessage, options = {}) {
  const clientOpts = getClient(options);
  const maxTokens = options.max_tokens ?? clientOpts.max_tokens;
  const resp = await clientOpts.client.chat.completions.create({
    model: clientOpts.model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
    temperature: clientOpts.temperature ?? 0.3,
    max_tokens: maxTokens ?? 2048,
  });
  return (resp.choices[0]?.message?.content ?? '').trim();
}

function parseJSON(text) {
  const stripped = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/i, '').trim();
  try {
    return JSON.parse(stripped);
  } catch {
    return null;
  }
}

/** Research Planner Agent: turns user query into a research plan and sub-questions. */
async function researchPlannerAgent(query, options = {}) {
  const system = `You are a Research Planner for evidence-based technology intelligence. Given a user question, output a short research plan as JSON with exactly these keys:
- "scope": one sentence describing what to focus on
- "sub_questions": array of 2-4 sub-questions to cover (strings)
- "report_type": either "standard" or "roadmap". Use "roadmap" ONLY if the user specifically asks for a timeline, evolution, maturity, roadmap, or future of a technology. Otherwise, use "standard".
Output only valid JSON, no other text.`;
  const out = await llmCall(system, `User question: ${query}`, options);
  const parsed = parseJSON(out);
  if (parsed && Array.isArray(parsed.sub_questions)) {
    return {
      scope: parsed.scope || query,
      sub_questions: parsed.sub_questions,
      report_type: (parsed.report_type === 'roadmap' || /\b(roadmap|timeline|maturity|evolution)\b/i.test(query)) ? 'roadmap' : 'standard',
    };
  }
  return { 
    scope: query, 
    sub_questions: [query], 
    report_type: /\b(roadmap|timeline|maturity|evolution)\b/i.test(query) ? 'roadmap' : 'standard' 
  };
}

/** Search Strategy Agent: produces primary and alternative search queries. */
async function searchStrategyAgent(query, plan, options = {}) {
  const system = `You are a Search Strategy agent. Given the user question and research plan, output JSON with:
- "primary_query": the main search query (string)
- "alternative_queries": array of 0-2 alternative phrasings (strings)
Output only valid JSON.`;
  const user = `Question: ${query}\nPlan scope: ${plan.scope}\nSub-questions: ${(plan.sub_questions || []).join('; ')}`;
  const out = await llmCall(system, user, options);
  const parsed = parseJSON(out);
  if (parsed && parsed.primary_query) {
    return parsed;
  }
  return { primary_query: query, alternative_queries: [] };
}

/** Evidence Screening Agent: which chunks to keep and what targeted queries to run. */
async function evidenceScreeningAgent(query, chunks, options = {}) {
  if (!chunks || chunks.length === 0) {
    return { keep_indices: [], targeted_queries: [] };
  }
  const list = chunks
    .slice(0, 25)
    .map((c, i) => `[${i}] ${(c.source || c.title || '')}: ${(c.text || '').slice(0, 280)}...`)
    .join('\n');
  const system = `You are an Evidence Screening agent. Given the user question and a list of retrieved excerpts (each prefixed with [index]), output JSON with:
- "keep_indices": array of indices (numbers) that are clearly relevant
- "targeted_queries": array of 0-3 short search queries to fill gaps (e.g. missing perspective, specific term). Omit if evidence seems sufficient.
Output only valid JSON.`;
  const out = await llmCall(system, `Question: ${query}\n\nExcerpts:\n${list}`, options);
  const parsed = parseJSON(out);
  if (parsed && Array.isArray(parsed.keep_indices)) {
    return {
      keep_indices: parsed.keep_indices.filter((n) => Number.isInteger(n) && n >= 0 && n < chunks.length),
      targeted_queries: Array.isArray(parsed.targeted_queries) ? parsed.targeted_queries.slice(0, MAX_TARGETED_QUERIES) : [],
    };
  }
  return { keep_indices: chunks.map((_, i) => i), targeted_queries: [] };
}

/** Evidence Extraction Agent: key facts and claims from combined evidence. */
async function evidenceExtractionAgent(chunks, query, options = {}) {
  if (!chunks || chunks.length === 0) {
    return { key_findings: [], barriers: [], themes: [], sources: [] };
  }
  const evidence = chunks
    .map((c, i) => {
      const title = c.title || c.source || '';
      const year = c.year || '';
      const text = (c.text || '').slice(0, 600);
      return `[${i + 1}] ${title} (${year}):\n${text}`;
    })
    .join('\n\n');
  const config = options.config || getConfig();
  const extractMaxChars = (config.deep_research || {}).max_evidence_chars ?? MAX_EVIDENCE_CHARS;
  const truncated = evidence.slice(0, Math.min(14000, Math.floor(extractMaxChars * 0.7)));
  const system = `You are an Evidence Extraction agent. From the provided excerpts, extract structured content as JSON. Be thorough: include all significant findings.
- "key_findings": array of strings (main findings, each ending with APA in-text citation, e.g. "Finding (Chen et al., 2025)" — avoid "Source N" labels; aim for 8–20 findings where evidence supports)
- "barriers": array of strings (adoption barriers or challenges mentioned)
- "themes": array of strings (recurring themes or topics)
- "sources": array of strings (one-line citation per unique source, APA-style)
Use only information from the excerpts. Output only valid JSON.`;
  const extractMaxTokens = (config.deep_research || {}).extraction_max_tokens ?? 4096;
  const out = await llmCall(system, `Question: ${query}\n\nExcerpts:\n${truncated}`, { ...options, max_tokens: extractMaxTokens });
  const parsed = parseJSON(out);
  if (parsed) {
    return {
      key_findings: Array.isArray(parsed.key_findings) ? parsed.key_findings : [],
      barriers: Array.isArray(parsed.barriers) ? parsed.barriers : [],
      themes: Array.isArray(parsed.themes) ? parsed.themes : [],
      sources: Array.isArray(parsed.sources) ? parsed.sources : [],
    };
  }
  return { key_findings: [], barriers: [], themes: [], sources: [] };
}

/** Theme Synthesis Agent: high-level themes and maturity signal. */
async function themeSynthesisAgent(query, extractions, options = {}) {
  const system = `You are a Theme Synthesis agent. Given the user question and extracted evidence, output JSON with:
- "themes": array of 2-5 theme strings (concise)
- "maturity_signal": one of "Emerging" | "Growth" | "Mature" | "Decline" with one-sentence justification
- "implications": array of 1-3 strategic implication strings
Output only valid JSON.`;
  const user = `Question: ${query}\n\nKey findings: ${(extractions.key_findings || []).join(' | ')}\nBarriers: ${(extractions.barriers || []).join(' | ')}\nThemes from evidence: ${(extractions.themes || []).join(' | ')}`;
  const out = await llmCall(system, user, options);
  const parsed = parseJSON(out);
  if (parsed) {
    return parsed;
  }
  return { themes: [], maturity_signal: 'Unknown', implications: [] };
}

/** Report Generation Agent: final structured report. */
async function reportGenerationAgent(query, themes, extractions, chunks, options = {}) {
  const cfg = options.config || getConfig();
  const identity = cfg.system_identity || 'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.';
  const evidence = chunks
    .slice(0, 30)
    .map((c, i) => {
      const a = (c.authors || '').split(/[;,]+/).map((p) => p.trim()).filter(Boolean);
      const authors = a.length <= 1 ? (a[0] || 'Unknown') : a.slice(0, -1).join(', ') + ', & ' + a[a.length - 1];
      const year = (c.year || '').trim() ? ` (${c.year}).` : '.';
      const title = (c.title || '').trim() || (c.source || 'Untitled');
      const doi = (c.doi || '').trim().replace(/^https?:\/\/doi\.org\//i, '');
      const url = doi ? ` https://doi.org/${doi}` : '';
      return `[Source ${i + 1}: ${authors}${year} ${title}.${url}]\n${(c.text || '').slice(0, 400)}`;
    })
    .join('\n\n');
  const maxEvidenceChars = (cfg.deep_research || {}).max_evidence_chars ?? MAX_EVIDENCE_CHARS;
  const evidenceTrim = evidence.slice(0, maxEvidenceChars);
  const reportMaxTokens = (cfg.deep_research || {}).report_max_tokens ?? DEFAULT_REPORT_MAX_TOKENS;

  let system = '';
  
  if (options.report_type === 'roadmap') {
    system = `You are ${identity} specializing in mapping the evolutionary timelines of emerging technologies. 
Your objective is to extract, synthesize, and structure a technology roadmap using ONLY the provided evidence. 
Do not invent timelines or maturity stages. If the evidence does not provide specific years or phases, infer the logical sequence of capability evolution or explicitly state that the timeline is unclear.

Output your response strictly in the following structure:

## Executive Summary
(2-3 sentences summarizing the overall trajectory and expected impact of the technology.)

## Maturity Assessment
- **Current Stage:** (Emerging / Growth / Mature / Decline)
- **Time to Plateau/Widespread Adoption:** (e.g., <2 years, 2-5 years, 5-10+ years - based on evidence)
- **Justification:** (Brief explanation citing the evidence.)

## Evolutionary Timeline & Milestones
*(Break down the projected evolution. Use approximate timeframes if exact years are missing.)*
* **Near-term / Current State (0-2 years):** Current capabilities, primary use cases, active pilots, and immediate hurdles.
* **Mid-term (2-5 years):** Expected maturation, required standardizations, and system integration.
* **Long-term (5+ years):** Next-generation capabilities, paradigm shifts, and widespread impact.

## Capability Evolution
*(Contrast what the technology can do today vs what it is projected to do in the future.)*
- **Today:** ...
- **Future:** ...

## Critical Dependencies & Enablers
*(Identify technologies, infrastructures, skills, or frameworks that MUST be developed to advance to the next stage.)*

## APA References
(One line per source cited. APA 7 format. Do not repeat authors.)
Citation rules for all narrative sections:
- Use APA in-text citations after claims: (Author, Year) or (Author et al., Year).
- For two authors use "&" in parentheses, e.g. (Smith & Lee, 2024).
- For three or more authors use "et al.".
- Do NOT use "Source N" style citations in the report body.
- If author/year is unavailable, use a short title with n.d., e.g. ("Untitled", n.d.).
Use only the evidence provided.`;
  } else {
    system = `You are ${identity}. You produce a deep-research report using ONLY the provided evidence. Do not invent citations.
Be thorough: expand Key Findings with multiple bullets where the evidence supports it; elaborate on Adoption Barriers and Strategic Implications with concrete detail.

Output your response in this structure:

## Executive Summary
(3–6 sentences: scope, main findings, maturity, and strategic takeaway.)

## Key Findings
(Detailed bullet points; cite in APA in-text format only, e.g. (Author et al., Year). Do not use Source N. Include 5–15 findings where evidence supports.)

## Adoption Barriers
(Barriers from the evidence with brief context; if none, say so.)

## Technology Maturity / Roadmap
(Use the maturity signal and themes; 2–4 sentences with justification.)

## Strategic Implications
(3–5 concrete implications with reasoning tied to evidence.)

## APA References
(One line per source cited. APA 7: Author, A. A., & Author, B. B. (Year). Title. https://doi.org/... Do not repeat authors.)
Rules:
- Use APA in-text citations after claims: (Author, Year) or (Author et al., Year).
- For two authors use "&" in parentheses, e.g. (Smith & Lee, 2024).
- For three or more authors use "et al.".
- Do NOT use "Source N" style citations in the report body.
- If author/year is unavailable, use a short title with n.d., e.g. ("Untitled", n.d.).
Use only the evidence provided.`;
  }

  const user = `Question: ${query}\n\nThemes: ${(themes.themes || []).join('; ')}\nMaturity: ${themes.maturity_signal || 'N/A'}\nImplications (draft): ${(themes.implications || []).join('; ')}\n\nExtracted findings: ${(extractions.key_findings || []).slice(0, 15).join(' | ')}\nBarriers: ${(extractions.barriers || []).join(' | ')}\n\nEvidence:\n${evidenceTrim}`;
  return llmCall(system, user, { ...options, max_tokens: reportMaxTokens });
}

function dedupeChunksById(chunks) {
  const seen = new Set();
  return chunks.filter((c) => {
    const id = c.chunk_id || c.source + ':' + (c.text || '').slice(0, 50);
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

/**
 * Run the full deep research pipeline. Returns { response, sources_used }.
 * Optional options.onStatus(status: string) is called before each step for transparent progress.
 */
export async function runDeepResearch(query, options = {}) {
  const config = options.config || getConfig();
  const apiKey = options.api_key || getApiKey();
  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');
  const opts = { config, api_key: apiKey };
  const onStatus = options.onStatus || (() => {});

  const topK_R1 = options.top_k_r1 ?? DEFAULT_TOP_K_R1;
  const topK_R2 = options.top_k_r2 ?? DEFAULT_TOP_K_R2;

  onStatus('Research Planner Agent');
  const plan = await researchPlannerAgent(query, opts);

  onStatus('Search Strategy Agent');
  const strategy = await searchStrategyAgent(query, plan, opts);

  onStatus('Retrieval Round 1');
  let chunksR1 = await retrieve(strategy.primary_query, {
    ...opts,
    top_k: topK_R1,
  });
  for (const alt of (strategy.alternative_queries || []).slice(0, 1)) {
    const extra = await retrieve(alt, { ...opts, top_k: 6 });
    chunksR1 = dedupeChunksById([...chunksR1, ...extra]);
  }

  onStatus('Evidence Screening Agent');
  const screening = await evidenceScreeningAgent(query, chunksR1, opts);
  let kept = screening.keep_indices.length > 0
    ? screening.keep_indices.map((i) => chunksR1[i]).filter(Boolean)
    : chunksR1;

  onStatus('Retrieval Round 2 (targeted)');
  let chunksR2 = [];
  for (const tq of (screening.targeted_queries || []).slice(0, MAX_TARGETED_QUERIES)) {
    const more = await retrieve(tq, { ...opts, top_k: topK_R2 });
    chunksR2 = dedupeChunksById([...chunksR2, ...more]);
  }
  const combined = dedupeChunksById([...kept, ...chunksR2]);

  onStatus('Evidence Extraction Agent');
  const extractions = await evidenceExtractionAgent(combined, query, opts);

  onStatus('Theme Synthesis Agent');
  const themes = await themeSynthesisAgent(query, extractions, opts);

  onStatus('Report Generation Agent');
  const finalReportType = options.report_type === 'roadmap' ? 'roadmap' 
                        : (options.report_type === 'standard' ? 'standard' 
                        : (plan.report_type || 'standard'));
  const report = await reportGenerationAgent(query, themes, extractions, combined, { ...opts, report_type: finalReportType });

  const seen = new Set();
  const sources_used = [];
  for (const c of combined) {
    const key = (c.doi || '').trim() || (c.apa_citation || '').trim() || c.chunk_id;
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    sources_used.push({
      source: c.source,
      authors: c.authors,
      title: c.title,
      year: c.year,
      doi: c.doi,
      apa_citation: c.apa_citation,
      chunk_id: c.chunk_id,
    });
  }

  return { response: report, sources_used };
}
