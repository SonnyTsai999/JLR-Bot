/**
 * Deep Research mode: multi-agent pipeline.
 * Flow: User Query → Research Planner → Search Strategy → Retrieval R1 → Evidence Screening →
 *       Retrieval R2 (targeted) → Evidence Extraction → Theme Synthesis → Report Generation.
 */
import { getConfig, getApiKey } from './config.js';
import { resolveModel } from './chat-models.js';
import { retrieve } from './retrieve.js';
import { splitAuthorField, joinAuthorsApaStyle } from './author-format.js';
import { isNarrowFactualQuery } from './query-modes.js';
import {
  isFrameworkStrictQuery,
  getFrameworkRetrievalBoosts,
  labelsForMatchingFrameworks,
  frameworkStrictPromptFragment,
  countChunksWithFrameworkMention,
  chunkExplicitlyMentionsFramework,
} from './framework-queries.js';
import { sanitizeChunkTextForLlm } from './strip-running-headers.js';
import { chatCompletionsPayload } from './llm-chat-params.js';
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
    model: resolveModel(options.model, llm.model),
    temperature: llm.temperature ?? 0.3,
    max_tokens: llm.max_tokens ?? 2048,
    report_max_tokens: deepCfg.report_max_tokens ?? DEFAULT_REPORT_MAX_TOKENS,
  };
}

async function llmCall(systemPrompt, userMessage, options = {}) {
  const clientOpts = getClient(options);
  const maxTokens = options.max_tokens ?? clientOpts.max_tokens;
  const config = options.config || getConfig();
  const llm = config.llm || {};
  const payload = chatCompletionsPayload(
    clientOpts.model,
    llm,
    [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
    maxTokens ?? 2048,
  );
  const resp = await clientOpts.client.chat.completions.create(payload);
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
  const narrow = options.narrow_query === true;
  const fw = options.framework_strict === true;
  const narrowHint = narrow
    ? '\nNARROW QUESTION: scope and sub_questions must stay strictly on the user\'s exact ask (e.g. criteria of VIKOR or TOE in AEC). Do not broaden to generic surveys of AI, BIM, or sustainability unless the question asks for them.'
    : '';
  const fwHint = fw
    ? '\nFRAMEWORK QUESTION: include sub_questions that use BOTH the acronym AND the full academic name (e.g. TOE + Technology Organization Environment + construction/AEC) so retrieval can find explicit mentions.'
    : '';
  const system = `You are a Research Planner for evidence-based technology intelligence. Given a user question, output a short research plan as JSON with exactly these keys:
- "scope": one sentence describing what to focus on
- "sub_questions": array of 2-4 sub-questions to cover (strings)
- "report_type": either "standard" or "roadmap". Use "roadmap" ONLY if the user specifically asks for a timeline, evolution, maturity, roadmap, or future of a technology. Otherwise, use "standard".
Output only valid JSON, no other text.${narrowHint}${fwHint}`;
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
  const narrow = options.narrow_query === true;
  const fw = options.framework_strict === true;
  let extra = narrow
    ? '\nBias primary_query and alternatives toward the user\'s exact terms (e.g. VIKOR OR TOPSIS OR MCDM, criteria, attributes, AEC, construction).'
    : '';
  if (fw) {
    extra +=
      '\nIf the question names TOE, TAM, or UTAUT: primary_query MUST include the acronym AND the spelled-out framework name AND AEC/construction/building context (e.g. "TOE Technology Organization Environment framework adoption construction").';
  }
  const user = `Question: ${query}\nPlan scope: ${plan.scope}\nSub-questions: ${(plan.sub_questions || []).join('; ')}${extra}`;
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
    .map((c, i) => `[${i}] ${(c.source || c.title || '')}: ${sanitizeChunkTextForLlm(c.text, 280)}...`)
    .join('\n');
  const narrow = options.narrow_query === true;
  const fw = options.framework_strict === true;
  let narrowHint = narrow
    ? ' Prefer excerpts that explicitly mention the user\'s key terms (e.g. VIKOR, criteria, MCDM). Deprioritize generic BIM/AR papers unless they clearly address the same question.'
    : '';
  if (fw) {
    narrowHint +=
      ' FRAMEWORK STRICT: keep ONLY excerpts whose text explicitly names the framework acronym (e.g. TOE) OR its standard full name (e.g. Technology–Organization–Environment). Reject excerpts that only discuss LEED, BREEAM, UB-EIA, or generic sustainability without naming the framework.';
  }
  const system = `You are an Evidence Screening agent. Given the user question and a list of retrieved excerpts (each prefixed with [index]), output JSON with:
- "keep_indices": array of indices (numbers) that are clearly relevant
- "targeted_queries": array of 0-3 short search queries to fill gaps (e.g. missing perspective, specific term). Omit if evidence seems sufficient.
Output only valid JSON.${narrowHint}`;
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
      const text = sanitizeChunkTextForLlm(c.text, 600);
      return `[${i + 1}] ${title} (${year}):\n${text}`;
    })
    .join('\n\n');
  const config = options.config || getConfig();
  const extractMaxChars = (config.deep_research || {}).max_evidence_chars ?? MAX_EVIDENCE_CHARS;
  const truncated = evidence.slice(0, Math.min(14000, Math.floor(extractMaxChars * 0.7)));
  const narrow = options.narrow_query === true;
  const fw = options.framework_strict === true;
  let narrowFindingsHint = narrow
    ? ' Prioritize findings that directly answer the user\'s exact question; omit tangential industry themes.'
    : '';
  if (fw) {
    narrowFindingsHint +=
      ' FRAMEWORK STRICT: each key_finding must be grounded in a passage that explicitly names the framework acronym or its standard full spelling. If NO passage qualifies, return key_findings as [] and barriers as [] — do not fill themes with analogies to other models.';
  }
  const system = `You are an Evidence Extraction agent. From the provided excerpts, extract structured content as JSON. Be thorough: include all significant findings.
- "key_findings": array of strings (main findings, each ending with APA in-text citation using author surnames, e.g. "Finding (Chen et al., 2025)" — never use journal names like (Mathematics, 2025) or (Buildings, 2025); avoid "Source N" labels; aim for ${narrow ? '4–15' : '8–20'} findings where evidence supports${narrowFindingsHint})
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
  const fw = options.framework_strict === true;
  const fwExtra = fw
    ? '\nIf key_findings is empty or none cite the named framework explicitly, set themes to one item: "No explicit framework-specific evidence in corpus" and implications to []. Do not invent TOE/TAM dimensions from other models.'
    : '';
  const system = `You are a Theme Synthesis agent. Given the user question and extracted evidence, output JSON with:
- "themes": array of 2-5 theme strings (concise)
- "maturity_signal": one of "Emerging" | "Growth" | "Mature" | "Decline" with one-sentence justification
- "implications": array of 1-3 strategic implication strings
Output only valid JSON.${fwExtra}`;
  const user = `Question: ${query}\n\nKey findings: ${(extractions.key_findings || []).join(' | ')}\nBarriers: ${(extractions.barriers || []).join(' | ')}\nThemes from evidence: ${(extractions.themes || []).join(' | ')}`;
  const out = await llmCall(system, user, options);
  const parsed = parseJSON(out);
  if (parsed) {
    return parsed;
  }
  return { themes: [], maturity_signal: 'Unknown', implications: [] };
}

/** Report Generation Agent: final structured report with flexible, agent-chosen sections. */
async function reportGenerationAgent(query, themes, extractions, chunks, options = {}) {
  const cfg = options.config || getConfig();
  const identity = cfg.system_identity || 'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.';
  const evidence = chunks
    .slice(0, 30)
    .map((c, i) => {
      const authors = joinAuthorsApaStyle(splitAuthorField(c.authors || ''));
      const year = (c.year || '').trim() ? ` (${c.year}).` : '.';
      const title = (c.title || '').trim() || (c.source || 'Untitled');
      const doi = (c.doi || '').trim().replace(/^https?:\/\/doi\.org\//i, '');
      const url = doi ? ` https://doi.org/${doi}` : '';
      return `[Source ${i + 1}: ${authors}${year} ${title}.${url}]\n${sanitizeChunkTextForLlm(c.text, 400)}`;
    })
    .join('\n\n');
  const maxEvidenceChars = (cfg.deep_research || {}).max_evidence_chars ?? MAX_EVIDENCE_CHARS;
  const evidenceTrim = evidence.slice(0, maxEvidenceChars);
  const reportMaxTokens = (cfg.deep_research || {}).report_max_tokens ?? DEFAULT_REPORT_MAX_TOKENS;

  const fwStrict = options.framework_strict === true;
  const fwLabels = labelsForMatchingFrameworks(query);
  const fwStats = options.framework_mention_stats || { total: 0, withMention: 0 };
  const strictFrameworkBlock = fwStrict
    ? `\n\n${frameworkStrictPromptFragment(fwLabels)}\n\nAutomated excerpt check: ${fwStats.withMention} of ${fwStats.total} evidence chunks contain an explicit match for the framework acronym or its standard full name in the question. If this count is 0, you MUST say the corpus does not support listing framework-specific criteria.\n`
    : '';

  const CITATION_RULES = `In-text citation rules (apply throughout):
- Use APA format: (Author, Year) or (Author et al., Year) — author surnames only
- Two authors: (Smith & Lee, 2024); three or more: (Smith et al., 2024)
- Do NOT use "Source N" style in the report body
- Never cite journal/periodical names as authors (e.g. not "(Buildings, 2025)" or "(Mathematics, 2025)")
- Never cite article titles as in-text references (e.g. not "(Generative AI in architectural design, 2025)")
- Never cite volume/page strings like "Mathematics 2025, 13, 1779" — that is not an author
- Do not add placeholder DOI links (https://doi.org/...) — only include full DOIs shown in the evidence
- If no author/year: use ("Short Title", n.d.)
- Use ONLY the evidence provided; do not invent facts or citations`;

  // Hint about question type for the agent
  const qType = options.report_type === 'roadmap'
    ? 'roadmap/timeline'
    : options.narrow_query
      ? 'specific factual / criteria / method definition'
      : 'broad research / technology survey';

  const system = `You are ${identity}.

Write a deep research report that directly and fully answers the user's question using ONLY the provided evidence. Do not invent facts or citations.

QUESTION TYPE DETECTED: ${qType}
${strictFrameworkBlock}
FLEXIBLE STRUCTURE — choose section headings that best serve this specific question. Do not apply a rigid template. Guidelines by question type:

• For criteria / method / definition questions (e.g. "what are the VIKOR criteria"):
  - Start with "## Direct Answer" — concise numbered/bulleted answer to the exact question
  - Follow with "## Supporting Evidence" if useful detail exists
  - Omit technology-survey sections (barriers, roadmap, implications) unless directly relevant

• For technology adoption / trend questions:
  - Use sections like "## Executive Summary", "## Key Findings", "## Adoption Landscape", "## Barriers", "## Strategic Implications" as the evidence supports
  - Include only sections you can fill with evidence

• For roadmap / maturity questions:
  - Include "## Maturity Assessment", "## Evolutionary Timeline", "## Critical Enablers"
  - Be explicit about timeframes when evidence supports them

• For any question type:
  - Create your own section headings if they better capture the answer
  - Omit any section the evidence does not support — leave nothing empty or vague
  - Always end with "## APA References" listing every source cited

${CITATION_RULES}`;

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
  const narrow_query = isNarrowFactualQuery(query);
  const framework_strict = isFrameworkStrictQuery(query);
  const opts = {
    config,
    api_key: apiKey,
    narrow_query,
    framework_strict,
    model: options.model,
  };
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
  if (framework_strict) {
    const boosts = getFrameworkRetrievalBoosts(query);
    for (const bq of boosts.slice(0, 3)) {
      const extra = await retrieve(bq, { ...opts, top_k: 12 });
      chunksR1 = dedupeChunksById([...chunksR1, ...extra]);
    }
  }

  onStatus('Evidence Screening Agent');
  const screening = await evidenceScreeningAgent(query, chunksR1, opts);
  let kept;
  if (screening.keep_indices.length > 0) {
    kept = screening.keep_indices.map((i) => chunksR1[i]).filter(Boolean);
  } else if (framework_strict) {
    kept = chunksR1.filter((c) =>
      chunkExplicitlyMentionsFramework(query, `${c.title || ''} ${c.text || ''}`),
    );
  } else {
    kept = chunksR1;
  }

  onStatus('Retrieval Round 2 (targeted)');
  let chunksR2 = [];
  for (const tq of (screening.targeted_queries || []).slice(0, MAX_TARGETED_QUERIES)) {
    const more = await retrieve(tq, { ...opts, top_k: topK_R2 });
    chunksR2 = dedupeChunksById([...chunksR2, ...more]);
  }
  const combined = dedupeChunksById([...kept, ...chunksR2]);
  const framework_mention_stats = framework_strict
    ? countChunksWithFrameworkMention(query, combined)
    : { total: 0, withMention: 0 };

  onStatus('Evidence Extraction Agent');
  const extractions = await evidenceExtractionAgent(combined, query, opts);

  onStatus('Theme Synthesis Agent');
  const themes = await themeSynthesisAgent(query, extractions, opts);

  onStatus('Report Generation Agent');
  const finalReportType = options.report_type === 'roadmap' ? 'roadmap' 
                        : (options.report_type === 'standard' ? 'standard' 
                        : (plan.report_type || 'standard'));
  const report = await reportGenerationAgent(query, themes, extractions, combined, {
    ...opts,
    report_type: finalReportType,
    narrow_query,
    framework_mention_stats,
  });

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
