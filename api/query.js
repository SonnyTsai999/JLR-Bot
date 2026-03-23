/**
 * POST /api/query — Standard RAG: retrieve → direct-answer synthesis.
 * Body: { query: string, top_k?, technology_L1?, technology_L2?, lifecycle_stage?, model? }
 */
import { retrieve } from '../lib/retrieve.js';
import { synthesize } from '../lib/synthesize.js';
import { getConfig, getApiKey } from '../lib/config.js';
import { resolveModel } from '../lib/chat-models.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  let body;
  try {
    body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body || {};
  } catch {
    return res.status(422).json({ detail: 'Invalid JSON body' });
  }

  const query = typeof body.query === 'string' ? body.query.trim() : '';
  if (!query) {
    return res.status(422).json({ detail: "Field 'query' is required and must be a non-empty string" });
  }

  const apiKey = getApiKey();
  if (!apiKey) {
    return res.status(500).json({ detail: 'API key not set. Set OPENAI_API_KEY in environment.' });
  }

  try {
    const config = getConfig();

    const chunks = await retrieve(query, {
      config,
      api_key: apiKey,
      top_k: body.top_k ?? null,
      technology_L1: body.technology_L1 ?? null,
      technology_L2: body.technology_L2 ?? null,
      lifecycle_stage: body.lifecycle_stage ?? null,
    });

    const llm = config.llm || {};
    const bodyModel = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : undefined;
    const chatModel = resolveModel(bodyModel, llm.model);

    const answer = await synthesize(query, chunks, {
      config,
      api_key: apiKey,
      model: chatModel,
    });

    const seen = new Set();
    const sources_used = [];
    for (const c of chunks) {
      const key = (c.doi || '').trim() || (c.apa_citation || '').trim();
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

    res.setHeader('X-Resolved-Chat-Model', chatModel);
    return res.status(200).json({ query, response: answer, sources_used });
  } catch (err) {
    const message = err.message || String(err);
    return res.status(500).json({ detail: message });
  }
}
