/**
 * POST /api/deep-research — Multi-agent deep research pipeline.
 * Body: { query: string, stream?: boolean }
 * If stream=true: SSE stream of { status } then { done: true, response, sources_used }.
 * Else: JSON { query, response, sources_used }.
 */
import { runDeepResearch } from '../lib/deep-research.js';
import { getConfig, getApiKey } from '../lib/config.js';
import { resolveModel } from '../lib/chat-models.js';

function sendEvent(res, data) {
  res.write('data: ' + JSON.stringify(data) + '\n\n');
}

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

  const stream = body.stream === true;
  const report_type = typeof body.report_type === 'string' ? body.report_type : 'auto';
  const config = getConfig();
  const llm = config.llm || {};
  const bodyModel = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : undefined;
  const model = resolveModel(bodyModel, llm.model);

  if (stream) {
    if (res.socket && res.socket.setNoDelay) res.socket.setNoDelay(true);
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    if (typeof res.flushHeaders === 'function') res.flushHeaders();
    // SSE comment (2KB) so first chunk is sent immediately and client starts reading (avoids Node/browser buffering)
    res.write(': ' + ' '.repeat(2048) + '\n\n');
    sendEvent(res, { status: 'Starting…' });

    runDeepResearch(query, {
      api_key: apiKey,
      report_type,
      model,
      config,
      onStatus: (status) => sendEvent(res, { status }),
    })
      .then(({ response, sources_used }) => {
        sendEvent(res, { done: true, query, response, sources_used: sources_used || [] });
        res.end();
      })
      .catch((err) => {
        sendEvent(res, { error: err.message || String(err) });
        res.end();
      });
    return;
  }

  try {
    const { response, sources_used } = await runDeepResearch(query, {
      api_key: apiKey,
      report_type,
      model,
      config,
    });
    res.setHeader('X-Resolved-Chat-Model', model);
    return res.status(200).json({
      query,
      response,
      sources_used: sources_used || [],
    });
  } catch (err) {
    const message = err.message || String(err);
    return res.status(500).json({ detail: message });
  }
}
