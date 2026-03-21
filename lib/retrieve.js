/**
 * In-memory retrieval: load index (JSON with embeddings + metadata), embed query, cosine similarity, top-k with per-source cap.
 * Index format: { chunks: [ { e: number[], text, source, title, year, authors, doi, apa_citation, chunk_id, ... } ] }
 * or from URL with same shape.
 */
import { getConfig, getApiKey } from './config.js';
import { vercelBlobFetchHeaders } from './vercel-blob-fetch.js';
import OpenAI from 'openai';

const defaultConfig = getConfig();

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  return Math.sqrt(dot(a, a)) || 1e-10;
}

function cosineSimilarity(a, b) {
  return dot(a, b) / (norm(a) * norm(b));
}

function l2Normalize(vec) {
  const n = norm(vec);
  return vec.map((x) => x / n);
}

let cachedIndex = null;
let cachedIndexUrl = null;

/**
 * Load index from INDEX_URL (fetch) or INDEX_PATH (file). Cached per process (Vercel serverless reuse).
 */
export async function loadIndex(config = defaultConfig) {
  const url = config.index_url;
  const path = config.index_path;
  if (url) {
    if (cachedIndex && cachedIndexUrl === url) return cachedIndex;
    const res = await fetch(url, { headers: vercelBlobFetchHeaders(url) });
    if (!res.ok) {
      const hint =
        res.status === 401 || res.status === 403
          ? ' For private Vercel Blob, set BLOB_READ_WRITE_TOKEN in env.'
          : '';
      throw new Error(`Failed to fetch index: ${res.status} ${res.statusText}.${hint}`);
    }
    const raw = await res.text();
    const data = JSON.parse(raw);
    cachedIndex = Array.isArray(data.chunks) ? data.chunks : data;
    cachedIndexUrl = url;
    return cachedIndex;
  }
  const fs = await import('fs');
  const pathMod = await import('path');
  const { fileURLToPath } = await import('url');
  const indexFileName = path ? path : pathMod.join('index', 'node_index.json');
  const projectRootFromLib = pathMod.resolve(pathMod.dirname(fileURLToPath(import.meta.url)), '..');
  const candidates = [
    pathMod.resolve(process.cwd(), indexFileName),
    pathMod.resolve(projectRootFromLib, indexFileName),
  ];
  let fullPath = null;
  for (const p of candidates) {
    if (fs.existsSync(p)) {
      fullPath = p;
      break;
    }
  }
  if (!fullPath) {
    throw new Error(
      path
        ? `Index file not found. Tried: ${candidates.join(', ')}`
        : 'No index available. Set INDEX_URL or INDEX_PATH, or place index/node_index.json (run: npm run export-index).'
    );
  }
  if (cachedIndex && cachedIndexUrl === fullPath) return cachedIndex;
  const content = fs.readFileSync(fullPath, 'utf8');
  const data = JSON.parse(content);
  cachedIndex = Array.isArray(data.chunks) ? data.chunks : data;
  cachedIndexUrl = fullPath;
  return cachedIndex;
}

export async function embedQuery(query, model, apiKey, baseUrl) {
  const client = new OpenAI({
    apiKey: apiKey || getApiKey(),
    ...(baseUrl ? { baseURL: baseUrl } : {}),
  });
  const res = await client.embeddings.create({ input: [query], model });
  return res.data[0].embedding;
}

function applyMetadataFilter(chunks, technology_L1, technology_L2, lifecycle_stage) {
  if (!technology_L1 && !technology_L2 && !lifecycle_stage) return chunks.map((_, i) => i);
  const allowed = [];
  for (let i = 0; i < chunks.length; i++) {
    const m = chunks[i];
    if (technology_L1 && (m.technology_L1 || '').toLowerCase() !== technology_L1.toLowerCase()) continue;
    if (technology_L2 && (m.technology_L2 || '').toLowerCase() !== technology_L2.toLowerCase()) continue;
    if (lifecycle_stage && (m.lifecycle_stage || '').toLowerCase() !== lifecycle_stage.toLowerCase()) continue;
    allowed.push(i);
  }
  return allowed;
}

/**
 * Retrieve top-k chunks by semantic similarity. Optional filters; cap per source.
 * Each chunk in index must have .e (embedding) and metadata (text, source, title, ...).
 */
export async function retrieve(query, options = {}) {
  const config = options.config || defaultConfig;
  const apiKey = options.api_key || getApiKey();
  if (!apiKey) throw new Error('OPENAI_API_KEY (or llm.api_key) is required.');

  const top_k = options.top_k ?? config.retrieval.top_k;
  const max_per_source = options.max_chunks_per_source ?? config.retrieval.max_chunks_per_source;
  const threshold = options.min_similarity_threshold ?? config.retrieval.min_similarity_threshold;
  const technology_L1 = options.technology_L1 ?? null;
  const technology_L2 = options.technology_L2 ?? null;
  const lifecycle_stage = options.lifecycle_stage ?? null;

  const chunks = await loadIndex(config);
  if (chunks.length === 0) return [];

  const allowedIndices = applyMetadataFilter(chunks, technology_L1, technology_L2, lifecycle_stage);
  if (allowedIndices.length === 0) return [];

  const embeddingModel = config.embedding?.model || 'text-embedding-3-small';
  const baseUrl = config.llm?.base_url || null;
  const qvec = l2Normalize(await embedQuery(query, embeddingModel, apiKey, baseUrl));

  const withScore = [];
  for (const i of allowedIndices) {
    const c = chunks[i];
    const e = c.e || c.embedding;
    if (!e || e.length !== qvec.length) continue;
    const score = cosineSimilarity(qvec, e);
    if (threshold != null && threshold > 0 && score < threshold) continue;
    withScore.push({ index: i, score, meta: c });
  }
  withScore.sort((a, b) => b.score - a.score);

  const sourceCount = {};
  const results = [];
  // Scan enough candidates to fill top_k across sources (one paper can dominate top scores)
  const searchK = Math.min(withScore.length, Math.max(top_k * 10, 200));
  for (let j = 0; j < searchK && results.length < top_k; j++) {
    const { score, meta } = withScore[j];
    const source = meta.source || '';
    if (sourceCount[source] >= max_per_source) continue;
    sourceCount[source] = (sourceCount[source] || 0) + 1;
    const out = { ...meta };
    delete out.e;
    delete out.embedding;
    out._score = score;
    results.push(out);
  }
  return results;
}
