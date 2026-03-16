/**
 * Config loader for Node/Vercel.
 * Priority: environment variables > config/settings.yaml > hardcoded defaults.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');
const SETTINGS_PATH = path.join(PROJECT_ROOT, 'config', 'settings.yaml');

const DEFAULTS = {
  embedding: { model: 'text-embedding-3-small', dimensions: 1536 },
  retrieval: { top_k: 10, max_chunks_per_source: 3, min_similarity_threshold: 0.0 },
  llm: { model: 'gpt-4o', temperature: 0.3, max_tokens: 2048, base_url: null, api_key: null },
  system_identity: 'Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.',
  index_path: null,
  index_url: null,
};

let cachedYaml = null;

function loadYamlConfig() {
  if (cachedYaml !== null) return cachedYaml;
  try {
    if (!fs.existsSync(SETTINGS_PATH)) {
      cachedYaml = {};
      return cachedYaml;
    }
    const raw = fs.readFileSync(SETTINGS_PATH, 'utf8');
    cachedYaml = yaml.load(raw) || {};
    return cachedYaml;
  } catch {
    cachedYaml = {};
    return cachedYaml;
  }
}

function envInt(name, fallback) {
  const v = process.env[name];
  if (v == null || v === '') return fallback;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

function envFloat(name, fallback) {
  const v = process.env[name];
  if (v == null || v === '') return fallback;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

export function getConfig() {
  const y = loadYamlConfig();
  const yEmbedding = y.embedding || {};
  const yRetrieval = y.retrieval || {};
  const yLlm = y.llm || {};

  return {
    embedding: {
      model: process.env.EMBEDDING_MODEL || yEmbedding.model || DEFAULTS.embedding.model,
      dimensions: envInt('EMBEDDING_DIMENSIONS', yEmbedding.dimensions ?? DEFAULTS.embedding.dimensions),
    },
    retrieval: {
      top_k: envInt('RETRIEVAL_TOP_K', yRetrieval.top_k ?? DEFAULTS.retrieval.top_k),
      max_chunks_per_source: envInt('RETRIEVAL_MAX_PER_SOURCE', yRetrieval.max_chunks_per_source ?? DEFAULTS.retrieval.max_chunks_per_source),
      min_similarity_threshold: envFloat('RETRIEVAL_MIN_SIMILARITY', yRetrieval.min_similarity_threshold ?? DEFAULTS.retrieval.min_similarity_threshold),
    },
    llm: {
      model: process.env.CHAT_MODEL || process.env.LLM_MODEL || yLlm.model || DEFAULTS.llm.model,
      temperature: envFloat('LLM_TEMPERATURE', yLlm.temperature ?? DEFAULTS.llm.temperature),
      max_tokens: envInt('LLM_MAX_TOKENS', yLlm.max_tokens ?? DEFAULTS.llm.max_tokens),
      base_url: process.env.OPENAI_BASE_URL || yLlm.base_url || DEFAULTS.llm.base_url,
      api_key: process.env.OPENAI_API_KEY || yLlm.api_key || DEFAULTS.llm.api_key,
    },
    system_identity: process.env.SYSTEM_IDENTITY || y.system_identity || DEFAULTS.system_identity,
    /** Local path to node_index.json or URL to fetch (Vercel Blob/CDN). */
    index_path: process.env.INDEX_PATH || y.index_path || DEFAULTS.index_path,
    index_url: process.env.INDEX_URL || y.index_url || DEFAULTS.index_url,
  };
}

export function getApiKey() {
  return process.env.OPENAI_API_KEY || (loadYamlConfig().llm || {}).api_key || null;
}
