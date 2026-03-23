/**
 * Shared model resolution for /api/query, /api/deep-research, and other chat routes.
 *
 * Wire IDs are aligned with public/index.html <select id="model-select">:
 *   gpt-4o, gpt-4o-mini, GPT-5, GPT5-mini, GPT5-nano
 *
 * - OpenAI-style 4.x models: lowercase (gpt-4o, …) for case-sensitive gateways.
 * - 5.x / gateway-branded: exact casing GPT-5, GPT5-mini, GPT5-nano (as on your aggregator).
 *
 * Synonyms (e.g. gpt-5-nano, GPT5nano, gpt5-nano) normalize to the same wire ID.
 */

function cleanModelString(s) {
  if (typeof s !== 'string') return '';
  return s
    .normalize('NFKC')
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    // Unify unicode dashes so "GPT5‑nano" (U+2011) still maps to gpt5nano
    .replace(/[\u2010\u2011\u2012\u2013\u2014\u2212]/g, '-')
    .trim();
}

/** Alphanumeric-only lowercase key for fuzzy matching (gpt-4o-mini → gpt4omini). */
function compactKey(s) {
  return cleanModelString(s).toLowerCase().replace(/[^a-z0-9]/g, '');
}

/**
 * Map compact keys → canonical wire id sent to the API.
 * Keep in sync with public/index.html model dropdown values.
 */
const COMPACT_TO_WIRE = {
  gpt4o: 'gpt-4o',
  gpt4omini: 'gpt-4o-mini',
  gpt4: 'gpt-4',
  gpt4turbo: 'gpt-4-turbo',
  gpt35turbo: 'gpt-3.5-turbo',
  gpt5: 'GPT-5',
  gpt5mini: 'GPT5-mini',
  gpt5nano: 'GPT5-nano',
};

/** Exact display aliases (case-sensitive key) → wire; used before compact match. */
const EXACT_ALIAS = {
  'GPT-4o': 'gpt-4o',
  'GPT-4o-mini': 'gpt-4o-mini',
  'GPT-4': 'gpt-4',
  'GPT-4-turbo': 'gpt-4-turbo',
  'GPT-3.5-turbo': 'gpt-3.5-turbo',
};

/** Canonical list for docs / UI parity checks. */
export const STANDARD_WIRE_CHAT_MODELS = ['gpt-4o', 'gpt-4o-mini', 'GPT-5', 'GPT5-mini', 'GPT5-nano'];

/**
 * Resolve the final upstream model ID.
 * @param {string|undefined} uiModel - from request body (e.g. model / model_selection)
 * @param {string|undefined} configModel - from settings.yaml / env
 * @returns {string} wire model id
 */
export function resolveModel(uiModel, configModel) {
  const ui = cleanModelString(uiModel);
  const cfg = cleanModelString(configModel);
  const raw = ui || cfg || 'gpt-4o';

  if (EXACT_ALIAS[raw]) return EXACT_ALIAS[raw];

  const ck = compactKey(raw);
  if (COMPACT_TO_WIRE[ck]) return COMPACT_TO_WIRE[ck];

  // Case-insensitive match on EXACT_ALIAS keys
  for (const [k, v] of Object.entries(EXACT_ALIAS)) {
    if (k.toLowerCase() === raw.toLowerCase()) return v;
  }

  // Only normalize classic OpenAI 3.x/4.x ids — do NOT lowercase gpt-5* (handled above via COMPACT_TO_WIRE).
  if (/^gpt-[34]/i.test(raw) || /^gpt-3\./i.test(raw)) return raw.toLowerCase();

  return raw;
}
