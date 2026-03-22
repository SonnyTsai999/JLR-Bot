/**
 * Shared model resolution for all API paths.
 * Normalizes UI display names to API-compatible IDs (case-sensitive gateways).
 * Custom gateway models (GPT5-nano, GPT5-mini, GPT-5) pass through unchanged.
 */

const MODEL_ALIASES = {
  'GPT-4o':        'gpt-4o',
  'GPT-4o-mini':   'gpt-4o-mini',
  'GPT-4':         'gpt-4',
  'GPT-4-turbo':   'gpt-4-turbo',
  'GPT-3.5-turbo': 'gpt-3.5-turbo',
};

/**
 * Resolve the final upstream model ID.
 * @param {string|undefined} uiModel - model string from the request body (may be undefined)
 * @param {string|undefined} configModel - model string from settings.yaml / config
 * @returns {string} normalized model ID safe to send to the upstream API
 */
export function resolveModel(uiModel, configModel) {
  const raw = (
    (typeof uiModel === 'string' && uiModel.trim()) ||
    (typeof configModel === 'string' && configModel.trim()) ||
    'gpt-4o'
  );
  return MODEL_ALIASES[raw] || raw;
}
