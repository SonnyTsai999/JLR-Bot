/**
 * Build chat.completions.create() parameters consistently for standard + deep-research.
 * Some OpenAI-compatible gateways expect `max_completion_tokens` (not `max_tokens`) for GPT-5 family;
 * using the wrong field can surface as 403 / model errors on aggregators.
 */

/**
 * @param {string} modelId - wire model id after resolveModel()
 * @returns {boolean}
 */
export function isGpt5FamilyModelId(modelId) {
  const m = (modelId || '').trim();
  if (!m) return false;
  if (/^GPT5/i.test(m)) return true;
  if (/^GPT-5/i.test(m)) return true;
  if (/^gpt-5/i.test(m)) return true;
  return false;
}

/**
 * @param {string} modelId
 * @param {object} llm - config.llm
 * @param {Array<{role:string,content:string}>} messages
 * @param {number | undefined} maxTokensOverride
 * @returns {object} spread into client.chat.completions.create(...)
 */
export function chatCompletionsPayload(modelId, llm, messages, maxTokensOverride) {
  const temperature = llm?.temperature ?? 0.3;
  const mt = maxTokensOverride ?? llm?.max_tokens ?? 2048;
  const base = {
    model: modelId,
    messages,
    temperature,
  };
  const legacy =
    process.env.LEGACY_CHAT_MAX_TOKENS === '1' || process.env.LEGACY_CHAT_MAX_TOKENS === 'true';
  if (!legacy && isGpt5FamilyModelId(modelId)) {
    return { ...base, max_completion_tokens: mt };
  }
  return { ...base, max_tokens: mt };
}
