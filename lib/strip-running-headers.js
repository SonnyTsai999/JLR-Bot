/**
 * MDPI (and similar) PDFs repeat journal running heads in chunk text, e.g.
 *   "Mathematics 2025, 13, 1779 2 of 38"
 *   "Mathematics 2025, 13, 1779" (inline)
 * Models wrongly cite these as (Mathematics, 2025) or (Mathematics 2025, 13, 1779).
 */
const MDPI_STYLE_LINE =
  /^[A-Za-z][A-Za-z\s&-]{0,55} \d{4}, \d+, \d+(?: \d+ of \d+)?\s*\r?\n/;

/** Same pattern anywhere in the string (headers mid-paragraph / after strip). */
const MDPI_STYLE_GLOBAL = /[A-Za-z][A-Za-z\s&-]{0,55} \d{4}, \d+, \d+(?: \d+ of \d+)?/g;

export function stripMdpiRunningHeadLines(text) {
  if (!text || typeof text !== 'string') return text;
  let s = text;
  for (let i = 0; i < 4; i++) {
    const m = s.match(MDPI_STYLE_LINE);
    if (!m) break;
    s = s.slice(m[0].length);
  }
  s = s.replace(MDPI_STYLE_GLOBAL, ' ');
  s = s.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').replace(/ {2,}/g, ' ');
  return s.trimStart();
}

/** Use before sending chunk text to the LLM (RAG evidence bodies). */
export function sanitizeChunkTextForLlm(text, maxLen) {
  const stripped = stripMdpiRunningHeadLines(text || '');
  const slice = maxLen != null ? stripped.slice(0, maxLen) : stripped;
  return slice;
}
