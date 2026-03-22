/**
 * Author strings from metadata are usually semicolon-separated (MDPI/Scopus):
 *   "Wan, H.; Zhang, J.; Chen, Y.; Xu, W.; Feng, F."
 * Do NOT split on every comma — that breaks "Wan, H." into "Wan" and "H.".
 */

export function splitAuthorField(raw) {
  if (raw == null || typeof raw !== 'string') return [];
  const t = raw.trim();
  if (!t) return [];
  if (t.includes(';')) {
    return t
      .split(';')
      .map((p) => p.trim())
      .filter(Boolean);
  }
  return [t];
}

/** One line for evidence headers: "A, B., & C, D." */
export function joinAuthorsApaStyle(parts) {
  if (!parts.length) return 'Unknown';
  if (parts.length === 1) return parts[0];
  return parts.slice(0, -1).join(', ') + ', & ' + parts[parts.length - 1];
}
