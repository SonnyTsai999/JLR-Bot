/**
 * Headers for fetching private Vercel Blob objects (same token as Project → Storage).
 */
export function vercelBlobFetchHeaders(url) {
  const headers = {};
  try {
    const host = new URL(url).hostname;
    if (host.includes('blob.vercel-storage.com')) {
      const token =
        process.env.BLOB_READ_WRITE_TOKEN ||
        process.env.VERCEL_BLOB_READ_WRITE_TOKEN ||
        process.env.BLOB_STORE_TOKEN;
      if (token) headers.Authorization = `Bearer ${token}`;
    }
  } catch {
    /* ignore */
  }
  return headers;
}
