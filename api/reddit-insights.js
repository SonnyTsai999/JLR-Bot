import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import { getConfig, getApiKey } from '../lib/config.js';
import { vercelBlobFetchHeaders } from '../lib/vercel-blob-fetch.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Simple vector dot product
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

let cachedRedditIndex = null;
/** @type {string | null} */
let cachedRedditIndexSource = null;

/**
 * Load chunks from REDDIT_INDEX_URL / config.reddit_index_url, else index/reddit_index.json.
 * The JSON is ~100MB+ and is gitignored — not on GitHub; use URL for serverless.
 */
async function loadRedditIndex(config) {
  const url = process.env.REDDIT_INDEX_URL || config?.reddit_index_url;
  const indexPath = path.join(__dirname, '..', 'index', 'reddit_index.json');

  if (url) {
    const key = `url:${url}`;
    if (cachedRedditIndex && cachedRedditIndexSource === key) return cachedRedditIndex;
    const res = await fetch(url, {
      headers: { Accept: 'application/json', ...vercelBlobFetchHeaders(url) },
    });
    if (!res.ok) {
      const hint =
        res.status === 401 || res.status === 403
          ? ' For private Vercel Blob, add BLOB_READ_WRITE_TOKEN to env (Vercel → Storage), or re-upload as public read.'
          : '';
      throw new Error(
        `Reddit index fetch failed (${res.status} ${res.statusText}). Check REDDIT_INDEX_URL.${hint}`
      );
    }
    const data = await res.json();
    cachedRedditIndex = Array.isArray(data.chunks) ? data.chunks : data;
    cachedRedditIndexSource = key;
    return cachedRedditIndex;
  }

  const key = `file:${indexPath}`;
  if (cachedRedditIndex && cachedRedditIndexSource === key) return cachedRedditIndex;

  if (!fs.existsSync(indexPath)) {
    throw new Error(
      'Reddit index not found. Locally: run `npm run embed:reddit` (needs sentiment/*.csv + OPENAI_API_KEY). ' +
        'Production: upload index/reddit_index.json to blob/HTTPS storage and set env REDDIT_INDEX_URL ' +
        '(file is gitignored and >100MB — GitHub rejects it without Git LFS).'
    );
  }

  const content = fs.readFileSync(indexPath, 'utf8');
  const data = JSON.parse(content);
  cachedRedditIndex = Array.isArray(data.chunks) ? data.chunks : data;
  cachedRedditIndexSource = key;
  return cachedRedditIndex;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') return res.status(405).json({ detail: 'Method not allowed' });

  let body = req.body || {};
  const { tech_tag, comment_count, model_selection } = body;
  
  if (!tech_tag) {
    return res.status(400).json({ detail: 'tech_tag is required' });
  }

  const numComments = parseInt(comment_count, 10) || 3;

  const apiKey = getApiKey();
  if (!apiKey) {
    return res.status(500).json({ detail: 'API key not set.' });
  }

  try {
    const config = getConfig();
    const llmConfig = config.llm || {};
    const model = model_selection || llmConfig.model || 'gpt-5-nano';
    const embModel = config.embedding?.model || 'text-embedding-3-small';
    
    const client = new OpenAI({
      apiKey,
      ...(llmConfig.base_url ? { baseURL: llmConfig.base_url } : {}),
    });

    // 1. Load index
    console.log("Loading reddit index...");
    const chunks = await loadRedditIndex(config);

    // 2. Filter by tech_tag
    const filteredChunks = chunks.filter(c => c.tech_tag === tech_tag);
    if (filteredChunks.length === 0) {
      return res.status(200).json({
        summary: `No discussions found for ${tech_tag}.`,
        screened_comments: []
      });
    }

    // 3. Embed query
    console.log(`Embedding query for ${tech_tag}...`);
    const query = `What are the core adoption barriers, challenges, potential opportunities, and practical use-cases for ${tech_tag}?`;
    
    const embRes = await client.embeddings.create({ input: [query], model: embModel });
    const queryVector = embRes.data[0].embedding;

    // 4. Retrieve top K
    console.log("Calculating similarities...");
    for (const c of filteredChunks) {
      c.sim = cosineSimilarity(queryVector, c.e);
    }
    
    filteredChunks.sort((a, b) => b.sim - a.sim);
    const topChunks = filteredChunks.slice(0, 30); // Grab top 30 most relevant chunks

    // 5. Send to LLM
    console.log("Generating insights with LLM...");
    const systemPrompt = `You are a Technology Strategy Analyst specializing in the AEC (Architecture, Engineering, Construction) and manufacturing industries.

You will be provided with a JSON array of highly relevant Reddit posts and comments regarding the technology: ${tech_tag}.

Your task is to analyze these specific texts and extract insights relevant to enterprise adoption readiness in the AEC industry. Do NOT mention specific company names.

Return ONLY valid JSON in this exact structure:
{
  "barriers_summary": "Markdown string (1-2 paragraphs). Summarize the core adoption barriers, specific implementation challenges, and cultural pushback.",
  "opportunities_summary": "Markdown string (1-2 paragraphs). Summarize the potential opportunities, benefits, and positive workflow impacts.",
  "use_cases_summary": "Markdown string (1-2 paragraphs). Summarize the specific, practical use-cases and workflows where this technology is actually being applied by users.",
  "screened_comments": [
    {
      "body": "original text quote",
      "insight": "1-sentence explanation of why this barrier, opportunity, or use-case is relevant to the AEC industry",
      "score": "upvote score if available",
      "permalink": "exact permalink provided in the input for this quote"
    }
  ]
}

Note for screened_comments: Return exactly ${numComments} MOST informative and insightful quotes that provide real-world evidence for the summaries.`;

    const userMessage = JSON.stringify(topChunks.map(c => ({
      id: c.id,
      subreddit: c.subreddit,
      text: c.text,
      score: c.score,
      permalink: "https://reddit.com/r/" + c.subreddit + "/comments/" + c.post_id + "/",
      relevance_score: c.sim.toFixed(3)
    })), null, 2);

    const resp = await client.chat.completions.create({
      model: model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
      temperature: 0.2,
      response_format: { type: "json_object" }
    });

    const responseText = (resp.choices[0]?.message?.content ?? '').trim();
    
    let parsed;
    try {
      parsed = JSON.parse(responseText);
    } catch (e) {
      console.error("JSON Parsing Error", e);
      return res.status(500).json({ detail: "Failed to parse JSON from LLM" });
    }

    return res.status(200).json(parsed);

  } catch (err) {
    console.error("Reddit Insights Error:", err);
    return res.status(500).json({ detail: err.message || String(err) });
  }
}
