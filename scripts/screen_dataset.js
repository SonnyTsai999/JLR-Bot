import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Papa from 'papaparse';
import OpenAI from 'openai';
import { getConfig, getApiKey } from '../lib/config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const config = getConfig();
const apiKey = getApiKey();

if (!apiKey) {
  console.error("No OpenAI API key found.");
  process.exit(1);
}

const client = new OpenAI({
  apiKey,
  ...(config.llm?.base_url ? { baseURL: config.llm.base_url } : {})
});

async function screenPosts() {
  const postsPath = path.join(__dirname, '..', 'sentiment', 'reddit_posts.csv');
  const postsCsv = fs.readFileSync(postsPath, 'utf8');
  const parsed = Papa.parse(postsCsv, { header: true, skipEmptyLines: true });
  const posts = parsed.data;

  console.log(`Loaded ${posts.length} posts to screen.`);
  
  const batchSize = 50;
  const validPostIds = new Set();

  for (let i = 0; i < posts.length; i += batchSize) {
    const batch = posts.slice(i, i + batchSize);
    console.log(`Processing batch ${i / batchSize + 1} of ${Math.ceil(posts.length / batchSize)}...`);
    
    const batchPayload = batch.map(p => ({
      id: p.post_id,
      title: p.title,
      subreddit: p.subreddit
    }));

    const prompt = `You are an AI filtering script. Review the following Reddit posts. 
Determine if each post is actually related to technology in Architecture, Engineering, Construction (AEC), Manufacturing, AI, software, or BIM.
Many posts are irrelevant noise (e.g. general construction accidents, jokes, personal finance, unrelated news).

Return a JSON object mapping each post ID to a boolean (true if relevant to technology/AEC/manufacturing/AI, false if irrelevant noise/unrelated).

Input:
${JSON.stringify(batchPayload, null, 2)}

Return ONLY valid JSON: { "results": { "id1": true, "id2": false, ... } }`;

    try {
      const resp = await client.chat.completions.create({
        model: config.llm?.model || "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        temperature: 0.1
      });

      const resJson = JSON.parse(resp.choices[0].message.content);
      if (resJson.results) {
        for (const [id, isRelevant] of Object.entries(resJson.results)) {
          if (isRelevant) {
            validPostIds.add(id);
          }
        }
      }
    } catch (err) {
      console.error(`Error processing batch ${i / batchSize + 1}:`, err);
      // fallback: keep them if error
      batch.forEach(p => validPostIds.add(p.post_id));
    }
  }

  console.log(`Screening complete. ${validPostIds.size} out of ${posts.length} posts deemed relevant.`);

  // Filter all CSVs
  const sentimentDir = path.join(__dirname, '..', 'sentiment');
  
  function filterCsv(filename, idField) {
    const filePath = path.join(sentimentDir, filename);
    if (!fs.existsSync(filePath)) return;
    
    const csv = fs.readFileSync(filePath, 'utf8');
    const p = Papa.parse(csv, { header: true, skipEmptyLines: true });
    const filtered = p.data.filter(row => validPostIds.has(row[idField]));
    
    const newCsv = Papa.unparse(filtered);
    const newPath = path.join(sentimentDir, `filtered_${filename}`);
    fs.writeFileSync(newPath, newCsv);
    console.log(`Saved ${filtered.length} rows to ${newPath}`);
  }

  filterCsv('reddit_posts.csv', 'post_id');
  filterCsv('reddit_comments.csv', 'parent_post_id');
  filterCsv('post_sentiment.csv', 'post_id');
  filterCsv('post_barriers.csv', 'post_id');

  console.log("Done! You can now rename the filtered_* files to overwrite the originals.");
}

screenPosts();