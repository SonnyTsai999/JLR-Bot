import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Papa from 'papaparse';
import OpenAI from 'openai';
import { getConfig, getApiKey } from '../lib/config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiKey = getApiKey();
const config = getConfig();

if (!apiKey) {
  console.error("No OpenAI API key found.");
  process.exit(1);
}

const client = new OpenAI({
  apiKey,
  ...(config.llm?.base_url ? { baseURL: config.llm.base_url } : {})
});

async function embedRedditData() {
  const postsPath = path.join(__dirname, '..', 'sentiment', 'reddit_posts.csv');
  const commentsPath = path.join(__dirname, '..', 'sentiment', 'reddit_comments.csv');
  
  if (!fs.existsSync(postsPath) || !fs.existsSync(commentsPath)) {
    console.error("CSV files not found in sentiment/");
    return;
  }

  const postsCsv = fs.readFileSync(postsPath, 'utf8');
  const commentsCsv = fs.readFileSync(commentsPath, 'utf8');
  
  const posts = Papa.parse(postsCsv, { header: true, skipEmptyLines: true }).data;
  const comments = Papa.parse(commentsCsv, { header: true, skipEmptyLines: true }).data;

  const commentsByPost = new Map();
  for (const c of comments) {
    if (!commentsByPost.has(c.parent_post_id)) {
      commentsByPost.set(c.parent_post_id, []);
    }
    commentsByPost.get(c.parent_post_id).push(c);
  }

  const chunks = [];
  
  console.log(`Processing ${posts.length} posts...`);
  
  for (const p of posts) {
    const postComments = commentsByPost.get(p.post_id) || [];
    // Sort comments by score
    postComments.sort((a, b) => parseInt(b.score || 0) - parseInt(a.score || 0));
    
    // Create one chunk for the post body
    const postText = `Title: ${p.title}\nSubreddit: ${p.subreddit}\n\n${p.selftext || p.full_text || ''}`;
    chunks.push({
      id: `post_${p.post_id}`,
      post_id: p.post_id,
      tech_tag: p.tech_tag,
      subreddit: p.subreddit,
      text: postText,
      type: 'post',
      score: p.score
    });
    
    // Create chunks for the top comments (max 10 per post to avoid huge index)
    const topComments = postComments.slice(0, 10);
    for (const c of topComments) {
      if (!c.body || c.body.trim().length < 20) continue; // skip tiny comments
      
      const commentText = `Comment on post "${p.title}":\n\n${c.body}`;
      chunks.push({
        id: `comment_${c.comment_id}`,
        post_id: p.post_id,
        comment_id: c.comment_id,
        tech_tag: p.tech_tag,
        subreddit: p.subreddit,
        text: commentText,
        type: 'comment',
        score: c.score
      });
    }
  }

  console.log(`Created ${chunks.length} chunks. Getting embeddings...`);

  const batchSize = 100;
  const model = config.embedding?.model || 'text-embedding-3-small';

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    console.log(`Embedding batch ${i / batchSize + 1} / ${Math.ceil(chunks.length / batchSize)}`);
    
    const texts = batch.map(c => c.text);
    
    try {
      const resp = await client.embeddings.create({
        input: texts,
        model: model
      });
      
      for (let j = 0; j < resp.data.length; j++) {
        batch[j].e = resp.data[j].embedding;
      }
    } catch (err) {
      console.error("Embedding error:", err);
    }
  }

  const indexData = { chunks: chunks.filter(c => c.e) }; // only save those with embeddings
  
  const indexPath = path.join(__dirname, '..', 'index');
  if (!fs.existsSync(indexPath)) fs.mkdirSync(indexPath);
  
  const outPath = path.join(indexPath, 'reddit_index.json');
  fs.writeFileSync(outPath, JSON.stringify(indexData));
  
  console.log(`Saved Reddit index with ${indexData.chunks.length} embeddings to ${outPath}`);
}

embedRedditData();