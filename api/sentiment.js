import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const baseDir = path.join(__dirname, '..', 'sentiment');
    
    // Read the CSV files
    const posts = fs.readFileSync(path.join(baseDir, 'reddit_posts.csv'), 'utf-8');
    const comments = fs.readFileSync(path.join(baseDir, 'reddit_comments.csv'), 'utf-8');
    const sentiment = fs.readFileSync(path.join(baseDir, 'post_sentiment.csv'), 'utf-8');
    const barriers = fs.readFileSync(path.join(baseDir, 'post_barriers.csv'), 'utf-8');

    return res.status(200).json({
      posts,
      comments,
      sentiment,
      barriers
    });
  } catch (err) {
    const message = err.message || String(err);
    return res.status(500).json({ detail: message });
  }
}
