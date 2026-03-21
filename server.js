/**
 * Local dev server: serves public/ and runs API routes (same as Vercel serverless).
 * Run: npm run dev  →  http://localhost:3000
 * Production: deploy to Vercel (no need for this file).
 */
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import queryHandler from './api/query.js';
import deepResearchHandler from './api/deep-research.js';
import healthHandler from './api/health.js';
import sentimentHandler from './api/sentiment.js';
import sentimentInsightsHandler from './api/sentiment-insights.js';
import redditInsightsHandler from './api/reddit-insights.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.PORT || '3000', 10);
const PUBLIC = path.join(__dirname, 'public');

function parseBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve({});
      }
    });
  });
}

/** Wrap Node ServerResponse so API handlers can use res.status().json() like on Vercel. */
function wrapRes(res) {
  res.status = function (code) {
    res.statusCode = code;
    return res;
  };
  res.json = function (obj) {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(obj));
    return res;
  };
  return res;
}

const server = http.createServer(async (req, res) => {
  const pathname = (req.url?.split('?')[0] || '/').replace(/\/+$/, '') || '/';
  const method = req.method;

  if (pathname === '/api/health' && method === 'GET') {
    return healthHandler(req, wrapRes(res));
  }
  if (pathname === '/api/query' && method === 'POST') {
    req.body = await parseBody(req);
    return queryHandler(req, wrapRes(res));
  }
  if (pathname === '/api/deep-research' && method === 'POST') {
    req.body = await parseBody(req);
    return deepResearchHandler(req, wrapRes(res));
  }
  if (pathname === '/api/sentiment' && method === 'GET') {
    return sentimentHandler(req, wrapRes(res));
  }
  if (pathname === '/api/sentiment-insights' && method === 'POST') {
    console.log("Routing to /api/sentiment-insights");
    req.body = await parseBody(req);
    return sentimentInsightsHandler(req, wrapRes(res));
  }
  if (pathname === '/api/reddit-insights' && method === 'POST') {
    console.log("Routing to /api/reddit-insights");
    req.body = await parseBody(req);
    return redditInsightsHandler(req, wrapRes(res));
  }
  if ((pathname === '/api/query' || pathname === '/api/deep-research' || pathname === '/api/sentiment-insights' || pathname === '/api/reddit-insights') && method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.writeHead(204).end();
  }

  let file = pathname === '/' ? '/index.html' : pathname;
  const filePath = path.join(PUBLIC, path.normalize(file));
  if (!filePath.startsWith(PUBLIC)) {
    res.writeHead(403).end();
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === 'ENOENT' && pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<h1>JLR Technology Intelligence</h1><p>Put index.html in the <code>public/</code> folder.</p>');
        return;
      }
      res.writeHead(404).end();
      return;
    }
    const ext = path.extname(filePath);
    const types = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.ico': 'image/x-icon', '.json': 'application/json' };
    res.setHeader('Content-Type', types[ext] || 'application/octet-stream');
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`JLR Technology Intelligence — http://localhost:${PORT}`);
  console.log('  API: POST /api/query, POST /api/deep-research (stream), GET /api/health');
});
