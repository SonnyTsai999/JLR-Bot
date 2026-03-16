# JLR Technology Intelligence Assistant

RAG-based **Technology Intelligence Assistant** for J.L. Richards (JLR): turns a curated academic journal corpus into structured, citation-grounded insights for emerging AEC design technologies. Decision-support only; not a general chatbot or document distribution system.

## Architecture

- **Ingest**: PDFs → text extraction → cleaning (no references/appendices) → chunking (500–800 tokens, 100 overlap) → metadata (source, title, year, technology_L1/L2, lifecycle_stage, APA, DOI).
- **Embed**: OpenAI `text-embedding-3-large` → FAISS index (stored under `index/`).
- **Retrieve**: Top-K semantic search with optional metadata filters; cap chunks per source to limit redundancy.
- **Synthesize**: LLM (e.g. GPT-4o) with a fixed structured template; evidence-only, no invented citations, APA references.
- **Interface**: Web UI + REST API.

See [SYSTEM_INTENT.md](SYSTEM_INTENT.md) for full development intent and constraints.

## Setup

1. **Clone and install**

   ```bash
   cd "d:\Shared Disk\Stage1\Prototype"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment**

   Set `OPENAI_API_KEY` for embedding and synthesis.

3. **Corpus**

   - Place peer-reviewed PDFs in `data/raw_pdfs/`.
   - Optionally add per-PDF metadata: next to `paper.pdf` create `paper.metadata.json` with keys: `title`, `year`, `technology_L1`, `technology_L2`, `lifecycle_stage`, `apa_citation`, `doi`.

4. **Config**

   Edit `config/settings.yaml` for paths, chunk sizes, model names, top_k, etc.

## Usage

**1. Ingest PDFs → chunks**

   ```bash
   python -m backend.ingest
   ```
   Output: `data/processed_chunks/chunks.jsonl`.

**2. Build FAISS index**

   ```bash
   set OPENAI_API_KEY=your_key
   python -m backend.embed
   ```
   Output: `index/faiss.index` and `index/metadata.jsonl`.

**3. Start API server**

   ```bash
   uvicorn backend.server:app --host 0.0.0.0 --port 8000
   ```

**4. Open frontend**

   Open `frontend/index.html` in a browser (or serve it). Use “Your question” and optional filters (Technology L1/L2, Lifecycle stage), then “Get analysis”. Responses follow the structured template and cite only retrieved evidence.

## API

- `GET /health` — service health.
- `POST /query` — body: `{ "query": "...", "technology_L1": null, "technology_L2": null, "lifecycle_stage": null, "top_k": null }`. Returns `{ "query", "response", "sources_used" }`.

## Constraints (summary)

- No full PDF exposure; no reconstruction of full papers.
- No external internet retrieval; no hallucinated citations.
- Responses are evidence-grounded, structured, and copyright-safe (excerpt limits, APA + DOI).
- System identity: *Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.*

## Node.js and Vercel deployment

The app can run on **Node.js** and deploy to **Vercel** (serverless).

### One-time: export index for Node

From the Python environment (after you have `index/faiss.index` and `index/metadata.jsonl`):

```bash
python scripts/export_index_for_node.py
```

This writes `index/node_index.json` (embeddings + metadata). If the file is large (> ~50MB), host it elsewhere (e.g. Vercel Blob) and set `INDEX_URL` in Vercel.

### Local Node dev

```bash
npm install
# Optional: set OPENAI_API_KEY and OPENAI_BASE_URL
npm run dev
```

Open http://localhost:3000. The app serves `public/index.html` and exposes `POST /api/query` and `GET /api/health`.

### Deploy to Vercel

1. Push the repo and import the project in [Vercel](https://vercel.com).
2. Set **Environment Variables** (Settings → Environment Variables):
   - `OPENAI_API_KEY` (required)
   - Optional: `OPENAI_BASE_URL`, `EMBEDDING_MODEL`, `CHAT_MODEL` (or `LLM_MODEL`), `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `RETRIEVAL_TOP_K`, `INDEX_PATH`, `INDEX_URL`
3. If the index is in the repo: ensure `index/node_index.json` is committed (or under 50MB). If it’s too large, upload to Vercel Blob (or another URL) and set `INDEX_URL` to that URL.
4. Deploy. The frontend is served from `public/`; API routes live under `/api/query` and `/api/health`.

**If you see "No index available" on Vercel:** Set `INDEX_URL` so the app loads the index from a URL instead of the filesystem. With the index in this repo, use the GitHub raw URL (replace `main` with your branch if different):

- `INDEX_URL` = `https://raw.githubusercontent.com/SonnyTsai999/JLR-Bot/main/index/node_index.json`

Then redeploy (Deployments → ⋮ → Redeploy).

### Config (Node / Vercel)

All config is via environment variables (no YAML in Node):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. API key for embeddings and chat. |
| `OPENAI_BASE_URL` | (OpenAI) | Base URL for OpenAI-compatible API. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |
| `CHAT_MODEL` / `LLM_MODEL` | `gpt-4o` | Chat model for synthesis. |
| `LLM_TEMPERATURE` | `0.3` | Chat temperature. |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per response. |
| `RETRIEVAL_TOP_K` | `10` | Number of chunks to retrieve. |
| `INDEX_PATH` | — | Path to `node_index.json` (relative to project root). |
| `INDEX_URL` | — | URL to fetch index JSON (overrides local file; use for large indexes). |

## Future integration

Design is modular to allow later addition of TLC curve data, MCDM/VIKOR scores, practitioner sentiment, and barrier intensity scoring.
