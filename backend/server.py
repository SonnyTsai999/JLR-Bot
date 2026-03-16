"""
API server for JLR Technology Intelligence Assistant.
Exposes /query (POST) and /health (GET). No external internet retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when running server
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.load_config import load_config, get_api_key
from backend.retrieve import retrieve
from backend.synthesize import synthesize

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None  # type: ignore
    BaseModel = None  # type: ignore
    FileResponse = None  # type: ignore


def create_app() -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("fastapi and uvicorn are required. Install with: pip install fastapi uvicorn")
    app = FastAPI(
        title="JLR Technology Intelligence Assistant",
        description="RAG-based decision-support for emerging AEC design technologies. Evidence-grounded only.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class QueryRequest(BaseModel):
        query: str = Field(..., min_length=1, description="Natural language question")
        technology_L1: str | None = Field(None, description="Filter by L1 technology category")
        technology_L2: str | None = Field(None, description="Filter by L2 technology category")
        lifecycle_stage: str | None = Field(None, description="Filter by lifecycle stage")
        top_k: int | None = Field(None, description="Override default top_k")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "jlr-technology-intelligence"}

    _project_root = Path(__file__).resolve().parent.parent
    _frontend_path = _project_root / "frontend" / "index.html"

    @app.get("/")
    def serve_frontend():
        if _frontend_path.exists():
            return FileResponse(_frontend_path)
        return {"message": "Frontend not found. Use POST /query or open frontend/index.html in a browser."}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    @app.post("/query")
    async def post_query(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="Body must be a JSON object")
        query = body.get("query")
        if not query or not str(query).strip():
            raise HTTPException(status_code=422, detail="Field 'query' is required and must be a non-empty string")
        query = str(query).strip()
        technology_L1 = body.get("technology_L1")
        technology_L2 = body.get("technology_L2")
        lifecycle_stage = body.get("lifecycle_stage")
        top_k = body.get("top_k")

        config = load_config()
        api_key = get_api_key()
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not set. Set OPENAI_API_KEY or llm.api_key in config/settings.yaml.")
        try:
            chunks = retrieve(
                query,
                config=config,
                top_k=top_k,
                technology_L1=technology_L1,
                technology_L2=technology_L2,
                lifecycle_stage=lifecycle_stage,
                api_key=api_key,
            )
            answer = synthesize(query, chunks, config=config, api_key=api_key)
            # Deduplicate sources by DOI (or apa_citation) so each paper appears once
            seen = set()
            sources_used = []
            for c in chunks:
                key = (c.get("doi") or "").strip() or (c.get("apa_citation") or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    sources_used.append({
                        "source": c.get("source"),
                        "authors": c.get("authors"),
                        "title": c.get("title"),
                        "year": c.get("year"),
                        "doi": c.get("doi"),
                        "apa_citation": c.get("apa_citation"),
                        "chunk_id": c.get("chunk_id"),
                    })
            return {
                "query": query,
                "response": answer,
                "sources_used": sources_used,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
