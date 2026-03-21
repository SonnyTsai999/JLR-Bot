"""
Configuration loader for JLR Technology Intelligence Assistant.
Loads settings from config/settings.yaml relative to project root.
"""
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_config() -> dict[str, Any]:
    config_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        return _default_config()
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or _default_config()


def get_api_key() -> str | None:
    """API key for OpenAI-compatible services. Env OPENAI_API_KEY overrides config."""
    import os
    return os.environ.get("OPENAI_API_KEY") or (load_config().get("llm") or {}).get("api_key")


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "raw_pdfs": "data/raw_pdfs",
            "processed_chunks": "data/processed_chunks",
            "index_dir": "index",
        },
        "chunking": {
            "token_target_min": 500,
            "token_target_max": 800,
            "overlap_tokens": 100,
            "max_overlap_tokens": 150,
        },
        "embedding": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "batch_size": 1,
            "max_chunks_per_run": 500,
            "timeout_seconds": 300,
        },
        "retrieval": {
            "top_k": 10,
            "max_chunks_per_source": 3,
            "min_similarity_threshold": 0.0,
        },
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 2048,
            "base_url": None,
            "api_key": None,
        },
        "system_identity": "Technology Strategy Analyst for JLR specializing in emerging AEC design technologies.",
        "lifecycle_stages": ["Emerging", "Growth", "Mature", "Decline"],
    }
