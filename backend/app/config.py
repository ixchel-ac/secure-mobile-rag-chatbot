"""Settings and environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Search for .env in backend/ first, then project root
_config_dir = Path(__file__).parent.parent
_project_root = _config_dir.parent

load_dotenv(_project_root / ".env", override=True)
load_dotenv(_config_dir / ".env", override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama", "groq", or "wandb"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "").strip()
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mobile-rag-firewall").strip()
WANDB_BASE_URL = os.getenv("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
DATA_DIR = os.getenv("DATA_DIR", "./data/")
INDEX_DIR = os.getenv("INDEX_DIR", "./index/")
TOP_K = int(os.getenv("TOP_K", "5"))

OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "10"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))