"""
Central configuration for the multihop BrowseComp-style dataset pipeline.
Set API keys via environment variables (recommended) or edit the defaults below.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# --- LLM (DeepSeek, OpenAI-compatible API) ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_CHEAP = os.environ.get("DEEPSEEK_MODEL_CHEAP", "deepseek-v4-flash")   # was "deepseek-chat"
DEEPSEEK_MODEL_STRONG = os.environ.get("DEEPSEEK_MODEL_STRONG", "deepseek-v4-pro")   # was "deepseek-reasoner"

# --- Secondary model for Steps 5 & 6a (redundancy check + blind-solve check) ---
# FREE, no billing: runs locally instead of calling a paid API, so these two
# steps - which specifically need a model from a DIFFERENT family than
# DeepSeek to avoid shared blind spots - cost nothing beyond your own GPU.
#
# Default: Qwen2.5-14B-Instruct, 4-bit quantized (~9GB VRAM, fits a 12GB card).
# Ungated on Hugging Face (no license request needed), Apache 2.0 license.
#
# Alternative if you prefer Meta's family or want a smaller footprint:
#   "meta-llama/Llama-3.1-8B-Instruct"  (~5-6GB in 4-bit)
#   Note: Llama repos are gated - you must accept Meta's license on the
#   model's Hugging Face page and run `huggingface-cli login` once before
#   first download. Still free, just an extra one-time step.
LOCAL_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_MODEL_4BIT = True                 # keep True to fit 12GB VRAM
LOCAL_MODEL_MAX_NEW_TOKENS = 1024

# --- Search (Tavily) ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# --- Pipeline tunables ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

RECENCY_START_DATE = os.environ.get("RECENCY_START_DATE", "2026-01-01")

MAX_RELATION_ATTEMPTS_PER_HOP = 6   # step 2: retry different relation types before dead-end
MAX_BACKTRACKS_PER_CHAIN = 6        # step 2: try alternate B candidates
HOP_COUNT = int(os.environ.get("HOP_COUNT", "2"))   # chain length: A -> B -> C -> D -> E (4 hops = 5 entities)
SEARCH_RESULTS_PER_QUERY = 10

# --- Diversity controls (Step 1 seed picking, Step 2 hop exploration) ---
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.2"))
GENERATION_TOP_P = float(os.environ.get("GENERATION_TOP_P", "0.3"))
DIVERSITY_HINT_SAMPLE_SIZE = int(os.environ.get("DIVERSITY_HINT_SAMPLE_SIZE", "8"))
MAX_DEDUP_RETRIES = int(os.environ.get("MAX_DEDUP_RETRIES", "4"))

# Domains treated as low-credibility for Step 3 filtering
LOW_CREDIBILITY_DOMAINS = [
    "pinterest.", "quora.", "reddit.com/r/", "answers.com",
    "blogspot.", "medium.com/@",  # personal/unverified posts
]
