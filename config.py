"""
Central configuration for the multihop BrowseComp-style dataset pipeline.
Set API keys via environment variables (recommended) or edit the defaults below.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

LOCAL_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen3-8B")
LOCAL_MODEL_4BIT = True                 # keep True to fit 12GB VRAM
LOCAL_MODEL_MAX_NEW_TOKENS = 1024

# --- Search (Tavily) ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# --- Pipeline tunables ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

START_DATE = os.environ.get("RECENCY_START_DATE", "2026-01-01")
END_DATE = os.environ.get("RECENCY_END_DATE", "2026-12-01")

MAX_RELATION_ATTEMPTS_PER_HOP = 8   # step 2: retry different relation types before dead-end
MAX_BACKTRACKS_PER_CHAIN = 12        # step 2: try alternate B candidates
HOP_COUNT = int(os.environ.get("HOP_COUNT", "3"))   # number of hops: 3 means A -> B -> C -> D
SEARCH_RESULTS_PER_QUERY = 10

# --- Diversity controls (Step 1 seed picking, Step 2 hop exploration) ---
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.2"))
GENERATION_TOP_P = float(os.environ.get("GENERATION_TOP_P", "0.3"))
DIVERSITY_HINT_SAMPLE_SIZE = int(os.environ.get("DIVERSITY_HINT_SAMPLE_SIZE", "8"))
MAX_DEDUP_RETRIES = int(os.environ.get("MAX_DEDUP_RETRIES", "4"))
MAX_CONSTRUCTION_RETRIES = int(os.environ.get("MAX_CONSTRUCTION_RETRIES", "5"))

# Domains treated as low-credibility for Step 3 filtering
LOW_CREDIBILITY_DOMAINS = [
    "pinterest.", "quora.", "reddit.com/r/", "answers.com",
    "blogspot.", "medium.com/@",  # personal/unverified posts
]