"""Config for COMET-only strategy selection framework (no KB dependency)."""
from pathlib import Path

# Default paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = Path("/data2/xqchen/Judge/data/demo.json")
DEFAULT_OUTPUT_DIR = Path("/data2/xqchen/Judge/output")
DEFAULT_LOG_DIR = Path("/data2/xqchen/Judge/log")
DEFAULT_HEAL_DIR = Path("/data2/xqchen/Judge/model/HEAL")
DEFAULT_MODEL_DIR = Path("/data2/xqchen/Judge/model/nokb_framework")
DEFAULT_RANKER_PATH = DEFAULT_MODEL_DIR / "strategy_ranker.pkl"
DEFAULT_RESPONSE_MAPPER_PATH = DEFAULT_MODEL_DIR / "response_strategy_mapper.pkl"

# COMET relations to request
COMET_RELATIONS = [
    "xNeed",
    "xIntent",
    "xWant",
    "xReact",
    "xEffect",
    "oReact",
    "oEffect",
    "oWant",
]

# Mapping COMET relations to ESConv-style categories (used when no KB is present)
COMET_REL_TO_CATEGORY = {
    "xNeed": "providing suggestions",
    "xIntent": "providing suggestions",
    "xWant": "providing suggestions",
    "oWant": "questions",
    "xReact": "affirmation and reassurance",
    "oReact": "reflection of feelings",
    "xEffect": "information",
    "oEffect": "information",
}

# Model names
# Small encoders keep resource usage low; swap to sentence-transformers if preferred.
CONTEXT_ENCODER_NAME = "bert-base-chinese"
COMMONSENSE_ENCODER_NAME = "bert-base-chinese"
CROSS_ENCODER_NAME = "bert-base-chinese"

# Device selection (torch will pick CPU if CUDA not available)
DEFAULT_DEVICE = "cuda"  # will fall back to CPU if not available
