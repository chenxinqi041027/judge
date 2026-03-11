"""Fixed strategy set (ESConv-aligned) used when no KB is available."""
from typing import List, Dict

# Each strategy has a category and a short description used for cross-encoder ranking.
STRATEGY_CANDIDATES: List[Dict[str, str]] = [
    {
        "category": "questions",
        "description": "Ask clarifying questions to understand the user's situation, needs, or feelings.",
    },
    {
        "category": "self-disclosure",
        "description": "Share brief, relevant personal experiences to build rapport and reduce feelings of isolation.",
    },
    {
        "category": "affirmation and reassurance",
        "description": "Provide emotional comfort, validate feelings, and offer reassurance.",
    },
    {
        "category": "providing suggestions",
        "description": "Offer practical, actionable suggestions or next steps tailored to the user's situation.",
    },
    {
        "category": "other",
        "description": "Neutral or uncategorized responses used as a fallback.",
    },
    {
        "category": "reflection of feelings",
        "description": "Reflect or mirror the user's emotions to convey empathy and understanding.",
    },
    {
        "category": "information",
        "description": "Provide factual information or explanations to help the user understand the situation.",
    },
    {
        "category": "restatement or paraphrasing",
        "description": "Paraphrase or restate the user's message to confirm mutual understanding.",
    },
]


def get_strategy_candidates() -> List[Dict[str, str]]:
    return STRATEGY_CANDIDATES
