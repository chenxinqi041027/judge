"""Commonsense feature generation using COMET (no KB required)."""
from typing import List, Dict, Optional

try:
    from .config import COMET_RELATIONS
except Exception:
    from config import COMET_RELATIONS

try:
    # reuse existing comet client implementation
    from ..comet_client import build_comet_client_or_none, CometClient
except Exception:  # fallback when run as script
    from comet_client import build_comet_client_or_none, CometClient  # type: ignore


def generate_commonsense(
    utterance: str,
    comet: Optional["CometClient"],
    relations: List[str] = None,
    num_return_sequences: int = 1,
) -> List[Dict[str, str]]:
    """Call COMET to get commonsense facts; returns a list of {relation, tail} dicts."""
    relations = relations or COMET_RELATIONS
    if comet is None or not comet.is_available():
        return []
    return comet.generate(utterance, relations, num_return_sequences=num_return_sequences)


def flatten_commonsense_text(facts: List[Dict[str, str]]) -> str:
    """Turn commonsense facts into a short text block for encoding."""
    parts = []
    for fact in facts:
        rel = fact.get("relation") or fact.get("rel") or "rel"
        tail = fact.get("tail") or fact.get("text") or ""
        if tail:
            parts.append(f"{rel}: {tail}")
    return " ; ".join(parts)
