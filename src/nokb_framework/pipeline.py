"""End-to-end COMET-only strategy selection pipeline (no KB dependency).

Flow: input JSONL -> COMET facts -> fused text -> cross-encoder ranking -> output top-k strategies.
Each input line should contain at least one of: `utterance`, `text`, or `reason`.
Optional: `dialog` (list of turns with `speaker` and `text`) to build richer context.
"""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import sys
# Ensure parent 'src' folder is on sys.path so imports like `comet_client` resolve when running as script
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from .config import (
        DEFAULT_INPUT_JSONL,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_LOG_DIR,
        COMET_RELATIONS,
        COMET_REL_TO_CATEGORY,
        CONTEXT_ENCODER_NAME,
        COMMONSENSE_ENCODER_NAME,
        CROSS_ENCODER_NAME,
        DEFAULT_DEVICE,
    )
    from .strategies import get_strategy_candidates
    from .comet_features import generate_commonsense, flatten_commonsense_text
    from .encoder import TextEncoder
    from .fuse import fuse_concat, l2_normalize
    from .ranker import StrategyRanker
except Exception:
    # fallback when running pipeline.py as a script from the nokb_framework dir
    from config import (
        DEFAULT_INPUT_JSONL,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_LOG_DIR,
        COMET_RELATIONS,
        COMET_REL_TO_CATEGORY,
        CONTEXT_ENCODER_NAME,
        COMMONSENSE_ENCODER_NAME,
        CROSS_ENCODER_NAME,
        DEFAULT_DEVICE,
    )
    from strategies import get_strategy_candidates
    from comet_features import generate_commonsense, flatten_commonsense_text
    from encoder import TextEncoder
    from fuse import fuse_concat, l2_normalize
    from ranker import StrategyRanker

try:
    from ..comet_client import build_comet_client_or_none
except Exception:
    try:
        from src.comet_client import build_comet_client_or_none
    except Exception:
        # last resort: top-level comet_client
        from comet_client import build_comet_client_or_none  # type: ignore


def _build_context_text(sample: Dict[str, any]) -> str:
    # Build a context string using dialog history if available
    dialog = sample.get("dialog") or []
    parts: List[str] = []
    for turn in dialog:
        speaker = turn.get("speaker", "user")
        txt = turn.get("text") or turn.get("utterance") or ""
        if txt:
            parts.append(f"{speaker}: {txt}")
    # Fallback to current utterance if no dialog
    if not parts:
        parts.append(sample.get("utterance") or sample.get("text") or sample.get("reason") or "")
    return " \n ".join(parts)


def _combine_context_commonsense(context_text: str, commonsense_text: str) -> str:
    if commonsense_text:
        return f"{context_text} [SEP] commonsense: {commonsense_text}"
    return context_text


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    log_dir: Path,
    top_k: int = 3,
    device: str = DEFAULT_DEVICE,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = output_dir / f"nokb_strategy_{timestamp}.jsonl"
    logpath = log_dir / f"nokb_strategy_{timestamp}.log"

    logging.basicConfig(
        filename=str(logpath),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("nokb_pipeline")

    # Build models
    comet = build_comet_client_or_none()
    ctx_encoder = TextEncoder(CONTEXT_ENCODER_NAME, device=device)
    cs_encoder = TextEncoder(COMMONSENSE_ENCODER_NAME, device=device)
    ranker = StrategyRanker(CROSS_ENCODER_NAME, device=device)

    candidates = get_strategy_candidates()
    strategy_texts = [c["description"] for c in candidates]

    logger.info(
        "Run start input=%s output=%s top_k=%d device=%s comet_available=%s", input_path, outpath, top_k, device, bool(comet and comet.is_available())
    )
    logger.info("COMET relations=%s", COMET_RELATIONS)
    logger.info("COMET relation -> category mapping=%s", json.dumps(COMET_REL_TO_CATEGORY, ensure_ascii=False))
    # counters for evaluation (if gold annotations present in input)
    tp_sum = 0
    pred_sum = 0
    gold_sum = 0

    def _extract_gold_strategies(sample: Dict[str, any]) -> List[str]:
        gold = []
        for turn in sample.get("dialog", []):
            ann = turn.get("annotation", {})
            s = ann.get("strategy")
            if s:
                gold.append(s if isinstance(s, str) else str(s))
        return list(set([s.lower() for s in gold if s]))

    with input_path.open("r", encoding="utf-8") as fin, outpath.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = sample.get("id", sample.get("dialog_id", f"idx_{idx}"))

            context_text = _build_context_text(sample)
            commonsense_facts = generate_commonsense(context_text, comet, relations=COMET_RELATIONS, num_return_sequences=1)
            commonsense_text = flatten_commonsense_text(commonsense_facts)

            # Encode context and commonsense separately, then fuse (concat + L2)
            ctx_emb = ctx_encoder.encode([context_text])  # (1, h)
            cs_emb = cs_encoder.encode([commonsense_text]) if commonsense_text else ctx_encoder.encode([""])
            fused_emb = fuse_concat(ctx_emb, cs_emb)
            fused_emb = l2_normalize(fused_emb)

            # Build fused text for cross-encoder scoring
            fused_text_for_ce = _combine_context_commonsense(context_text, commonsense_text)
            scores = ranker.score(fused_text_for_ce, strategy_texts)

            scored = []
            for s, cand in zip(scores, candidates):
                scored.append({
                    "category": cand["category"],
                    "description": cand["description"],
                    "score": float(s),
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[:top_k]

            enriched = dict(sample)
            enriched["commonsense"] = commonsense_facts
            enriched["strategy_candidates"] = topk
            enriched["strategy"] = [topk[0]["category"]] if topk else []
            enriched["strategy_confidence"] = topk[0]["score"] if topk else 0.0

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            logger.info(
                "Sample %s idx=%d selected=%s score=%.3f", sample_id, idx, enriched.get("strategy"), enriched.get("strategy_confidence", 0.0)
            )
            logger.info("Sample %s commonsense=%s", sample_id, json.dumps(commonsense_facts, ensure_ascii=False))
            logger.info("Sample %s candidates=%s", sample_id, json.dumps(topk, ensure_ascii=False))

            # Evaluation accumulation
            gold_labels = set(_extract_gold_strategies(sample))
            pred_list = enriched.get("strategy") or []
            preds = set([p.lower() for p in pred_list if p])
            tp = len(preds & gold_labels)
            tp_sum += tp
            pred_sum += len(preds)
            gold_sum += len(gold_labels)

    logger.info("Saved outputs to %s", outpath)
    print(f"Saved outputs to {outpath}")
    print(f"Log written to {logpath}")
    # Compute evaluation metrics if we saw any gold labels
    if gold_sum > 0:
        precision = tp_sum / pred_sum if pred_sum > 0 else 0.0
        recall = tp_sum / gold_sum if gold_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        eval_res = {"precision": precision, "recall": recall, "f1": f1, "tp": tp_sum, "predictions": pred_sum, "gold": gold_sum}
        logger.info("Evaluation: %s", json.dumps(eval_res, ensure_ascii=False))
        print("Evaluation:", json.dumps(eval_res, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMET-only strategy selection (no KB)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL, help="Input JSONL with utterances/dialog")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write outputs")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Where to write logs")
    parser.add_argument("--top_k", type=int, default=3, help="Number of strategy candidates to keep")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run encoders/cross-encoder")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, args.log_dir, top_k=args.top_k, device=args.device)
