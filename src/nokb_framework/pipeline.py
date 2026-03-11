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
from typing import Dict, List, Any

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
        DEFAULT_HEAL_DIR,
        DEFAULT_RANKER_PATH,
        DEFAULT_RESPONSE_MAPPER_PATH,
        COMET_RELATIONS,
        COMET_REL_TO_CATEGORY,
        DEFAULT_DEVICE,
    )
    from .strategies import get_strategy_candidates
    from .comet_features import generate_commonsense, flatten_commonsense_text
    from .heal_retriever import HealRetriever
    from .ranker import StrategyRanker
    from .response_strategy_mapper import ResponseStrategyMapper
except Exception:
    # fallback when running pipeline.py as a script from the nokb_framework dir
    from config import (
        DEFAULT_INPUT_JSONL,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_LOG_DIR,
        DEFAULT_HEAL_DIR,
        DEFAULT_RANKER_PATH,
        DEFAULT_RESPONSE_MAPPER_PATH,
        COMET_RELATIONS,
        COMET_REL_TO_CATEGORY,
        DEFAULT_DEVICE,
    )
    from strategies import get_strategy_candidates
    from comet_features import generate_commonsense, flatten_commonsense_text
    from heal_retriever import HealRetriever
    from ranker import StrategyRanker
    from response_strategy_mapper import ResponseStrategyMapper

try:
    from ..comet_client import build_comet_client_or_none
except Exception:
    try:
        from src.comet_client import build_comet_client_or_none
    except Exception:
        # last resort: top-level comet_client
        from comet_client import build_comet_client_or_none  # type: ignore


def _canonical_strategy_label(label: str) -> str:
    t = (label or "").strip().lower()
    mapping = {
        "question": "questions",
        "questions": "questions",
        "self-disclosure": "self-disclosure",
        "self disclosure": "self-disclosure",
        "affirmation and reassurance": "affirmation and reassurance",
        "affirmation": "affirmation and reassurance",
        "reassurance": "affirmation and reassurance",
        "providing suggestions": "providing suggestions",
        "suggestion": "providing suggestions",
        "reflection of feelings": "reflection of feelings",
        "information": "information",
        "restatement or paraphrasing": "restatement or paraphrasing",
        "restatement": "restatement or paraphrasing",
        "paraphrasing": "restatement or paraphrasing",
        "others": "other",
        "other": "other",
    }
    return mapping.get(t, t or "other")


def _build_context_text(sample: Dict[str, Any]) -> str:
    # Build a context string using dialog history if available
    dialog = sample.get("dialog") or []
    parts: List[str] = []
    for turn in dialog:
        speaker = turn.get("speaker", "user")
        txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
        if txt:
            parts.append(f"{speaker}: {txt}")
    # Fallback to current utterance if no dialog
    if not parts:
        parts.append(sample.get("utterance") or sample.get("text") or sample.get("reason") or sample.get("situation") or "")
    return " \n ".join(parts)


def _extract_current_utterance(sample: Dict[str, Any]) -> str:
    dialog = sample.get("dialog") or []
    for turn in reversed(dialog):
        if turn.get("speaker") == "seeker":
            txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
            if txt:
                return txt
    return sample.get("utterance") or sample.get("text") or sample.get("reason") or sample.get("situation") or ""


def _combine_context_commonsense(context_text: str, commonsense_text: str) -> str:
    if commonsense_text:
        return f"{context_text} [SEP] commonsense: {commonsense_text}"
    return context_text


def _build_expanded_query(utterance: str, commonsense_facts: List[Dict[str, Any]], reason: str = "", emotion: str = "") -> str:
    parts = []
    if utterance:
        parts.append(f"utterance: {utterance}")
    if emotion:
        parts.append(f"emotion: {emotion}")
    if reason:
        parts.append(f"reason: {reason}")
    for fact in commonsense_facts:
        rel = fact.get("relation", "")
        tail = fact.get("tail", "")
        if tail:
            parts.append(f"{rel}: {tail}")
    return " [SEP] ".join(parts)


def _extract_gold_strategies(sample: Dict[str, Any]) -> List[str]:
    gold = []
    for turn in sample.get("dialog", []):
        ann = turn.get("annotation", {})
        s = ann.get("strategy")
        if s:
            gold.append(_canonical_strategy_label(str(s)))
    top_level = sample.get("strategy")
    if isinstance(top_level, list):
        gold.extend(_canonical_strategy_label(str(x)) for x in top_level if x)
    elif top_level:
        gold.append(_canonical_strategy_label(str(top_level)))
    return list(set(x for x in gold if x))


def _fallback_response(category: str) -> str:
    mapping = {
        "questions": "Can you tell me a little more about what feels hardest right now?",
        "self-disclosure": "I've seen situations like this feel overwhelming, and it can help to take things one step at a time.",
        "affirmation and reassurance": "I'm sorry you're going through this, and your feelings make sense.",
        "providing suggestions": "It may help to start with one small, manageable step today.",
        "reflection of feelings": "It sounds like this situation has been really emotionally draining for you.",
        "information": "Stress like this can affect both mood and decision-making, so your reaction is understandable.",
        "restatement or paraphrasing": "So what I'm hearing is that this situation has left you feeling stuck and overwhelmed.",
        "other": "Thank you for sharing this.",
    }
    return mapping.get(category, mapping["other"])


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    log_dir: Path,
    top_k: int = 3,
    device: str = DEFAULT_DEVICE,
    heal_dir: Path = DEFAULT_HEAL_DIR,
    ranker_path: Path = DEFAULT_RANKER_PATH,
    response_mapper_path: Path = DEFAULT_RESPONSE_MAPPER_PATH,
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
    response_mapper = ResponseStrategyMapper(str(response_mapper_path))
    heal = HealRetriever(heal_dir, response_mapper=response_mapper)
    ranker = StrategyRanker(str(ranker_path))

    candidates = get_strategy_candidates()

    logger.info(
        "Run start input=%s output=%s top_k=%d device=%s comet_available=%s heal_dir=%s heal_available=%s ranker_path=%s ranker_fitted=%s response_mapper_path=%s response_mapper_fitted=%s",
        input_path,
        outpath,
        top_k,
        device,
        bool(comet and comet.is_available()),
        heal_dir,
        heal.is_available(),
        ranker_path,
        ranker.is_fitted(),
        response_mapper_path,
        response_mapper.is_fitted(),
    )
    logger.info("COMET relations=%s", COMET_RELATIONS)
    logger.info("COMET relation -> category mapping=%s", json.dumps(COMET_REL_TO_CATEGORY, ensure_ascii=False))
    logger.info("Strategy candidates=%s", json.dumps(candidates, ensure_ascii=False))
    # counters for evaluation (if gold annotations present in input)
    tp_sum = 0
    pred_sum = 0
    gold_sum = 0

    with input_path.open("r", encoding="utf-8") as fin, outpath.open("w", encoding="utf-8") as fout:
        # Support either JSONL (one JSON per line) or a single JSON array file
        if input_path.suffix.lower() == ".json":
            try:
                data = json.load(fin)
            except Exception as e:
                logger.error("Failed to parse JSON file %s: %s", input_path, e)
                return
            if isinstance(data, list):
                iterable = data
            else:
                iterable = [data]
        else:
            # JSONL
            def gen_lines():
                for l in fin:
                    l = l.strip()
                    if not l:
                        continue
                    yield json.loads(l)

            iterable = gen_lines()

        for idx, sample in enumerate(iterable):
            sample_id = sample.get("id", sample.get("sample_id", sample.get("dialog_id", f"idx_{idx}")))

            emotion = sample.get("pred_emotion") or sample.get("gold_emotion") or sample.get("emotion_type") or ""
            reason = sample.get("reason") or ""
            utterance = _extract_current_utterance(sample)

            context_text = _build_context_text(sample)
            commonsense_facts = generate_commonsense(utterance, comet, relations=COMET_RELATIONS, num_return_sequences=1)
            commonsense_text = flatten_commonsense_text(commonsense_facts)
            expanded_query = _build_expanded_query(utterance, commonsense_facts, reason=reason, emotion=emotion)
            heal_result = heal.retrieve(expanded_query, emotion=emotion)

            scored = []
            comet_category_priors = {}
            if commonsense_facts:
                total_facts = max(len(commonsense_facts), 1)
                for fact in commonsense_facts:
                    cat = COMET_REL_TO_CATEGORY.get(fact.get("relation", ""))
                    if cat:
                        comet_category_priors[cat] = comet_category_priors.get(cat, 0.0) + 1.0 / total_facts

            for cand in candidates:
                heal_strategy_text = heal.build_strategy_specific_knowledge(heal_result, cand["category"])
                fused_text_for_ce = " [SEP] ".join(
                    x for x in [
                        context_text,
                        f"current_utterance: {utterance}",
                        f"reason: {reason}" if reason else "",
                        f"emotion: {emotion}" if emotion else "",
                        f"comet: {commonsense_text}" if commonsense_text else "",
                        f"heal: {heal_strategy_text}" if heal_strategy_text else heal_result.get("heal_text", ""),
                    ] if x
                )
                model_score = float(ranker.score(fused_text_for_ce, [cand["description"]])[0])
                heal_prior = float(heal_result.get("strategy_priors", {}).get(cand["category"], 0.0))
                comet_prior = float(comet_category_priors.get(cand["category"], 0.0))
                final_score = model_score + 0.5 * heal_prior + 0.2 * comet_prior
                scored.append({
                    "category": cand["category"],
                    "description": cand["description"],
                    "score": float(final_score),
                    "score_components": {
                        "model": round(model_score, 4),
                        "heal_prior": round(heal_prior, 4),
                        "comet_prior": round(comet_prior, 4),
                    },
                    "heal_support": heal_strategy_text,
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[:top_k]
            selected_category = topk[0]["category"] if topk else "other"
            matched_responses = [r for r in heal_result.get("top_responses", []) if r.get("strategy") == selected_category]
            response_text = matched_responses[0]["text"] if matched_responses else (heal_result.get("top_responses", [{}])[0].get("text") if heal_result.get("top_responses") else _fallback_response(selected_category))

            enriched = dict(sample)
            enriched["commonsense"] = commonsense_facts
            enriched["expanded_query"] = expanded_query
            enriched["heal_retrieval"] = {
                "top_stressors": heal_result.get("top_stressors", []),
                "top_expectations": heal_result.get("top_expectations", []),
                "top_responses": heal_result.get("top_responses", []),
                "strategy_priors": heal_result.get("strategy_priors", {}),
                "subgraphs": heal_result.get("subgraphs", []),
            }
            enriched["strategy_candidates"] = topk
            enriched["strategy"] = [selected_category] if topk else []
            enriched["strategy_confidence"] = topk[0]["score"] if topk else 0.0
            enriched["response"] = response_text
            enriched["decision_trace"] = {
                "emotion": emotion,
                "reason": reason,
                "utterance": utterance,
                "context_text": context_text,
                "commonsense_text": commonsense_text,
                "heal_text": heal_result.get("heal_text", ""),
                "comet_category_priors": comet_category_priors,
            }

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            logger.info("========== Sample %s idx=%d ==========" , sample_id, idx)
            logger.info("Sample %s dialog=%s", sample_id, json.dumps(sample.get("dialog", []), ensure_ascii=False))
            logger.info("Sample %s original_tags=%s", sample_id, json.dumps({
                "emotion_type": sample.get("emotion_type"),
                "gold_emotion": sample.get("gold_emotion"),
                "pred_emotion": sample.get("pred_emotion"),
                "reason": sample.get("reason"),
                "explicitness": sample.get("explicitness"),
                "strategy": sample.get("strategy"),
            }, ensure_ascii=False))
            logger.info("Sample %s utterance=%s", sample_id, utterance)
            logger.info("Sample %s context_text=%s", sample_id, context_text)
            logger.info("Sample %s expanded_query=%s", sample_id, expanded_query)
            logger.info("Sample %s commonsense=%s", sample_id, json.dumps(commonsense_facts, ensure_ascii=False))
            logger.info("Sample %s heal_retrieval=%s", sample_id, json.dumps(enriched["heal_retrieval"], ensure_ascii=False))
            logger.info("Sample %s comet_category_priors=%s", sample_id, json.dumps(comet_category_priors, ensure_ascii=False))
            logger.info("Sample %s strategy_candidates=%s", sample_id, json.dumps(topk, ensure_ascii=False))
            logger.info("Sample %s selected_strategy=%s score=%.4f response=%s", sample_id, enriched.get("strategy"), enriched.get("strategy_confidence", 0.0), response_text)

            # Evaluation accumulation
            gold_labels = set(_extract_gold_strategies(sample))
            pred_list = enriched.get("strategy") or []
            preds = set([_canonical_strategy_label(p) for p in pred_list if p])
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
    parser = argparse.ArgumentParser(description="COMET + HEAL strategy selection (no KB)")
    parser.add_argument("--input", type=Path, default=Path("/data2/xqchen/Judge/data/demo.json"), help="Input JSON or JSONL with utterances/dialog")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write outputs")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Where to write logs")
    parser.add_argument("--heal_dir", type=Path, default=DEFAULT_HEAL_DIR, help="HEAL model/data root directory")
    parser.add_argument("--ranker_path", type=Path, default=DEFAULT_RANKER_PATH, help="Trained strategy ranker model path")
    parser.add_argument("--response_mapper_path", type=Path, default=DEFAULT_RESPONSE_MAPPER_PATH, help="Trained response-to-strategy mapper path")
    parser.add_argument("--top_k", type=int, default=3, help="Number of strategy candidates to keep")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run encoders/cross-encoder")
    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output_dir,
        args.log_dir,
        top_k=args.top_k,
        device=args.device,
        heal_dir=args.heal_dir,
        ranker_path=args.ranker_path,
        response_mapper_path=args.response_mapper_path,
    )
