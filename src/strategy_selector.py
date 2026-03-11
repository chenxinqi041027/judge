#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy selection with optional COMET commonsense support.
- Loads strategies from a JSON/JSONL KB under /data2/xqchen/Judge/data by default.
- Optionally calls COMET to expand the reason into commonsense cues for scoring.
- Uses a simple weighted scoring (emotion match + trigger overlap + commonsense overlap) and returns top-k candidates.

Integration points:
- Fill `model_path` in build_comet_client_or_none() (see comet_client.py) to enable COMET.
- Extend the KB in /data2/xqchen/Judge/data/strategy_kb_example.jsonl or provide your own path.
- Use attach_strategy(...) to enrich a single sample (with pred_emotion + reason) before写出最终 JSON。
"""

import argparse
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    # when used as a package
    from .comet_client import build_comet_client_or_none, DEFAULT_RELATIONS, CometClient
except Exception:
    # when executed as a script (python ./src/strategy_selector.py)
    from comet_client import build_comet_client_or_none, DEFAULT_RELATIONS, CometClient

# 路径默认值
DEFAULT_KB_PATH = Path("/data2/xqchen/Judge/data/strategy_kb_example.jsonl")
DEFAULT_INPUT_JSONL = Path("/data2/xqchen/Qwen3_test/output/emotion_prediction_20260310_220307.jsonl")
DEFAULT_OUTPUT_DIR = Path("/data2/xqchen/Judge/output")
DEFAULT_LOG_DIR = Path("/data2/xqchen/Judge/log")

# 评分权重（可根据 Todo 中的建议做加权合并）
WEIGHT_EMOTION_MATCH = 1.0  # high weight
WEIGHT_TRIGGER_OVERLAP = 0.6  # medium
WEIGHT_TFIDF_SIM = 0.6  # medium
WEIGHT_COMET_OVERLAP = 0.4  # optional commonsense
EXPLICIT_BONUS = 0.1  # 当 explicitness=explicit 且有触发重叠时给予轻微加成

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

# ESConv 策略类别（小写）
ESCONV_STRATEGIES = {
    "questions",
    "self-disclosure",
    "affirmation and reassurance",
    "providing suggestions",
    "other",
    "reflection of feelings",
    "information",
    "restatement or paraphrasing",
}


def canonical_strategy_label(label: str) -> str:
    if not label:
        return "other"
    t = label.strip().lower()
    mapping = {
        "question": "questions",
        "questions": "questions",
        "ask": "questions",
        "self-disclosure": "self-disclosure",
        "self disclosure": "self-disclosure",
        "self_disclosure": "self-disclosure",
        "affirmation": "affirmation and reassurance",
        "affirmation and reassurance": "affirmation and reassurance",
        "reassurance": "affirmation and reassurance",
        "providing suggestions": "providing suggestions",
        "suggestion": "providing suggestions",
        "suggestions": "providing suggestions",
        "reflection of feelings": "reflection of feelings",
        "reflect feelings": "reflection of feelings",
        "reflection": "reflection of feelings",
        "information": "information",
        "inform": "information",
        "restatement or paraphrasing": "restatement or paraphrasing",
        "paraphrasing": "restatement or paraphrasing",
        "restatement": "restatement or paraphrasing",
        "other": "other",
    }
    if t in mapping:
        return mapping[t]
    # 尝试关键字
    if "question" in t:
        return "questions"
    if "disclos" in t:
        return "self-disclosure"
    if "affirm" in t or "reassur" in t:
        return "affirmation and reassurance"
    if "suggest" in t or "advice" in t:
        return "providing suggestions"
    if "reflect" in t or "feeling" in t:
        return "reflection of feelings"
    if "info" in t or "fact" in t:
        return "information"
    if "restat" in t or "paraphr" in t:
        return "restatement or paraphrasing"
    return "other"


def load_kb(path: str = "") -> List[Dict[str, Any]]:
    target = Path(path) if path else DEFAULT_KB_PATH
    if not target.exists():
        return []
    if target.suffix.lower() == ".jsonl":
        records = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        with target.open("r", encoding="utf-8") as f:
            records = json.load(f)

    # 标准化 KB 条目：确保有 category 字段
    normalized = []
    for item in records:
        entry = dict(item)
        entry["category"] = canonical_strategy_label(entry.get("category", ""))
        normalized.append(entry)
    return normalized


def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _build_tfidf(kb: List[Dict[str, Any]]):
    """可选：为 KB 构建 TF-IDF 特征，若 sklearn 不可用则返回 (None, None)。"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None, None, None

    corpus = []
    for item in kb:
        parts = []
        for trg in item.get("triggers", []):
            parts.append(str(trg))
        for st in item.get("strategies", []):
            parts.append(str(st))
        corpus.append(" ".join(parts))

    if not corpus:
        return None, None, None

    vectorizer = TfidfVectorizer(stop_words="english")
    kb_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, kb_matrix, cosine_similarity


def _tfidf_score(vectorizer, kb_matrix, cosine_similarity_fn, query: str, idx: int) -> float:
    if vectorizer is None or kb_matrix is None or cosine_similarity_fn is None:
        return 0.0
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity_fn(q_vec, kb_matrix)
    if sims.size == 0:
        return 0.0
    return float(sims[0, idx])


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _score_item(
    item: Dict[str, Any],
    emotion: str,
    reason: str,
    explicitness: str = "implicit",
    comet_facts: List[Dict[str, Any]] = None,
    tfidf_tuple=None,
    item_idx: int = 0,
) -> Tuple[float, Dict[str, float]]:
    comet_facts = comet_facts or []
    score_components: Dict[str, float] = {}

    vectorizer, kb_matrix, cosine_similarity_fn = tfidf_tuple if tfidf_tuple else (None, None, None)

    # Emotion exact match
    emo_score = WEIGHT_EMOTION_MATCH if emotion and item.get("emotion") == emotion else 0.0
    score_components["emotion"] = emo_score

    # Trigger overlap with reason
    reason_tokens = _tokenize(reason)
    trigger_tokens = set()
    for trg in item.get("triggers", []):
        trigger_tokens |= _tokenize(trg)
    trig_overlap = _jaccard(reason_tokens, trigger_tokens)
    trigger_score = WEIGHT_TRIGGER_OVERLAP * trig_overlap
    score_components["trigger"] = trigger_score

    # TF-IDF 语义相似度（可选）
    tfidf_sim = _tfidf_score(vectorizer, kb_matrix, cosine_similarity_fn, reason, item_idx)
    score_components["tfidf"] = WEIGHT_TFIDF_SIM * tfidf_sim

    # COMET overlap: compare generated tails with trigger tokens
    comet_tokens = set()
    for fact in comet_facts:
        comet_tokens |= _tokenize(fact.get("tail", ""))
    comet_overlap = _jaccard(comet_tokens, trigger_tokens)
    score_components["comet"] = WEIGHT_COMET_OVERLAP * comet_overlap

    # 显性触发加成
    if explicitness == "explicit" and trig_overlap > 0:
        score_components["explicit_bonus"] = EXPLICIT_BONUS
    else:
        score_components["explicit_bonus"] = 0.0

    total = sum(score_components.values())
    return total, score_components


def select_strategy(
    emotion: str,
    reason: str,
    explicitness: str,
    kb: List[Dict[str, Any]],
    comet: CometClient = None,
    relations: List[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Returns a dict with:
    - candidates: sorted list of strategy entries with scores
    - commonsense: COMET facts (if any)
    """
    relations = relations or COMET_RELATIONS
    commonsense: List[Dict[str, Any]] = []

    if comet is not None and comet.is_available():
        commonsense = comet.generate(reason, relations, num_return_sequences=1)

    # 可选 TF-IDF
    tfidf_tuple = _build_tfidf(kb)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for idx, item in enumerate(kb):
        total, comps = _score_item(
            item,
            emotion,
            reason,
            explicitness=explicitness,
            comet_facts=commonsense,
            tfidf_tuple=tfidf_tuple,
            item_idx=idx,
        )
        enriched = dict(item)
        enriched["score_components"] = comps
        enriched["score"] = round(total, 3)
        enriched["category"] = canonical_strategy_label(item.get("category", ""))
        scored.append((total, enriched))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [entry for _, entry in scored[:top_k]]

    return {
        "candidates": candidates,
        "commonsense": commonsense,
    }


def attach_strategy(
    sample: Dict[str, Any],
    kb: List[Dict[str, Any]],
    comet: CometClient = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Attach strategy candidates to a sample dict containing pred_emotion, reason, explicitness."""
    emotion = sample.get("pred_emotion", "")
    reason = sample.get("reason", "")
    explicitness = sample.get("explicitness", "implicit")

    result = select_strategy(emotion, reason, explicitness, kb, comet=comet, top_k=top_k)

    enriched = dict(sample)
    enriched["strategy_candidates"] = result["candidates"]
    if result["candidates"]:
        top = result["candidates"][0]
        enriched["strategy"] = [top.get("category", "other")]
        enriched["strategy_texts"] = top.get("strategies", [])
    else:
        enriched["strategy"] = []
    if result["commonsense"]:
        enriched["commonsense"] = result["commonsense"]
    # confidence 取第一候选 score，若无候选则 0
    enriched["strategy_confidence"] = result["candidates"][0].get("score", 0.0) if result["candidates"] else 0.0
    return enriched


def demo():
    kb = load_kb()
    comet_client = build_comet_client_or_none()
    sample = {
        "pred_emotion": "afraid",
        "reason": "The speaker was scared by a tire burst on a busy road",
        "explicitness": "explicit",
    }
    enriched = attach_strategy(sample, kb, comet=comet_client, top_k=3)
    print(json.dumps(enriched, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach strategies to emotion prediction outputs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL, help="输入 emotion_prediction*.jsonl 路径")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB_PATH, help="策略知识库 JSON/JSONL")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--top_k", type=int, default=3, help="返回策略候选个数")
    parser.add_argument("--disable_comet", action="store_true", help="不调用 COMET 共识推理")
    args = parser.parse_args()

    kb = load_kb(str(args.kb))
    comet_client = None if args.disable_comet else build_comet_client_or_none()
    # Ensure output and log directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = args.output_dir / f"strategy_{timestamp}.jsonl"
    logpath = DEFAULT_LOG_DIR / f"strategy_{timestamp}.log"

    # Configure logging to file (and keep default handler if desired)
    logging.basicConfig(
        filename=str(logpath),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("strategy_selector")

    # Run-level metadata
    logger.info("Run start: input=%s output=%s kb=%s disable_comet=%s", args.input, outpath, args.kb, args.disable_comet)
    logger.info("KB size=%d", len(kb))
    try:
        comet_available = bool(comet_client and comet_client.is_available())
    except Exception:
        comet_available = False
    logger.info("COMET available=%s", comet_available)
    logger.info("COMET relation -> ESConv category mapping: %s", json.dumps(COMET_REL_TO_CATEGORY, ensure_ascii=False))

    with args.input.open("r", encoding="utf-8") as fin, outpath.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # Extract minimal fields for selection
            emotion = sample.get("pred_emotion", "")
            reason = sample.get("reason", "")
            explicitness = sample.get("explicitness", "implicit")

            # Run selection (this returns commonsense + scored candidates)
            result = select_strategy(emotion, reason, explicitness, kb, comet=comet_client, top_k=args.top_k)

            # Build enriched record (same format as attach_strategy)
            enriched = dict(sample)
            enriched["strategy_candidates"] = result["candidates"]
            if result["candidates"]:
                top = result["candidates"][0]
                enriched["strategy"] = [top.get("category", "other")]
                enriched["strategy_texts"] = top.get("strategies", [])
            else:
                enriched["strategy"] = []
            if result.get("commonsense"):
                enriched["commonsense"] = result["commonsense"]
            enriched["strategy_confidence"] = result["candidates"][0].get("score", 0.0) if result["candidates"] else 0.0

            # Write enriched sample to output
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            # Logging per-sample: id (if present), emotion, explicitness, selected strategy + confidence
            sample_id = sample.get("id", sample.get("dialog_id", f"idx_{i}"))
            logger.info(
                "Sample %s idx=%d pred_emotion=%s explicitness=%s selected=%s confidence=%.3f",
                sample_id,
                i,
                emotion,
                explicitness,
                enriched.get("strategy"),
                enriched.get("strategy_confidence", 0.0),
            )

            # Log each candidate's score components and strategies
            for rank, cand in enumerate(result["candidates"], start=1):
                logger.info(
                    "Sample %s Candidate %d category=%s score=%s components=%s strategies=%s",
                    sample_id,
                    rank,
                    cand.get("category"),
                    cand.get("score"),
                    json.dumps(cand.get("score_components", {}), ensure_ascii=False),
                    cand.get("strategies"),
                )

            # Log commonsense facts if present
            if result.get("commonsense"):
                logger.info("Sample %s commonsense=%s", sample_id, json.dumps(result.get("commonsense"), ensure_ascii=False))

    print(f"Saved strategies to {outpath}")
    logger.info("Saved strategies to %s", outpath)

    # 如果用户请求评估生成文件与原始对话中的策略标签，计算 P/R/F1
    # 使用原始样本中 dialog 的 annotation.strategy 字段作为 gold 标签
    def extract_gold_strategies(sample: Dict[str, Any]) -> List[str]:
        gold = []
        for turn in sample.get("dialog", []):
            ann = turn.get("annotation", {})
            s = ann.get("strategy")
            if s:
                gold.append(canonical_strategy_label(str(s)))
        # 去重
        return list(set(gold))

    def evaluate_generated_file(gen_path: Path):
        tp_sum = 0
        pred_sum = 0
        gold_sum = 0
        with gen_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gold_labels = set(extract_gold_strategies(rec))
                pred_list = rec.get("strategy")
                preds = set()
                if isinstance(pred_list, list):
                    for p in pred_list:
                        preds.add(canonical_strategy_label(str(p)))
                elif pred_list:
                    preds.add(canonical_strategy_label(str(pred_list)))
                else:
                    preds = set()

                tp = len(preds & gold_labels)
                tp_sum += tp
                pred_sum += len(preds)
                gold_sum += len(gold_labels)

        precision = tp_sum / pred_sum if pred_sum > 0 else 0.0
        recall = tp_sum / gold_sum if gold_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp_sum, "predictions": pred_sum, "gold": gold_sum}

    # 自动运行评估并输出结果文件
    eval_result = evaluate_generated_file(outpath)
    print("Evaluation:", json.dumps(eval_result, ensure_ascii=False))
    try:
        logger.info("Evaluation: %s", json.dumps(eval_result, ensure_ascii=False))
    except Exception:
        # logger may not exist if script used differently; ignore safely
        pass
