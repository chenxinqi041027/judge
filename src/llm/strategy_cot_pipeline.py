"""LLM-based strategy selection pipeline with COMET and HEAL evidence.

Flow: input JSON/JSONL -> COMET facts + HEAL retrieval -> prompt local Qwen ->
structured strategy scoring -> output top candidate and trace.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from ..comet_client import build_comet_client_or_none
    from ..nokb_framework.comet_features import generate_commonsense, flatten_commonsense_text
    from ..nokb_framework.config import DEFAULT_DEVICE, DEFAULT_HEAL_DIR, DEFAULT_INPUT_JSONL
    from ..nokb_framework.heal_retriever import HealRetriever
    from ..nokb_framework.strategies import get_strategy_candidates
except Exception:
    from comet_client import build_comet_client_or_none
    from nokb_framework.comet_features import generate_commonsense, flatten_commonsense_text
    from nokb_framework.config import DEFAULT_DEVICE, DEFAULT_HEAL_DIR, DEFAULT_INPUT_JSONL
    from nokb_framework.heal_retriever import HealRetriever
    from nokb_framework.strategies import get_strategy_candidates

try:
    from .qwen_client import QwenClient, extract_json_block
except Exception:
    from qwen_client import QwenClient, extract_json_block


DEFAULT_OUTPUT_DIR = Path("/data2/xqchen/Judge/output/llm")
DEFAULT_LOG_DIR = Path("/data2/xqchen/Judge/log/llm")
DEFAULT_QWEN_PATH = Path("/data2/xqchen/models/Qwen3-8B")


def build_logger(logpath: Path) -> logging.Logger:
    logger = logging.getLogger("llm_strategy_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(logpath, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def canonical_strategy_label(label: str) -> str:
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


def build_context_text(sample: Dict[str, Any]) -> str:
    dialog = sample.get("dialog") or []
    parts: List[str] = []
    for turn in dialog:
        speaker = turn.get("speaker", "user")
        txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
        if txt:
            parts.append(f"{speaker}: {txt}")
    if not parts:
        parts.append(sample.get("utterance") or sample.get("text") or sample.get("reason") or sample.get("situation") or "")
    return "\n".join(parts)


def extract_current_utterance(sample: Dict[str, Any]) -> str:
    dialog = sample.get("dialog") or []
    for turn in reversed(dialog):
        if turn.get("speaker") == "seeker":
            txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
            if txt:
                return txt
    return sample.get("utterance") or sample.get("text") or sample.get("reason") or sample.get("situation") or ""


def extract_supporter_response(sample: Dict[str, Any]) -> str:
    dialog = sample.get("dialog") or []
    for turn in reversed(dialog):
        if turn.get("speaker") == "supporter":
            txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
            if txt:
                return txt
    return ""


def build_expanded_query(utterance: str, commonsense_facts: List[Dict[str, Any]], reason: str = "", emotion: str = "") -> str:
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


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        rows = []
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def infer_mode(sample: Dict[str, Any], task_mode: str) -> str:
    if task_mode in {"classify", "select"}:
        return task_mode
    return "classify" if extract_supporter_response(sample) else "select"


def compute_comet_priors(commonsense_facts: List[Dict[str, Any]]) -> Dict[str, float]:
    relation_to_category = {
        "xNeed": "providing suggestions",
        "xIntent": "providing suggestions",
        "xWant": "providing suggestions",
        "oWant": "questions",
        "xReact": "affirmation and reassurance",
        "oReact": "reflection of feelings",
        "xEffect": "information",
        "oEffect": "information",
    }
    priors: Dict[str, float] = {}
    total = max(len(commonsense_facts), 1)
    for fact in commonsense_facts:
        category = relation_to_category.get(str(fact.get("relation", "")))
        if category:
            priors[category] = priors.get(category, 0.0) + 1.0 / total
    return priors


def response_style_prior(response_text: str, heal: HealRetriever, candidates: List[Dict[str, str]]) -> Dict[str, float]:
    priors = {item["category"]: 0.0 for item in candidates}
    if not response_text:
        return priors
    category = classify_supporter_response(response_text, heal)
    if category in priors:
        priors[category] = 1.0
    return priors


def classify_supporter_response(response_text: str, heal: HealRetriever | None = None) -> str:
    text = (response_text or "").strip().lower()
    if not text:
        return "other"

    if text.endswith("?") or re.search(r"\b(what|how|when|where|which|who|could you|would you|can you|do you)\b", text):
        return "questions"
    if re.search(r"\b(i once|i remember|when i went through|i went through|i also found|i have been there|before i|i felt the same)\b", text):
        return "self-disclosure"
    if re.search(r"\b(i'm sorry|i am sorry|that sounds hard|that sounds really hard|makes sense|your feelings make sense|you are not alone|i am here|i'm here|glad you shared|thank you for saying that)\b", text):
        return "affirmation and reassurance"
    if re.search(r"\b(try|consider|maybe|perhaps|it may help|it might help|could you aim|you could|one small step|start with)\b", text):
        return "providing suggestions"
    if re.search(r"\b(it sounds like|you sound|you seem|must feel|sounds like this has been|you look)\b", text):
        return "reflection of feelings"
    if re.search(r"\b(can show up as|it is common|stress can|can affect|often|typically|this can happen|depression can|anxiety can)\b", text):
        return "information"
    if re.search(r"\b(you are saying|what i hear|what i'm hearing|so what i'm hearing|in other words|you mean that)\b", text):
        return "restatement or paraphrasing"
    if heal is not None:
        return heal.map_response_to_strategy(response_text)
    return "other"


def build_prompt(
    sample: Dict[str, Any],
    mode: str,
    context_text: str,
    utterance: str,
    supporter_response: str,
    emotion: str,
    reason: str,
    commonsense_facts: List[Dict[str, Any]],
    heal_result: Dict[str, Any],
    comet_priors: Dict[str, float],
    response_priors: Dict[str, float],
    candidates: List[Dict[str, str]],
) -> str:
    stressor_lines = [
        f"- {item.get('label', '')} (score={item.get('score', 0.0)})"
        for item in heal_result.get("top_stressors", [])[:3]
    ]
    expectation_lines = [
        f"- {item.get('label', '')} (score={item.get('score', 0.0)})"
        for item in heal_result.get("top_expectations", [])[:3]
    ]
    response_lines = [
        f"- [{item.get('strategy', 'other')}] {item.get('text', '')} (score={item.get('score', 0.0)})"
        for item in heal_result.get("top_responses", [])[:5]
    ]
    candidate_lines = [
        "- {category}: {description} | heal_prior={heal:.3f} | comet_prior={comet:.3f} | response_style_prior={style:.3f}".format(
            category=item["category"],
            description=item["description"],
            heal=float(heal_result.get("strategy_priors", {}).get(item["category"], 0.0)),
            comet=float(comet_priors.get(item["category"], 0.0)),
            style=float(response_priors.get(item["category"], 0.0)),
        )
        for item in candidates
    ]
    comet_lines = [
        f"- {fact.get('relation', '')}: {fact.get('tail', '')}"
        for fact in commonsense_facts[:8]
        if fact.get("tail")
    ]

    if mode == "classify":
        task_instruction = (
            "Task: identify which single emotional support strategy best matches the supporter response.\n\n"
            "Think step by step before deciding. The actual wording of the supporter response matters most.\n"
            "Use dialogue context, emotion, reason, COMET cues, and HEAL evidence only as auxiliary evidence.\n"
            "Compare all candidate strategies one by one, and explain why the final one is best and why close alternatives lose.\n"
        )
        target_block = f"Supporter response to classify:\n{supporter_response or 'N/A'}\n\n"
        schema = (
            "Output naturally. You do not need to follow JSON.\n"
            "But your last line must be exactly in this form:\n"
            "Final strategy: <one candidate category>\n"
            "If possible, include short score lines such as '<candidate>: 4.5/5 - reason'.\n\n"
        )
    else:
        task_instruction = (
            "Task: choose the single best emotional support strategy for the seeker.\n\n"
            "Think step by step before deciding.\n"
            "Use dialogue, emotion, reason, COMET commonsense cues, and HEAL retrieval evidence to compare all candidate strategies.\n"
            "Explain the strengths and weaknesses of the main candidates, then choose one final strategy.\n"
        )
        target_block = ""
        schema = (
            "Output naturally. You do not need to follow JSON.\n"
            "But your last line must be exactly in this form:\n"
            "Final strategy: <one candidate category>\n"
            "If possible, include short score lines such as '<candidate>: 4.5/5 - reason'.\n\n"
        )

    return "".join([
        task_instruction,
        "Do not choose self-disclosure unless there is strong, explicit evidence in the response or context.\n",
        "Do not invent evidence outside the provided context.\n\n",
        schema,
        f"Dialogue context:\n{context_text}\n\n",
        f"Current seeker utterance:\n{utterance}\n\n",
        target_block,
        f"Emotion:\n{emotion or 'unknown'}\n\n",
        f"Reason:\n{reason or 'unknown'}\n\n",
        "COMET facts:\n",
        "\n".join(comet_lines) + "\n\n",
        "HEAL top stressors:\n",
        "\n".join(stressor_lines) + "\n\n",
        "HEAL top expectations:\n",
        "\n".join(expectation_lines) + "\n\n",
        "HEAL top responses:\n",
        "\n".join(response_lines) + "\n\n",
        "Candidate strategies:\n",
        "\n".join(candidate_lines) + "\n",
    ])


def normalize_result(result: Dict[str, Any], mode: str, candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    allowed = {item["category"] for item in candidates}
    selected = canonical_strategy_label(str(result.get("selected_strategy", "other")))
    if selected not in allowed:
        raise ValueError(f"Invalid selected strategy: {selected}")

    rows = result.get("candidate_scores") or []
    normalized_rows = []
    for row in rows:
        category = canonical_strategy_label(str(row.get("category", "other")))
        if category not in allowed:
            continue
        normalized_rows.append(
            {
                "category": category,
                "response_style_match": float(row.get("response_style_match", 0.0)) if mode == "classify" else 0.0,
                "empathy_fit": float(row.get("empathy_fit", 0.0)),
                "actionability": float(row.get("actionability", 0.0)),
                "evidence_support": float(row.get("evidence_support", 0.0)),
                "safety": float(row.get("safety", 0.0)),
                "final_score": float(row.get("final_score", 0.0)),
                "reason": str(row.get("reason", "")),
            }
        )
    normalized_rows.sort(key=lambda item: item["final_score"], reverse=True)
    return {
        "selected_strategy": selected,
        "brief_rationale": str(result.get("brief_rationale", "")),
        "candidate_scores": normalized_rows,
    }


def parse_strategy_from_freeform(text: str, mode: str, candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    candidate_names = [item["category"] for item in candidates]
    candidate_pattern = "|".join(re.escape(name) for name in sorted(candidate_names, key=len, reverse=True))

    final_match = re.search(rf"final\s+strategy\s*[:：]\s*({candidate_pattern})", text, flags=re.IGNORECASE)
    selected = canonical_strategy_label(final_match.group(1)) if final_match else ""

    if not selected:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-12:]):
            match = re.search(candidate_pattern, line, flags=re.IGNORECASE)
            if match:
                selected = canonical_strategy_label(match.group(0))
                break

    if not selected:
        raise ValueError("Failed to parse final strategy from freeform output")

    score_rows = []
    score_regex = re.compile(
        rf"^\s*({candidate_pattern})\s*[:：-]\s*([0-5](?:\.\d+)?)\s*(?:/\s*5)?\s*(?:[-:：]\s*(.*))?$",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        match = score_regex.search(line.strip())
        if not match:
            continue
        category = canonical_strategy_label(match.group(1))
        score = float(match.group(2))
        reason = (match.group(3) or "").strip()
        score_rows.append(
            {
                "category": category,
                "response_style_match": score if mode == "classify" else 0.0,
                "empathy_fit": score if mode != "classify" else 0.0,
                "actionability": 0.0,
                "evidence_support": 0.0,
                "safety": 0.0,
                "final_score": score,
                "reason": reason,
            }
        )

    rationale = ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if re.search(r"final\s+strategy\s*[:：]", line, flags=re.IGNORECASE):
            if idx > 0:
                rationale = lines[idx - 1]
            break

    return normalize_result(
        {
            "selected_strategy": selected,
            "brief_rationale": rationale,
            "candidate_scores": score_rows,
        },
        mode,
        candidates,
    )


def blend_candidate_scores(
    llm_result: Dict[str, Any],
    heal_result: Dict[str, Any],
    comet_priors: Dict[str, float],
    response_priors: Dict[str, float],
    candidates: List[Dict[str, str]],
) -> Dict[str, Any]:
    llm_rows = {row["category"]: dict(row) for row in llm_result.get("candidate_scores", [])}
    blended_rows = []
    for candidate in candidates:
        category = candidate["category"]
        row = llm_rows.get(
            category,
            {
                "category": category,
                "response_style_match": 0.0,
                "empathy_fit": 0.0,
                "actionability": 0.0,
                "evidence_support": 0.0,
                "safety": 0.0,
                "final_score": 0.0,
                "reason": "",
            },
        )
        heal_prior = float(heal_result.get("strategy_priors", {}).get(category, 0.0))
        comet_prior = float(comet_priors.get(category, 0.0))
        response_prior = float(response_priors.get(category, 0.0))
        blended_score = float(row.get("final_score", 0.0)) + 0.6 * heal_prior + 0.2 * comet_prior + 1.2 * response_prior
        merged = dict(row)
        merged["heal_prior"] = heal_prior
        merged["comet_prior"] = comet_prior
        merged["response_style_prior"] = response_prior
        merged["blended_score"] = blended_score
        blended_rows.append(merged)
    blended_rows.sort(key=lambda item: item["blended_score"], reverse=True)
    selected = blended_rows[0]["category"] if blended_rows else llm_result.get("selected_strategy", "other")
    return {
        "selected_strategy": selected,
        "brief_rationale": llm_result.get("brief_rationale", ""),
        "candidate_scores": blended_rows,
    }


def fallback_result_from_priors(
    candidates: List[Dict[str, str]],
    heal_result: Dict[str, Any],
    comet_priors: Dict[str, float],
    response_priors: Dict[str, float],
    reason: str,
) -> Dict[str, Any]:
    rows = []
    for candidate in candidates:
        category = candidate["category"]
        heal_prior = float(heal_result.get("strategy_priors", {}).get(category, 0.0))
        comet_prior = float(comet_priors.get(category, 0.0))
        response_prior = float(response_priors.get(category, 0.0))
        blended_score = 0.6 * heal_prior + 0.2 * comet_prior + 1.8 * response_prior
        rows.append(
            {
                "category": category,
                "response_style_match": 5.0 if response_prior > 0 else 0.0,
                "empathy_fit": 0.0,
                "actionability": 0.0,
                "evidence_support": 5.0 * heal_prior,
                "safety": 4.0,
                "final_score": blended_score,
                "heal_prior": heal_prior,
                "comet_prior": comet_prior,
                "response_style_prior": response_prior,
                "blended_score": blended_score,
                "reason": reason,
            }
        )
    rows.sort(key=lambda item: item["blended_score"], reverse=True)
    return {
        "selected_strategy": rows[0]["category"] if rows else "other",
        "brief_rationale": reason,
        "candidate_scores": rows,
    }


def compute_metrics(gold_labels: List[str], pred_labels: List[str], labels: List[str]) -> Dict[str, Any]:
    total = len(gold_labels)
    correct = sum(int(g == p) for g, p in zip(gold_labels, pred_labels))
    accuracy = correct / total if total else 0.0

    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in zip(gold_labels, pred_labels):
        if gold in confusion and pred in confusion[gold]:
            confusion[gold][pred] += 1

    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_support = 0
    tp_sum = fp_sum = fn_sum = 0

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        total_support += support
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    label_count = max(len(labels), 1)
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision / label_count,
        "macro_recall": macro_recall / label_count,
        "macro_f1": macro_f1 / label_count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_precision": weighted_precision / total_support if total_support else 0.0,
        "weighted_recall": weighted_recall / total_support if total_support else 0.0,
        "weighted_f1": weighted_f1 / total_support if total_support else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[confusion[gold][pred] for pred in labels] for gold in labels],
        },
    }


def parse_with_retry(qwen: QwenClient, system_prompt: str, prompt: str, mode: str, candidates: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str]:
    raw_output = qwen.generate(system_prompt, prompt)
    try:
        parsed = extract_json_block(raw_output)
        return normalize_result(parsed, mode, candidates), raw_output
    except Exception:
        try:
            return parse_strategy_from_freeform(raw_output, mode, candidates), raw_output
        except Exception:
            retry_prompt = prompt + "\n\nPlease answer again. Think step by step if needed, but make the last line exactly 'Final strategy: <candidate>'."
            retry_output = qwen.generate(system_prompt, retry_prompt)
            try:
                parsed = extract_json_block(retry_output)
                return normalize_result(parsed, mode, candidates), retry_output
            except Exception:
                try:
                    return parse_strategy_from_freeform(retry_output, mode, candidates), retry_output
                except Exception as exc:
                    raise ValueError(f"Failed to parse model output. first_output={raw_output!r} retry_output={retry_output!r}") from exc


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    log_dir: Path,
    qwen_path: Path,
    heal_dir: Path = DEFAULT_HEAL_DIR,
    device: str = DEFAULT_DEVICE,
    max_new_tokens: int = 1024,
    task_mode: str = "auto",
    limit: int = 0,
    enable_thinking: bool = True,
    rule_first_classify: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = output_dir / f"llm_strategy_{timestamp}.jsonl"
    logpath = log_dir / f"llm_strategy_{timestamp}.log"
    metricspath = output_dir / f"llm_strategy_metrics_{timestamp}.json"

    logger = build_logger(logpath)

    run_config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_path": str(outpath),
        "log_path": str(logpath),
        "metrics_path": str(metricspath),
        "qwen_path": str(qwen_path),
        "heal_dir": str(heal_dir),
        "device": device,
        "max_new_tokens": max_new_tokens,
        "task_mode": task_mode,
        "limit": limit,
        "enable_thinking": enable_thinking,
        "rule_first_classify": rule_first_classify,
    }
    logger.info("run_config=%s", json.dumps(run_config, ensure_ascii=False, indent=2))

    logger.info(
        "run_start input=%s output=%s qwen_path=%s heal_dir=%s device=%s max_new_tokens=%d",
        input_path,
        outpath,
        qwen_path,
        heal_dir,
        device,
        max_new_tokens,
    )

    comet = build_comet_client_or_none()
    heal = HealRetriever(heal_dir)
    qwen = QwenClient(model_path=qwen_path, max_new_tokens=max_new_tokens, temperature=0.0, enable_thinking=enable_thinking)
    candidates = get_strategy_candidates()
    samples = load_samples(input_path)
    if limit > 0:
        samples = samples[:limit]

    logger.info("data_summary=%s", json.dumps({
        "samples": len(samples),
        "strategy_distribution": dict(Counter(canonical_strategy_label(str(item.get("strategy", "other"))) for item in samples)),
        "comet_available": bool(comet and comet.is_available()),
        "heal_available": heal.is_available(),
    }, ensure_ascii=False, indent=2))

    system_prompt = (
        "You are an expert emotional support strategy analyst. "
        "Use deliberate reasoning internally, but output only the required JSON."
    )

    gold_labels: List[str] = []
    pred_labels: List[str] = []

    with outpath.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", sample.get("sample_id", sample.get("dialog_id", f"idx_{idx}")))
            emotion = sample.get("pred_emotion") or sample.get("gold_emotion") or sample.get("emotion_type") or ""
            reason = sample.get("reason") or ""
            utterance = extract_current_utterance(sample)
            supporter_response = extract_supporter_response(sample)
            context_text = build_context_text(sample)
            mode = infer_mode(sample, task_mode)

            commonsense_facts = generate_commonsense(utterance, comet, relations=None, num_return_sequences=1)
            commonsense_text = flatten_commonsense_text(commonsense_facts)
            expanded_query = build_expanded_query(utterance, commonsense_facts, reason=reason, emotion=emotion)
            heal_result = heal.retrieve(expanded_query, emotion=emotion)
            comet_priors = compute_comet_priors(commonsense_facts)
            response_priors = response_style_prior(supporter_response, heal, candidates)

            logger.info("sample=%s comet_facts=%s", sample_id, json.dumps(commonsense_facts, ensure_ascii=False))
            logger.info("sample=%s heal_retrieval=%s", sample_id, json.dumps({
                "top_stressors": heal_result.get("top_stressors", []),
                "top_expectations": heal_result.get("top_expectations", []),
                "top_responses": heal_result.get("top_responses", []),
                "strategy_priors": heal_result.get("strategy_priors", {}),
            }, ensure_ascii=False))

            prompt = build_prompt(
                sample=sample,
                mode=mode,
                context_text=context_text,
                utterance=utterance,
                supporter_response=supporter_response,
                emotion=emotion,
                reason=reason,
                commonsense_facts=commonsense_facts,
                heal_result=heal_result,
                comet_priors=comet_priors,
                response_priors=response_priors,
                candidates=candidates,
            )
            logger.info("sample=%s prompt=%s", sample_id, prompt)

            if mode == "classify" and rule_first_classify and max(response_priors.values(), default=0.0) >= 1.0:
                raw_output = "SKIPPED_LLM_RULE_PRIOR"
                llm_result = fallback_result_from_priors(
                    candidates=candidates,
                    heal_result=heal_result,
                    comet_priors=comet_priors,
                    response_priors=response_priors,
                    reason="Rule-first classify fallback from supporter response style.",
                )
            else:
                try:
                    llm_raw_result, raw_output = parse_with_retry(qwen, system_prompt, prompt, mode, candidates)
                    llm_result = blend_candidate_scores(llm_raw_result, heal_result, comet_priors, response_priors, candidates)
                except Exception as exc:
                    raw_output = f"PARSE_FAILED: {exc}"
                    logger.warning("sample=%s parse_failed=%s raw_output=%s", sample_id, exc, raw_output)
                    llm_result = fallback_result_from_priors(
                        candidates=candidates,
                        heal_result=heal_result,
                        comet_priors=comet_priors,
                        response_priors=response_priors,
                        reason=f"Fallback from priors after parse failure: {exc}",
                    )
            logger.info("sample=%s raw_llm_output=%s", sample_id, raw_output)

            enriched = dict(sample)
            enriched["expanded_query"] = expanded_query
            enriched["commonsense"] = commonsense_facts
            enriched["commonsense_text"] = commonsense_text
            enriched["heal_retrieval"] = {
                "top_stressors": heal_result.get("top_stressors", []),
                "top_expectations": heal_result.get("top_expectations", []),
                "top_responses": heal_result.get("top_responses", []),
                "strategy_priors": heal_result.get("strategy_priors", {}),
            }
            enriched["task_mode"] = mode
            enriched["llm_strategy_result"] = llm_result
            enriched["strategy_candidates"] = llm_result.get("candidate_scores", [])[:3]
            enriched["strategy"] = [llm_result.get("selected_strategy", "other")]
            enriched["strategy_confidence"] = (
                llm_result.get("candidate_scores", [{}])[0].get("blended_score", 0.0)
                if llm_result.get("candidate_scores") else 0.0
            )
            enriched["decision_trace"] = {
                "emotion": emotion,
                "reason": reason,
                "utterance": utterance,
                "supporter_response": supporter_response,
                "context_text": context_text,
                "comet_priors": comet_priors,
                "response_style_priors": response_priors,
                "raw_llm_output": raw_output,
            }
            gold = canonical_strategy_label(str(sample.get("strategy", "other")))
            pred = canonical_strategy_label(str(llm_result.get("selected_strategy", "other")))
            gold_labels.append(gold)
            pred_labels.append(pred)
            enriched["gold_strategy"] = gold

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            logger.info(
                "sample=%s mode=%s gold=%s pred=%s confidence=%.4f",
                sample_id,
                mode,
                gold,
                enriched["strategy"],
                enriched["strategy_confidence"],
            )
            logger.info("sample=%s top_candidates=%s", sample_id, json.dumps(enriched["strategy_candidates"], ensure_ascii=False))

    labels = [item["category"] for item in candidates]
    metrics = compute_metrics(gold_labels, pred_labels, labels)
    metrics_payload = {
        "run_config": run_config,
        "metrics": metrics,
    }
    metricspath.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("final_metrics=%s", json.dumps(metrics, ensure_ascii=False, indent=2))
    logger.info("artifacts=%s", json.dumps({"output_path": str(outpath), "log_path": str(logpath), "metrics_path": str(metricspath)}, ensure_ascii=False))

    print(f"Saved outputs to {outpath}")
    print(f"Log written to {logpath}")
    print(f"Metrics written to {metricspath}")
    return {"output_path": str(outpath), "log_path": str(logpath), "metrics_path": str(metricspath), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM strategy selection with COMET and HEAL evidence")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT_JSONL), help="Input JSON or JSONL")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="Log directory")
    parser.add_argument("--heal_dir", type=Path, default=DEFAULT_HEAL_DIR, help="HEAL root directory")
    parser.add_argument("--qwen_path", type=Path, default=DEFAULT_QWEN_PATH, help="Local Qwen model path")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Reserved for compatibility")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum generation length")
    parser.add_argument("--task_mode", choices=["auto", "classify", "select"], default="auto", help="Inference mode")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for quick tests")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable model thinking mode")
    parser.add_argument("--rule_first_classify", action="store_true", help="Enable rule-first shortcut in classify mode")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        qwen_path=args.qwen_path,
        heal_dir=args.heal_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        task_mode=args.task_mode,
        limit=args.limit,
        enable_thinking=not args.disable_thinking,
        rule_first_classify=args.rule_first_classify,
    )