"""Train the lightweight strategy ranker and HEAL response->strategy mapper on demo.json.

This uses weak supervision from the demo labels:
- ranker: binary pair classification over (features, strategy_description)
- response mapper: labels retrieved HEAL responses and supporter utterances with the sample gold strategy
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple


DEFAULT_INPUT_PATH = Path("/data2/xqchen/Judge/data/demo.json")
DEFAULT_SEED = 42

DEFAULT_RANKER_MODEL_TYPE = "mlp"
DEFAULT_RANKER_NB_ALPHA = 1.0 
DEFAULT_RANKER_ENCODER_BACKEND = "sbert"
DEFAULT_RANKER_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RANKER_FEATURE_DIM = 2048
DEFAULT_RANKER_HIDDEN_DIM = 256
DEFAULT_RANKER_DROPOUT = 0.2 
DEFAULT_RANKER_EPOCHS = 30
DEFAULT_RANKER_BATCH_SIZE = 16
DEFAULT_RANKER_LEARNING_RATE = 1e-3
DEFAULT_RANKER_POSITIVE_CLASS_WEIGHT = 0.0

DEFAULT_MAPPER_MODEL_TYPE = "mlp"
DEFAULT_MAPPER_NB_ALPHA = 1.0
DEFAULT_MAPPER_ENCODER_BACKEND = "sbert"
DEFAULT_MAPPER_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAPPER_FEATURE_DIM = 2048
DEFAULT_MAPPER_HIDDEN_DIM = 256
DEFAULT_MAPPER_DROPOUT = 0.2
DEFAULT_MAPPER_EPOCHS = 30
DEFAULT_MAPPER_BATCH_SIZE = 16
DEFAULT_MAPPER_LEARNING_RATE = 1e-3

try:
    from .config import (
        DEFAULT_HEAL_DIR,
        DEFAULT_INPUT_JSONL,
        DEFAULT_MODEL_DIR,
        DEFAULT_RANKER_PATH,
        DEFAULT_RESPONSE_MAPPER_PATH,
    )
    from .heal_retriever import HealRetriever
    from .ranker import StrategyRanker
    from .response_strategy_mapper import ResponseStrategyMapper
    from .strategies import get_strategy_candidates
except Exception:
    from config import (
        DEFAULT_HEAL_DIR,
        DEFAULT_INPUT_JSONL,
        DEFAULT_MODEL_DIR,
        DEFAULT_RANKER_PATH,
        DEFAULT_RESPONSE_MAPPER_PATH,
    )
    from heal_retriever import HealRetriever
    from ranker import StrategyRanker
    from response_strategy_mapper import ResponseStrategyMapper
    from strategies import get_strategy_candidates


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
    return " [SEP] ".join(parts)


def extract_current_utterance(sample: Dict[str, Any]) -> str:
    dialog = sample.get("dialog") or []
    for turn in reversed(dialog):
        if turn.get("speaker") == "seeker":
            return turn.get("content") or turn.get("text") or turn.get("utterance") or ""
    return sample.get("utterance") or sample.get("text") or sample.get("reason") or sample.get("situation") or ""


def build_expanded_query(sample: Dict[str, Any]) -> str:
    utterance = extract_current_utterance(sample)
    emotion = sample.get("pred_emotion") or sample.get("gold_emotion") or sample.get("emotion_type") or ""
    reason = sample.get("reason") or ""
    parts = []
    if utterance:
        parts.append(f"utterance: {utterance}")
    if emotion:
        parts.append(f"emotion: {emotion}")
    if reason:
        parts.append(f"reason: {reason}")
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


def prepare_training_data(samples: List[Dict[str, Any]], heal: HealRetriever) -> Tuple[List[str], List[str], List[int], List[str], List[str]]:
    candidates = get_strategy_candidates()
    pair_contexts: List[str] = []
    pair_strategy_texts: List[str] = []
    pair_labels: List[int] = []
    mapper_texts: List[str] = []
    mapper_labels: List[str] = []

    for sample in samples:
        gold = canonical_strategy_label(sample.get("strategy") if not isinstance(sample.get("strategy"), list) else (sample.get("strategy") or ["other"])[0])
        context = build_context_text(sample)
        utterance = extract_current_utterance(sample)
        emotion = sample.get("gold_emotion") or sample.get("pred_emotion") or sample.get("emotion_type") or ""
        reason = sample.get("reason") or ""
        expanded_query = build_expanded_query(sample)
        heal_result = heal.retrieve(expanded_query, emotion=emotion)
        heal_text = heal_result.get("heal_text", "")

        feature_text = " [SEP] ".join(x for x in [
            context,
            f"current_utterance: {utterance}" if utterance else "",
            f"emotion: {emotion}" if emotion else "",
            f"reason: {reason}" if reason else "",
            f"heal: {heal_text}" if heal_text else "",
        ] if x)

        for cand in candidates:
            pair_contexts.append(feature_text)
            pair_strategy_texts.append(cand["description"])
            pair_labels.append(1 if cand["category"] == gold else 0)

        # weak labels for mapper: support utterances + retrieved HEAL responses
        for turn in sample.get("dialog", []):
            if turn.get("speaker") == "supporter":
                txt = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
                if txt:
                    mapper_texts.append(txt)
                    mapper_labels.append(gold)
        for resp in heal_result.get("top_responses", [])[:5]:
            txt = resp.get("text") or ""
            if txt:
                mapper_texts.append(txt)
                mapper_labels.append(gold)

    return pair_contexts, pair_strategy_texts, pair_labels, mapper_texts, mapper_labels


def evaluate_ranker(samples: List[Dict[str, Any]], heal: HealRetriever, ranker: StrategyRanker) -> Dict[str, Any]:
    candidates = get_strategy_candidates()
    correct = 0
    total = 0
    for sample in samples:
        gold = canonical_strategy_label(sample.get("strategy") if not isinstance(sample.get("strategy"), list) else (sample.get("strategy") or ["other"])[0])
        context = build_context_text(sample)
        utterance = extract_current_utterance(sample)
        emotion = sample.get("gold_emotion") or sample.get("pred_emotion") or sample.get("emotion_type") or ""
        reason = sample.get("reason") or ""
        heal_result = heal.retrieve(build_expanded_query(sample), emotion=emotion)
        heal_text = heal_result.get("heal_text", "")
        feature_text = " [SEP] ".join(x for x in [context, f"current_utterance: {utterance}", f"emotion: {emotion}", f"reason: {reason}", f"heal: {heal_text}"] if x)
        scores = []
        for cand in candidates:
            s = ranker.score(feature_text, [cand["description"]])[0]
            scores.append((s, cand["category"]))
        pred = sorted(scores, key=lambda x: x[0], reverse=True)[0][1]
        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def summarize_binary_scores(scores: List[float], labels: List[int], threshold: float) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for score, gold in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_mapper(texts: List[str], labels: List[str], mapper: ResponseStrategyMapper) -> Dict[str, Any]:
    if not texts:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    correct = 0
    for txt, gold in zip(texts, labels):
        pred = mapper.predict(txt)
        correct += int(pred == gold)
    total = len(texts)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


class TrainLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.log_path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def build_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "input": str(args.input),
        "heal_dir": str(args.heal_dir),
        "model_dir": str(args.model_dir),
        "ranker_path": str(args.ranker_path),
        "response_mapper_path": str(args.response_mapper_path),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "early_stop_patience": args.early_stop_patience,
        "ranker": {
            "model_type": args.ranker_model_type,
            "nb_alpha": args.ranker_nb_alpha,
            "encoder_backend": args.ranker_encoder_backend,
            "encoder_name": args.ranker_encoder_name,
            "feature_dim": args.ranker_feature_dim,
            "hidden_dim": args.ranker_hidden_dim,
            "dropout": args.ranker_dropout,
            "epochs": args.ranker_epochs,
            "batch_size": args.ranker_batch_size,
            "learning_rate": args.ranker_learning_rate,
            "positive_class_weight": args.ranker_positive_class_weight,
        },
        "mapper": {
            "model_type": args.mapper_model_type,
            "nb_alpha": args.mapper_nb_alpha,
            "encoder_backend": args.mapper_encoder_backend,
            "encoder_name": args.mapper_encoder_name,
            "feature_dim": args.mapper_feature_dim,
            "hidden_dim": args.mapper_hidden_dim,
            "dropout": args.mapper_dropout,
            "epochs": args.mapper_epochs,
            "batch_size": args.mapper_batch_size,
            "learning_rate": args.mapper_learning_rate,
        },
    }


def format_epoch_message(component: str, metrics: Dict[str, float]) -> str:
    epoch = int(metrics.get("epoch", 0))
    total_epochs = int(metrics.get("total_epochs", 0))
    parts = [f"[{component}] epoch {epoch}/{total_epochs}"]
    if "train_loss" in metrics:
        parts.append(f"train_loss={metrics['train_loss']:.4f}")
    if "val_accuracy" in metrics:
        parts.append(f"val_acc={metrics['val_accuracy']:.4f}")
    if "val_precision" in metrics:
        parts.append(f"val_precision={metrics['val_precision']:.4f}")
    if "val_recall" in metrics:
        parts.append(f"val_recall={metrics['val_recall']:.4f}")
    if "val_f1" in metrics:
        parts.append(f"val_f1={metrics['val_f1']:.4f}")
    if "val_threshold" in metrics:
        parts.append(f"threshold={metrics['val_threshold']:.4f}")
    if metrics.get("best_epoch"):
        parts.append("best_so_far")
    if metrics.get("stopped_early"):
        parts.append("early_stop")
    if metrics.get("used_nb"):
        parts.append("nb_backend")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Train ranker and response mapper on demo data")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Training samples in JSON or JSONL")
    parser.add_argument("--heal_dir", type=Path, default=DEFAULT_HEAL_DIR, help="HEAL graph root")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory to save trained models")
    parser.add_argument("--ranker_path", type=Path, default=DEFAULT_RANKER_PATH, help="Path to save ranker model")
    parser.add_argument("--response_mapper_path", type=Path, default=DEFAULT_RESPONSE_MAPPER_PATH, help="Path to save response mapper model")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for trainable models")
    parser.add_argument("--ranker_model_type", choices=["nb", "mlp"], default=DEFAULT_RANKER_MODEL_TYPE, help="Ranker backend")
    parser.add_argument("--ranker_nb_alpha", type=float, default=DEFAULT_RANKER_NB_ALPHA, help="Naive Bayes smoothing for ranker")
    parser.add_argument("--ranker_encoder_backend", choices=["auto", "hash", "sbert"], default=DEFAULT_RANKER_ENCODER_BACKEND, help="Text encoder backend for ranker MLP")
    parser.add_argument("--ranker_encoder_name", type=str, default=DEFAULT_RANKER_ENCODER_NAME, help="Sentence encoder name for ranker MLP")
    parser.add_argument("--ranker_feature_dim", type=int, default=DEFAULT_RANKER_FEATURE_DIM, help="Hashed feature dimension for ranker")
    parser.add_argument("--ranker_hidden_dim", type=int, default=DEFAULT_RANKER_HIDDEN_DIM, help="Hidden layer size for ranker MLP")
    parser.add_argument("--ranker_dropout", type=float, default=DEFAULT_RANKER_DROPOUT, help="Dropout for ranker MLP")
    parser.add_argument("--ranker_epochs", type=int, default=DEFAULT_RANKER_EPOCHS, help="Training epochs for ranker")
    parser.add_argument("--ranker_batch_size", type=int, default=DEFAULT_RANKER_BATCH_SIZE, help="Batch size for ranker")
    parser.add_argument("--ranker_learning_rate", type=float, default=DEFAULT_RANKER_LEARNING_RATE, help="Learning rate for ranker")
    parser.add_argument("--ranker_positive_class_weight", type=float, default=DEFAULT_RANKER_POSITIVE_CLASS_WEIGHT, help="Positive class weight for ranker BCE loss")
    parser.add_argument("--mapper_model_type", choices=["nb", "mlp"], default=DEFAULT_MAPPER_MODEL_TYPE, help="Response mapper backend")
    parser.add_argument("--mapper_nb_alpha", type=float, default=DEFAULT_MAPPER_NB_ALPHA, help="Naive Bayes smoothing for mapper")
    parser.add_argument("--mapper_encoder_backend", choices=["auto", "hash", "sbert"], default=DEFAULT_MAPPER_ENCODER_BACKEND, help="Text encoder backend for mapper MLP")
    parser.add_argument("--mapper_encoder_name", type=str, default=DEFAULT_MAPPER_ENCODER_NAME, help="Sentence encoder name for mapper MLP")
    parser.add_argument("--mapper_feature_dim", type=int, default=DEFAULT_MAPPER_FEATURE_DIM, help="Hashed feature dimension for mapper")
    parser.add_argument("--mapper_hidden_dim", type=int, default=DEFAULT_MAPPER_HIDDEN_DIM, help="Hidden layer size for mapper MLP")
    parser.add_argument("--mapper_dropout", type=float, default=DEFAULT_MAPPER_DROPOUT, help="Dropout for mapper MLP")
    parser.add_argument("--mapper_epochs", type=int, default=DEFAULT_MAPPER_EPOCHS, help="Training epochs for mapper")
    parser.add_argument("--mapper_batch_size", type=int, default=DEFAULT_MAPPER_BATCH_SIZE, help="Batch size for mapper")
    parser.add_argument("--mapper_learning_rate", type=float, default=DEFAULT_MAPPER_LEARNING_RATE, help="Learning rate for mapper")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of data to use as validation set")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience (epochs) on validation")
    parser.add_argument("--log_path", type=Path, default=None, help="Optional path to save training log")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_path = args.model_dir / f"train_demo_{run_timestamp}.log"
    log_path = args.log_path or default_log_path
    logger = TrainLogger(log_path)

    training_config = build_training_config(args)
    logger.log(f"[train_demo] started_at={run_timestamp}")
    logger.log("[train_demo] core_config=")
    logger.log(json.dumps(training_config, ensure_ascii=False, indent=2))

    try:
        samples = load_samples(args.input)
        heal = HealRetriever(args.heal_dir)
        pair_contexts, pair_strategy_texts, pair_labels, mapper_texts, mapper_labels = prepare_training_data(samples, heal)
        data_summary = {
            "samples": len(samples),
            "pair_examples": len(pair_labels),
            "pair_label_distribution": dict(Counter(pair_labels)),
            "mapper_examples": len(mapper_labels),
            "mapper_label_distribution": dict(Counter(mapper_labels)),
        }
        logger.log("[train_demo] data_summary=")
        logger.log(json.dumps(data_summary, ensure_ascii=False, indent=2))

        # optionally split train/val
        val_frac = max(0.0, min(1.0, float(args.val_frac)))
        if val_frac > 0.0:
            import random as _random

            _random.seed(args.seed)
            n = len(pair_labels)
            idxs = list(range(n))
            _random.shuffle(idxs)
            cut = int(n * (1 - val_frac))
            train_idx = set(idxs[:cut])
            tc, ts, tl = [], [], []
            vc, vs, vl = [], [], []
            for i in range(n):
                if i in train_idx:
                    tc.append(pair_contexts[i]); ts.append(pair_strategy_texts[i]); tl.append(pair_labels[i])
                else:
                    vc.append(pair_contexts[i]); vs.append(pair_strategy_texts[i]); vl.append(pair_labels[i])
            pair_contexts_train, pair_strategy_texts_train, pair_labels_train = tc, ts, tl
            pair_contexts_val, pair_strategy_texts_val, pair_labels_val = vc, vs, vl
        else:
            pair_contexts_train, pair_strategy_texts_train, pair_labels_train = pair_contexts, pair_strategy_texts, pair_labels
            pair_contexts_val, pair_strategy_texts_val, pair_labels_val = None, None, None

        ranker = StrategyRanker(
            model_type=args.ranker_model_type,
            nb_alpha=args.ranker_nb_alpha,
            encoder_backend=args.ranker_encoder_backend,
            encoder_name=args.ranker_encoder_name,
            feature_dim=args.ranker_feature_dim,
            hidden_dim=args.ranker_hidden_dim,
            dropout=args.ranker_dropout,
            epochs=args.ranker_epochs,
            batch_size=args.ranker_batch_size,
            learning_rate=args.ranker_learning_rate,
            positive_class_weight=args.ranker_positive_class_weight,
            seed=args.seed,
        )
        ranker._progress_callback = lambda metrics: logger.log(format_epoch_message("ranker", metrics))
        logger.log("[train_demo] ranker training started")
        if pair_contexts_val is not None:
            ranker._val_texts = [ranker._build_pair_text(c, s) for c, s in zip(pair_contexts_val, pair_strategy_texts_val)]
            ranker._val_labels = pair_labels_val
            ranker._early_stop_patience = args.early_stop_patience
            ranker.fit(pair_contexts_train, pair_strategy_texts_train, pair_labels_train)
            delattr(ranker, "_val_texts")
            delattr(ranker, "_val_labels")
            delattr(ranker, "_early_stop_patience")
        else:
            ranker.fit(pair_contexts_train, pair_strategy_texts_train, pair_labels_train)
        delattr(ranker, "_progress_callback")
        ranker.save(args.ranker_path)
        logger.log(f"[train_demo] ranker saved to {args.ranker_path}")

        mapper = ResponseStrategyMapper(
            model_type=args.mapper_model_type,
            nb_alpha=args.mapper_nb_alpha,
            encoder_backend=args.mapper_encoder_backend,
            encoder_name=args.mapper_encoder_name,
            feature_dim=args.mapper_feature_dim,
            hidden_dim=args.mapper_hidden_dim,
            dropout=args.mapper_dropout,
            epochs=args.mapper_epochs,
            batch_size=args.mapper_batch_size,
            learning_rate=args.mapper_learning_rate,
            seed=args.seed,
        )
        mapper._progress_callback = lambda metrics: logger.log(format_epoch_message("mapper", metrics))
        # split mapper train/val (use same val_frac)
        if val_frac > 0.0 and mapper_texts:
            import random as _random

            _random.seed(args.seed)
            m_idx = list(range(len(mapper_texts)))
            _random.shuffle(m_idx)
            m_cut = int(len(mapper_texts) * (1 - val_frac))
            m_train_idx = set(m_idx[:m_cut])
            m_tc, m_tl = [], []
            m_vc, m_vl = [], []
            for i in range(len(mapper_texts)):
                if i in m_train_idx:
                    m_tc.append(mapper_texts[i]); m_tl.append(mapper_labels[i])
                else:
                    m_vc.append(mapper_texts[i]); m_vl.append(mapper_labels[i])
            mapper_train_texts, mapper_train_labels = m_tc, m_tl
            mapper_val_texts, mapper_val_labels = m_vc, m_vl
        else:
            mapper_train_texts, mapper_train_labels = mapper_texts, mapper_labels
            mapper_val_texts, mapper_val_labels = None, None

        logger.log("[train_demo] mapper training started")
        if mapper_val_texts is not None:
            mapper._val_texts = mapper_val_texts
            mapper._val_labels = mapper_val_labels
            mapper._early_stop_patience = args.early_stop_patience
            mapper.fit(mapper_train_texts, mapper_train_labels)
            delattr(mapper, "_val_texts")
            delattr(mapper, "_val_labels")
            delattr(mapper, "_early_stop_patience")
        else:
            mapper.fit(mapper_train_texts, mapper_train_labels)
        delattr(mapper, "_progress_callback")
        mapper.save(args.response_mapper_path)
        logger.log(f"[train_demo] mapper saved to {args.response_mapper_path}")

        # reload mapper into HEAL for evaluation
        heal_eval = HealRetriever(args.heal_dir, response_mapper=mapper)
        ranker_metrics = evaluate_ranker(samples, heal_eval, ranker)
        mapper_metrics = evaluate_mapper(mapper_texts, mapper_labels, mapper)

        # If validation was used, compute confusion matrices on val sets
        def confusion_matrix(y_true, y_pred, labels=None):
            lab = sorted(list(set(labels or (list(set(y_true)) + list(set(y_pred))))))
            idx = {l: i for i, l in enumerate(lab)}
            mat = [[0 for _ in lab] for _ in lab]
            for t, p in zip(y_true, y_pred):
                mat[idx[t]][idx[p]] += 1
            return lab, mat

        val_info = {}
        if pair_contexts_val is not None:
            # evaluate ranker on val set
            val_scores = []
            for c, s in zip(pair_contexts_val, pair_strategy_texts_val):
                sc = ranker.score(c, [s])[0]
                val_scores.append(sc)
            threshold = float(getattr(ranker, "decision_threshold", 0.0))
            preds = [1 if sc >= threshold else 0 for sc in val_scores]
            lab, mat = confusion_matrix(pair_labels_val, preds, labels=[0, 1])
            val_info['ranker_confusion'] = {'labels': lab, 'matrix': mat}
            val_info['ranker_positive_metrics'] = summarize_binary_scores(val_scores, pair_labels_val, threshold)
        if mapper_val_texts is not None:
            m_preds = [mapper.predict(t) for t in mapper_val_texts]
            lab_m, mat_m = confusion_matrix(mapper_val_labels, m_preds)
            val_info['mapper_confusion'] = {'labels': lab_m, 'matrix': mat_m}

        result = {
            "samples": len(samples),
            "pair_examples": len(pair_labels),
            "pair_label_distribution": dict(Counter(pair_labels)),
            "mapper_examples": len(mapper_labels),
            "mapper_label_distribution": dict(Counter(mapper_labels)),
            "training_config": training_config,
            "log_path": str(log_path),
            "ranker_path": str(args.ranker_path),
            "response_mapper_path": str(args.response_mapper_path),
            "ranker_metrics": ranker_metrics,
            "mapper_metrics": mapper_metrics,
        }
        if isinstance(getattr(ranker, "pipeline", None), dict):
            result["ranker_training_diagnostics"] = {
                "decision_threshold": ranker.pipeline.get("decision_threshold", 0.0),
                "resolved_positive_class_weight": ranker.pipeline.get("positive_class_weight", args.ranker_positive_class_weight),
                "val_positive_metrics": ranker.pipeline.get("val_positive_metrics"),
                "encoder_backend_used": ranker.pipeline.get("encoder_backend"),
            }
        # include validation info if present
        result["val_info"] = val_info

        # print to stdout and save to log
        logger.log("[train_demo] final_result=")
        logger.log(json.dumps(result, ensure_ascii=False, indent=2))

        # also write summary to model_dir with timestamp
        try:
            args.model_dir.mkdir(parents=True, exist_ok=True)
            summary_path = args.model_dir / f"train_summary_{run_timestamp}.json"
            with summary_path.open("w", encoding="utf-8") as sf:
                json.dump(result, sf, ensure_ascii=False, indent=2)
            logger.log(f"[train_demo] summary saved to {summary_path}")
            logger.log(f"[train_demo] log saved to {log_path}")
        except Exception as exc:
            logger.log(f"[train_demo] failed to save summary: {exc}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
