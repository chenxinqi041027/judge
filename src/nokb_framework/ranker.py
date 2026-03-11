"""Trainable strategy ranker.

Supports two backends:
- `nb`: bag-of-words Naive Bayes with smoothing
- `mlp`: stable-hash or sentence-embedding features + small PyTorch MLP
"""
from __future__ import annotations

import math
import importlib
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from .text_encoder import TextEncoder
except Exception:
    from text_encoder import TextEncoder


def _load_torch_modules():
    try:
        torch_module = importlib.import_module("torch")
        nn_module = importlib.import_module("torch.nn")
        optim_module = importlib.import_module("torch.optim")
        return torch_module, nn_module, optim_module
    except Exception:
        return None, None, None


def _build_pair_mlp(input_dim: int, hidden_dim: int, bottleneck_dim: int, dropout: float, nn_module):
    return nn_module.Sequential(
        nn_module.Linear(input_dim, hidden_dim),
        nn_module.ReLU(),
        nn_module.Dropout(dropout),
        nn_module.Linear(hidden_dim, bottleneck_dim),
        nn_module.ReLU(),
        nn_module.Dropout(dropout),
        nn_module.Linear(bottleneck_dim, 1),
    )


class StrategyRanker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "nb",
        nb_alpha: float = 1.0,
        feature_dim: int = 2048,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        positive_class_weight: float = 1.0,
        seed: int = 42,
        encoder_backend: str = "auto",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.pipeline = None
        self.model = None
        self.model_path = model_path
        self.model_type = model_type
        self.nb_alpha = nb_alpha
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.positive_class_weight = positive_class_weight
        self.seed = seed
        self.encoder_backend = encoder_backend
        self.encoder_name = encoder_name
        self.input_dim = feature_dim
        self.decision_threshold = 0.0
        self.encoder = TextEncoder(backend=encoder_backend, model_name=encoder_name, hash_dim=feature_dim)
        if model_path:
            self.load(model_path)

    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def _build_pair_text(self, context_plus_features: str, strategy_text: str) -> str:
        return f"context: {context_plus_features} [STRATEGY] {strategy_text}"

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _hash_vector(self, text: str) -> List[float]:
        vector, _ = self.encoder.encode_one(text)
        return vector

    def _vectorize_texts(self, texts: List[str]) -> List[List[float]]:
        vectors, input_dim = self.encoder.encode_many(texts)
        self.input_dim = input_dim
        return vectors

    def _fit_nb(self, texts: List[str], labels: List[int]) -> None:
        pos_counts = Counter()
        neg_counts = Counter()
        pos_docs = 0
        neg_docs = 0
        vocab = set()
        for text, label in zip(texts, labels):
            toks = self._tokenize(text)
            vocab.update(toks)
            if int(label) == 1:
                pos_counts.update(toks)
                pos_docs += 1
            else:
                neg_counts.update(toks)
                neg_docs += 1
        self.pipeline = {
            "model_type": "nb",
            "pos_counts": pos_counts,
            "neg_counts": neg_counts,
            "pos_total": sum(pos_counts.values()),
            "neg_total": sum(neg_counts.values()),
            "pos_docs": pos_docs,
            "neg_docs": neg_docs,
            "vocab": vocab,
            "vocab_size": max(len(vocab), 1),
            "nb_alpha": self.nb_alpha,
            "decision_threshold": 0.0,
        }
        self.model = None

    def _resolve_positive_class_weight(self, labels: List[int]) -> float:
        if self.positive_class_weight > 0:
            return self.positive_class_weight
        positives = sum(int(label == 1) for label in labels)
        negatives = max(len(labels) - positives, 0)
        if positives == 0:
            return 1.0
        return max(negatives / positives, 1.0)

    def _positive_metrics(self, probs: List[float], labels: List[int], threshold: float) -> Dict[str, float]:
        tp = fp = fn = tn = 0
        for prob, gold in zip(probs, labels):
            pred = 1 if prob >= threshold else 0
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
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
        }

    def _search_best_threshold(self, probs: List[float], labels: List[int]) -> tuple[float, Dict[str, float]]:
        if not probs:
            return 0.5, self._positive_metrics([], labels, 0.5)
        candidates = sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, *probs})
        best_threshold = 0.5
        best_metrics = self._positive_metrics(probs, labels, best_threshold)
        best_key = (best_metrics["f1"], best_metrics["recall"], best_metrics["precision"], -abs(best_threshold - 0.5))
        for threshold in candidates:
            metrics = self._positive_metrics(probs, labels, threshold)
            key = (metrics["f1"], metrics["recall"], metrics["precision"], -abs(threshold - 0.5))
            if key > best_key:
                best_threshold = threshold
                best_metrics = metrics
                best_key = key
        return float(best_threshold), best_metrics

    def _fit_mlp(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: List[str] | None = None,
        val_labels: List[int] | None = None,
        early_stop_patience: int = 0,
        progress_callback: Callable[[Dict[str, float]], None] | None = None,
    ) -> None:
        torch, nn, optim = _load_torch_modules()
        if torch is None or nn is None or optim is None:
            self._fit_nb(texts, labels)
            return
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        features = torch.tensor(self._vectorize_texts(texts), dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.float32)
        bottleneck_dim = max(self.hidden_dim // 4, 64)
        self.model = _build_pair_mlp(self.input_dim, self.hidden_dim, bottleneck_dim, self.dropout, nn)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        resolved_pos_weight = self._resolve_positive_class_weight(labels)
        pos_weight = torch.tensor([max(resolved_pos_weight, 1e-6)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        indices = list(range(len(labels)))

        best_val = (-1.0, -1.0, -1.0)
        patience = 0
        best_state = None
        best_threshold = 0.5
        best_metrics = None

        for epoch in range(max(self.epochs, 1)):
            self.model.train()
            random.shuffle(indices)
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(indices), max(self.batch_size, 1)):
                batch_ids = indices[start:start + max(self.batch_size, 1)]
                batch_x = features[batch_ids]
                batch_y = targets[batch_ids]
                optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(-1)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
                batch_count += 1

            epoch_log: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "total_epochs": float(max(self.epochs, 1)),
                "train_loss": epoch_loss / max(batch_count, 1),
            }

            # validation check
            if val_texts is not None and val_labels is not None and len(val_texts) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_feats = torch.tensor(self._vectorize_texts(val_texts), dtype=torch.float32)
                    val_logits = self.model(val_feats).squeeze(-1)
                    val_probs = torch.sigmoid(val_logits).detach().cpu().tolist()
                epoch_threshold, epoch_metrics = self._search_best_threshold(val_probs, val_labels)
                score_key = (epoch_metrics["f1"], epoch_metrics["recall"], epoch_metrics["precision"])
                if score_key > best_val:
                    best_val = score_key
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_threshold = epoch_threshold
                    best_metrics = epoch_metrics
                    epoch_log["best_epoch"] = 1.0
                else:
                    patience += 1
                    epoch_log["best_epoch"] = 0.0
                    if early_stop_patience and patience >= early_stop_patience:
                        epoch_log["stopped_early"] = 1.0
                        epoch_log["val_threshold"] = float(epoch_threshold)
                        epoch_log["val_accuracy"] = epoch_metrics["accuracy"]
                        epoch_log["val_precision"] = epoch_metrics["precision"]
                        epoch_log["val_recall"] = epoch_metrics["recall"]
                        epoch_log["val_f1"] = epoch_metrics["f1"]
                        if progress_callback is not None:
                            progress_callback(epoch_log)
                        break
                epoch_log["val_threshold"] = float(epoch_threshold)
                epoch_log["val_accuracy"] = epoch_metrics["accuracy"]
                epoch_log["val_precision"] = epoch_metrics["precision"]
                epoch_log["val_recall"] = epoch_metrics["recall"]
                epoch_log["val_f1"] = epoch_metrics["f1"]

            if progress_callback is not None:
                progress_callback(epoch_log)

        # restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.decision_threshold = best_threshold if best_metrics is not None else 0.0

        self.pipeline = {
            "model_type": "mlp",
            "encoder_backend": self.encoder.backend,
            "encoder_name": self.encoder_name,
            "feature_dim": self.feature_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "positive_class_weight": resolved_pos_weight,
            "decision_threshold": self.decision_threshold,
            "val_positive_metrics": best_metrics,
            "seed": self.seed,
        }

    def fit(self, pair_contexts: List[str], strategy_texts: List[str], labels: List[int]) -> None:
        texts = [self._build_pair_text(c, s) for c, s in zip(pair_contexts, strategy_texts)]
        # support optional validation data passed via kwargs
        val_texts = None
        val_labels = None
        early_stop_patience = 0
        progress_callback = getattr(self, "_progress_callback", None)
        # allow callers to pass attributes set on self temporarily
        if hasattr(self, "_val_texts"):
            val_texts = getattr(self, "_val_texts")
            val_labels = getattr(self, "_val_labels")
            early_stop_patience = getattr(self, "_early_stop_patience", 0)

        if self.model_type == "mlp":
            self._fit_mlp(
                texts,
                labels,
                val_texts=val_texts,
                val_labels=val_labels,
                early_stop_patience=early_stop_patience,
                progress_callback=progress_callback,
            )
        else:
            self._fit_nb(texts, labels)
            if progress_callback is not None:
                progress_callback({"epoch": 1.0, "total_epochs": 1.0, "train_loss": 0.0, "used_nb": 1.0})

    def _score_nb(self, texts: List[str]) -> List[float]:
        pos_counts = self.pipeline["pos_counts"]
        neg_counts = self.pipeline["neg_counts"]
        pos_total = self.pipeline["pos_total"]
        neg_total = self.pipeline["neg_total"]
        pos_docs = self.pipeline["pos_docs"]
        neg_docs = self.pipeline["neg_docs"]
        vocab_size = self.pipeline["vocab_size"]
        alpha = self.pipeline.get("nb_alpha", 1.0)
        total_docs = max(pos_docs + neg_docs, 1)
        log_prior_pos = math.log((pos_docs + alpha) / (total_docs + 2 * alpha))
        log_prior_neg = math.log((neg_docs + alpha) / (total_docs + 2 * alpha))
        out = []
        for text in texts:
            toks = self._tokenize(text)
            lp = log_prior_pos
            ln = log_prior_neg
            for tok in toks:
                lp += math.log((pos_counts.get(tok, 0) + alpha) / (pos_total + alpha * vocab_size))
                ln += math.log((neg_counts.get(tok, 0) + alpha) / (neg_total + alpha * vocab_size))
            out.append(lp - ln)
        return out

    def _score_mlp(self, texts: List[str]) -> List[float]:
        torch, _, _ = _load_torch_modules()
        if torch is None or self.model is None:
            return [0.0 for _ in texts]
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(self._vectorize_texts(texts), dtype=torch.float32)
            logits = self.model(features).squeeze(-1)
            return logits.detach().cpu().tolist()

    def score(self, context_plus_features: str, strategy_texts: List[str]) -> List[float]:
        if not strategy_texts:
            return []
        if self.pipeline is None:
            return [0.0 for _ in strategy_texts]
        texts = [self._build_pair_text(context_plus_features, st) for st in strategy_texts]
        model_type = self.pipeline.get("model_type", "nb") if isinstance(self.pipeline, dict) else "nb"
        if model_type == "mlp":
            return self._score_mlp(texts)
        return self._score_nb(texts)

    def save(self, model_path: str | Path) -> None:
        if self.pipeline is None:
            raise ValueError("Cannot save an unfitted ranker")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "pipeline": self.pipeline,
            "model_type": self.pipeline.get("model_type", self.model_type) if isinstance(self.pipeline, dict) else self.model_type,
        }
        if payload["model_type"] == "mlp" and self.model is not None:
            payload["state_dict"] = self.model.state_dict()
        with path.open("wb") as f:
            pickle.dump(payload, f)
        self.model_path = str(path)

    def load(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            self.pipeline = None
            self.model = None
            self.model_path = str(path)
            return
        with path.open("rb") as f:
            payload = pickle.load(f)

        # backward compatibility with the old NB-only payload
        if isinstance(payload, dict) and "pipeline" in payload:
            self.pipeline = payload.get("pipeline")
            model_type = payload.get("model_type", self.pipeline.get("model_type", "nb"))
        else:
            self.pipeline = payload
            model_type = self.pipeline.get("model_type", "nb") if isinstance(self.pipeline, dict) else "nb"

        self.model_type = model_type
        self.model = None
        if self.model_type == "mlp" and isinstance(self.pipeline, dict):
            torch, nn, _ = _load_torch_modules()
            self.feature_dim = int(self.pipeline.get("feature_dim", self.feature_dim))
            self.input_dim = int(self.pipeline.get("input_dim", self.feature_dim))
            self.hidden_dim = int(self.pipeline.get("hidden_dim", self.hidden_dim))
            self.dropout = float(self.pipeline.get("dropout", self.dropout))
            self.encoder_backend = str(self.pipeline.get("encoder_backend", self.encoder_backend))
            self.encoder_name = str(self.pipeline.get("encoder_name", self.encoder_name))
            self.decision_threshold = float(self.pipeline.get("decision_threshold", self.decision_threshold))
            self.encoder = TextEncoder(backend=self.encoder_backend, model_name=self.encoder_name, hash_dim=self.feature_dim)
            if torch is not None and nn is not None:
                bottleneck_dim = max(self.hidden_dim // 4, 64)
                self.model = _build_pair_mlp(self.input_dim, self.hidden_dim, bottleneck_dim, self.dropout, nn)
                state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
                if state_dict:
                    self.model.load_state_dict(state_dict)
        self.model_path = str(path)
