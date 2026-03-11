"""Trainable mapping from HEAL response text to strategy category."""
from __future__ import annotations

import math
import importlib
import pickle
import random
import re
from collections import Counter, defaultdict
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


def _build_mapper_mlp(input_dim: int, hidden_dim: int, bottleneck_dim: int, num_labels: int, dropout: float, nn_module):
    return nn_module.Sequential(
        nn_module.Linear(input_dim, hidden_dim),
        nn_module.ReLU(),
        nn_module.Dropout(dropout),
        nn_module.Linear(hidden_dim, bottleneck_dim),
        nn_module.ReLU(),
        nn_module.Dropout(dropout),
        nn_module.Linear(bottleneck_dim, num_labels),
    )


class ResponseStrategyMapper:
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
        seed: int = 42,
        encoder_backend: str = "auto",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.pipeline = None
        self.labels_: List[str] = []
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
        self.seed = seed
        self.encoder_backend = encoder_backend
        self.encoder_name = encoder_name
        self.input_dim = feature_dim
        self.encoder = TextEncoder(backend=encoder_backend, model_name=encoder_name, hash_dim=feature_dim)
        if model_path:
            self.load(model_path)

    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _hash_vector(self, text: str) -> List[float]:
        vector, _ = self.encoder.encode_one(text)
        return vector

    def _vectorize_texts(self, texts: List[str]) -> List[List[float]]:
        vectors, input_dim = self.encoder.encode_many(texts)
        self.input_dim = input_dim
        return vectors

    def _fit_nb(self, texts: List[str], labels: List[str]) -> None:
        class_doc_counts = Counter(labels)
        token_counts = defaultdict(Counter)
        token_totals = Counter()
        vocab = set()
        for text, label in zip(texts, labels):
            toks = self._tokenize(text)
            vocab.update(toks)
            token_counts[label].update(toks)
            token_totals[label] += len(toks)
        self.labels_ = sorted(list(set(labels)))
        self.pipeline = {
            "model_type": "nb",
            "class_doc_counts": class_doc_counts,
            "token_counts": dict(token_counts),
            "token_totals": token_totals,
            "vocab": vocab,
            "vocab_size": max(len(vocab), 1),
            "total_docs": max(len(labels), 1),
            "nb_alpha": self.nb_alpha,
        }
        self.model = None

    def _fit_mlp(self, texts: List[str], labels: List[str], progress_callback: Callable[[Dict[str, float]], None] | None = None) -> None:
        torch, nn, optim = _load_torch_modules()
        if torch is None or nn is None or optim is None:
            self._fit_nb(texts, labels)
            return
        self.labels_ = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(self.labels_)}
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        features = torch.tensor(self._vectorize_texts(texts), dtype=torch.float32)
        targets = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)
        bottleneck_dim = max(self.hidden_dim // 4, 64)
        self.model = _build_mapper_mlp(self.input_dim, self.hidden_dim, bottleneck_dim, len(self.labels_), self.dropout, nn)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        indices = list(range(len(labels)))

        best_val = -1e9
        patience = 0
        best_state = None

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
                logits = self.model(batch_x)
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

            # early-stop handled externally via temporary attributes
            val_texts = getattr(self, "_val_texts", None)
            val_labels = getattr(self, "_val_labels", None)
            early_stop_patience = getattr(self, "_early_stop_patience", 0)
            if val_texts is not None and val_labels is not None and len(val_texts) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_feats = torch.tensor(self._vectorize_texts(val_texts), dtype=torch.float32)
                    val_logits = self.model(val_feats)
                    val_preds = val_logits.argmax(dim=-1).tolist()
                correct = sum(int(p == label_to_idx[g]) for p, g in zip(val_preds, val_labels))
                val_acc = correct / max(len(val_labels), 1)
                epoch_log["val_accuracy"] = float(val_acc)
                if val_acc > best_val:
                    best_val = val_acc
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epoch_log["best_epoch"] = 1.0
                else:
                    patience += 1
                    epoch_log["best_epoch"] = 0.0
                    if early_stop_patience and patience >= early_stop_patience:
                        epoch_log["stopped_early"] = 1.0
                        if progress_callback is not None:
                            progress_callback(epoch_log)
                        break

            if progress_callback is not None:
                progress_callback(epoch_log)

        if best_state is not None:
            self.model.load_state_dict(best_state)

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
            "seed": self.seed,
            "labels": self.labels_,
        }

    def fit(self, texts: List[str], labels: List[str]) -> None:
        progress_callback = getattr(self, "_progress_callback", None)
        if self.model_type == "mlp":
            self._fit_mlp(texts, labels, progress_callback=progress_callback)
        else:
            self._fit_nb(texts, labels)
            if progress_callback is not None:
                progress_callback({"epoch": 1.0, "total_epochs": 1.0, "train_loss": 0.0, "used_nb": 1.0})

    def predict(self, text: str) -> str:
        if self.pipeline is None:
            return "other"
        scores = self.predict_scores(text)
        if not scores:
            return "other"
        return max(scores.items(), key=lambda x: x[1])[0]

    def _predict_scores_nb(self, text: str) -> Dict[str, float]:
        toks = self._tokenize(text)
        class_doc_counts = self.pipeline["class_doc_counts"]
        token_counts = self.pipeline["token_counts"]
        token_totals = self.pipeline["token_totals"]
        vocab_size = self.pipeline["vocab_size"]
        total_docs = self.pipeline["total_docs"]
        alpha = self.pipeline.get("nb_alpha", 1.0)
        log_scores = {}
        for label in self.labels_:
            logp = math.log((class_doc_counts.get(label, 0) + alpha) / (total_docs + len(self.labels_) * alpha))
            counts = token_counts.get(label, Counter())
            total = token_totals.get(label, 0)
            for tok in toks:
                logp += math.log((counts.get(tok, 0) + alpha) / (total + alpha * vocab_size))
            log_scores[label] = logp
        max_log = max(log_scores.values()) if log_scores else 0.0
        exps = {k: math.exp(v - max_log) for k, v in log_scores.items()}
        z = sum(exps.values()) or 1.0
        return {k: float(v / z) for k, v in exps.items()}

    def _predict_scores_mlp(self, text: str) -> Dict[str, float]:
        torch, _, _ = _load_torch_modules()
        if torch is None or self.model is None or not self.labels_:
            return {}
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(self._vectorize_texts([text]), dtype=torch.float32)
            logits = self.model(features)[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        return {label: float(prob) for label, prob in zip(self.labels_, probs)}

    def predict_scores(self, text: str) -> Dict[str, float]:
        if self.pipeline is None:
            return {}
        model_type = self.pipeline.get("model_type", "nb") if isinstance(self.pipeline, dict) else "nb"
        if model_type == "mlp":
            return self._predict_scores_mlp(text)
        return self._predict_scores_nb(text)

    def save(self, model_path: str | Path) -> None:
        if self.pipeline is None:
            raise ValueError("Cannot save an unfitted mapper")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "pipeline": self.pipeline,
            "model_type": self.pipeline.get("model_type", self.model_type) if isinstance(self.pipeline, dict) else self.model_type,
            "labels": self.labels_,
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

        if isinstance(payload, dict) and "pipeline" in payload:
            self.pipeline = payload.get("pipeline")
            self.labels_ = list(payload.get("labels") or self.pipeline.get("labels") or [])
            model_type = payload.get("model_type", self.pipeline.get("model_type", "nb"))
        else:
            self.pipeline = payload
            self.labels_ = sorted(list(self.pipeline.get("class_doc_counts", {}).keys())) if isinstance(self.pipeline, dict) else []
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
            self.encoder = TextEncoder(backend=self.encoder_backend, model_name=self.encoder_name, hash_dim=self.feature_dim)
            if not self.labels_:
                self.labels_ = list(self.pipeline.get("labels") or [])
            if torch is not None and nn is not None and self.labels_:
                bottleneck_dim = max(self.hidden_dim // 4, 64)
                self.model = _build_mapper_mlp(self.input_dim, self.hidden_dim, bottleneck_dim, len(self.labels_), self.dropout, nn)
                state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
                if state_dict:
                    self.model.load_state_dict(state_dict)
        self.model_path = str(path)
