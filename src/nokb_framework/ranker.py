"""Cross-encoder style ranker for strategy selection (no KB)."""
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StrategyRanker:
    def __init__(self, model_name: str = "bert-base-chinese", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, context_plus_comet: str, strategy_texts: List[str]) -> List[float]:
        """Return a score for each strategy text given fused context+commonsense text."""
        if not strategy_texts:
            return []
        pairs = [(context_plus_comet, st) for st in strategy_texts]
        batch = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(**batch).logits  # (batch, 1)
        return logits.squeeze(-1).cpu().tolist()

    def train_step(self, batch_inputs, optimizer, loss_fn):
        """Placeholder for fine-tuning (binary classification or ranking)."""
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch_inputs.items()}
        outputs = self.model(**batch)
        loss = loss_fn(outputs.logits.squeeze(-1), batch.get("labels"))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.model.eval()
        return float(loss.detach().cpu())
