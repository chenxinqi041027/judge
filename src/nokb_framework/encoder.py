"""Lightweight text encoder using HuggingFace models (BERT-base by default)."""
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel


class TextEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts; returns tensor of shape (len(texts), hidden_size)."""
        if not texts:
            return torch.zeros(0, self.model.config.hidden_size, device=self.device)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state  # (batch, seq, hidden)
        # Mean pooling with attention mask
        mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return pooled
