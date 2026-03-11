#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMET commonsense generator wrapper.
- Optional: load a COMET/ATOMIC model (e.g., `mismtallen/comet-atomic-2020`).
- If model_path is missing, the client stays inactive (is_available=False) and returns empty generations.
- TODO: download/prepare a COMET checkpoint under /data2/xqchen/Judge/commonsense and pass its path.
"""

from typing import Any, Dict, List
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class CometClient:
    def __init__(self, model_path: str = "", device: str = "cpu", max_new_tokens: int = 32):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

        if model_path:
            p = Path(model_path)
            if p.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            else:
                # model path not found; stay inactive
                self.model = None

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(self, head: str, relations: List[str], num_beams: int = 4, num_return_sequences: int = 2) -> List[Dict[str, Any]]:
        """Generate commonsense tails for each relation; returns list of {relation, tail}."""
        if not self.is_available():
            return []

        results: List[Dict[str, Any]] = []
        for rel in relations:
            prompt = f"{head} {rel}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=self.max_new_tokens,
                    early_stopping=True,
                )
            for seq in outputs:
                tail = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                results.append({"relation": rel, "tail": tail})
        return results


# Preset ATOMIC relations commonly used by COMET models.
DEFAULT_RELATIONS = [
    "oEffect", "oReact", "oWant",  # effect on others
    "xNeed", "xIntent", "xReact", "xWant",  # effect on actor
]


def build_comet_client_or_none() -> CometClient:
    """
    Helper to construct a COMET client. If no checkpoint exists locally, returns an inactive client.
    TODO: set `model_path` to your COMET checkpoint, e.g., /data2/xqchen/Judge/commonsense/comet-atomic-2020.
    """
    # 默认指向你存放的模型目录
    candidate = Path("/data2/xqchen/Judge/model/comet-atomic_2020_BART")
    model_path = str(candidate) if candidate.exists() else ""
    return CometClient(model_path=model_path, device="cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    client = build_comet_client_or_none()
    if client.is_available():
        gens = client.generate("person is afraid of a tire burst", DEFAULT_RELATIONS, num_return_sequences=1)
        for g in gens:
            print(g)
    else:
        print("COMET model not found; please place a checkpoint under /data2/xqchen/Judge/commonsense and set model_path.")
