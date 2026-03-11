from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class QwenClient:
    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        enable_thinking: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = str(Path(model_path))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")

    candidates = []
    if "<json>" in text and "</json>" in text:
        for part in text.split("<json>")[1:]:
            block = part.split("</json>", 1)[0].strip()
            if block:
                candidates.append(block)
    if "```json" in text:
        for part in text.split("```json")[1:]:
            block = part.split("```", 1)[0].strip()
            if block:
                candidates.append(block)
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to parse JSON from model output")
