#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate COMET/ATOMIC commonsense cues for emotion-labeled dialogs and save to JSONL.
- Input: JSONL where each line has at least `pred_emotion` and `reason` fields
         (optionally `dialog`). See /data2/xqchen/Qwen3_test/output/emotion_prediction_*.jsonl.
- Output: JSONL with added `commonsense_head` and `commonsense` (list of {relation, tail}).

Example:
python commonsense_infer.py \
  --input_file /data2/xqchen/Qwen3_test/output/emotion_prediction_20260310_102114.jsonl \
  --output_dir /data2/xqchen/Judge/output \
  --model_path /data2/xqchen/Judge/model/comet-atomic_2020_BART
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RELATIONS = [
    "xIntent",
    "xReact",
    "oReact",
    "xEffect",
    "oEffect",
    "xNeed",
    "xWant",
    "oWant",
]

# 默认路径（直接硬编码，避免每次命令行输入）
DEFAULT_INPUT_FILE = Path("/data2/xqchen/Qwen3_test/output/emotion_prediction_20260310_102114.jsonl")
DEFAULT_OUTPUT_DIR = Path("/data2/xqchen/Judge/output")
DEFAULT_MODEL_PATH = "/data2/xqchen/Judge/model/comet-atomic_2020_BART"


class CometGenerator:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_gen_token: bool = True,
        max_new_tokens: int = 32,
        num_beams: int = 5,
        num_return_sequences: int = 3,
    ) -> None:
        self.device = device
        self.use_gen_token = use_gen_token
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    def _format_query(self, head: str, relation: str) -> str:
        suffix = " [GEN]" if self.use_gen_token else ""
        return f"{head} {relation}{suffix}"

    def generate(self, head: str, relations: List[str]) -> List[Dict[str, str]]:
        """Return a list of commonsense facts as {relation, tail} pairs."""
        queries = [self._format_query(head, rel) for rel in relations]
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=max(self.num_beams, self.num_return_sequences),
                num_return_sequences=self.num_return_sequences,
                max_new_tokens=self.max_new_tokens,
                early_stopping=True,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        facts: List[Dict[str, str]] = []
        for idx, rel in enumerate(relations):
            start = idx * self.num_return_sequences
            end = start + self.num_return_sequences
            tails = [t.strip() for t in decoded[start:end] if t.strip()]
            for tail in tails:
                facts.append({"relation": rel, "tail": tail})
        return facts


def build_head(sample: Dict[str, Any]) -> str:
    """Construct a concise head event for COMET based on reason + predicted emotion."""
    emotion = sample.get("pred_emotion") or sample.get("gold_emotion") or ""
    reason = sample.get("reason") or ""
    head = f"PersonX feels {emotion} because {reason}".strip()
    if not head:
        dialog = sample.get("dialog", [])
        utterances = " ".join(turn.get("text", "") for turn in dialog)
        head = utterances[:300]
    return head


def process_file(
    input_file: Path,
    output_dir: Path,
    model_path: str,
    use_gen_token: bool,
    relations: List[str],
    device: str,
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"commonsense_{timestamp}.jsonl"

    generator = CometGenerator(
        model_path=model_path,
        device=device,
        use_gen_token=use_gen_token,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            head = build_head(sample)
            facts = generator.generate(head, relations)
            sample["commonsense_head"] = head
            sample["commonsense"] = facts
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate COMET commonsense facts for emotion predictions.")
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT_FILE, help="Path to input JSONL (emotion_prediction_*.jsonl)")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save enriched JSONL")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="COMET/ATOMIC model checkpoint path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no_gen_token", action="store_true", help="Disable [GEN] suffix (useful for _aaai checkpoint)")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam width for generation")
    parser.add_argument("--num_return_sequences", type=int, default=3, help="Number of candidates per relation")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max tokens to generate per tail")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    output_path = process_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_path=args.model_path,
        use_gen_token=not args.no_gen_token,
        relations=RELATIONS,
        device=device,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Saved commonsense-enriched file to {output_path}")
