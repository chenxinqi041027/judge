#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper to forward to src/strategy_selector.py."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy_selector import load_kb, attach_strategy, select_strategy  # type: ignore


def demo():
    kb = load_kb()
    sample = {
        "pred_emotion": "afraid",
        "reason": "The speaker was scared by a tire burst on a busy road",
    }
    enriched = attach_strategy(sample, kb)
    print(json.dumps(enriched, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
