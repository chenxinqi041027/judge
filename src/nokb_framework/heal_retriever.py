"""Lightweight HEAL graph retriever.

Uses the local HEAL graph text files under /data2/xqchen/Judge/model/HEAL/HEAL
without relying on the missing precomputed pickle dumps.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower()))


def _safe_text(node: Dict[str, Any]) -> str:
    pieces = [str(node.get("label", ""))]
    if node.get("story"):
        pieces.append(str(node["story"]))
    labels = node.get("labels") or []
    if isinstance(labels, list):
        pieces.extend(str(x) for x in labels[:8])
    return " ".join(x for x in pieces if x)


class HealRetriever:
    def __init__(self, heal_root: str | Path, response_mapper=None):
        self.heal_root = Path(heal_root)
        self.response_mapper = response_mapper
        self.graph_root = self.heal_root / "HEAL"
        self.topic_path = self.heal_root / "topic_dict.txt"

        self.stressors = self._load_nodes(self.graph_root / "nodes" / "stressors.txt")
        self.expectations = self._load_nodes(self.graph_root / "nodes" / "expectations.txt")
        self.responses = self._load_nodes(self.graph_root / "nodes" / "responses.txt")
        self.affective_states = self._load_nodes(self.graph_root / "nodes" / "affective_states.txt")

        self.stressor_to_response = self._load_edge_map(self.graph_root / "edges" / "stressors-responses.txt")
        self.expectation_to_response = self._load_edge_map(self.graph_root / "edges" / "expectations-responses.txt")
        self.stressor_to_expectation = self._load_edge_map(self.graph_root / "edges" / "expectations-stressors.txt", reverse_from_to=True)

        self.topic_dict = {}
        if self.topic_path.exists():
            try:
                self.topic_dict = json.loads(self.topic_path.read_text(encoding="utf-8"))
            except Exception:
                self.topic_dict = {}

        self.stressor_by_id = {n["id"]: n for n in self.stressors}
        self.expectation_by_id = {n["id"]: n for n in self.expectations}
        self.response_by_id = {n["id"]: n for n in self.responses}
        self.affective_by_stressor = self._build_affective_by_stressor()

    def is_available(self) -> bool:
        return bool(self.stressors and self.responses)

    def _load_nodes(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj.get("nodes", [])

    def _load_edge_map(self, path: Path, reverse_from_to: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        edge_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in obj.get("edges", []):
            src = edge.get("to") if reverse_from_to else edge.get("from")
            dst = edge.get("from") if reverse_from_to else edge.get("to")
            edge_map[src].append({"id": dst, "value": float(edge.get("value", 0.0))})
        return dict(edge_map)

    def _build_affective_by_stressor(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in self.affective_states:
            node_id = str(node.get("id", ""))
            if "-" not in node_id:
                continue
            stressor_id = node_id.split("-")[-1]
            out[stressor_id].append(node)
        return dict(out)

    def _lexical_score(self, query_tokens: set, text: str, freq: float = 1.0) -> float:
        target_tokens = _tokenize(text)
        if not query_tokens or not target_tokens:
            return 0.0
        inter = len(query_tokens & target_tokens)
        union = len(query_tokens | target_tokens)
        jaccard = inter / union if union else 0.0
        tf_bonus = min(inter / max(1.0, len(query_tokens)), 1.0)
        freq_bonus = min(math.log1p(max(freq, 0.0)) / 10.0, 0.5)
        return 0.6 * jaccard + 0.3 * tf_bonus + 0.1 * freq_bonus

    def _emotion_bonus(self, stressor_id: str, emotion: str) -> float:
        if not emotion:
            return 0.0
        emotion = emotion.lower().strip()
        affs = self.affective_by_stressor.get(stressor_id, [])
        for aff in affs:
            label = str(aff.get("label", "")).lower()
            if emotion in label:
                return 0.25
            # rough mapping
            if emotion == "depression" and label in {"sad", "hopeless", "lonely"}:
                return 0.2
            if emotion == "anxiety" and label in {"apprehensive", "afraid", "anxious"}:
                return 0.2
            if emotion == "sadness" and label == "sad":
                return 0.2
            if emotion == "anger" and label in {"angry", "frustrated"}:
                return 0.2
        return 0.0

    def _response_text(self, response_node: Dict[str, Any]) -> str:
        labels = response_node.get("labels") or []
        if isinstance(labels, list) and labels:
            return str(labels[0])
        return str(response_node.get("label", ""))

    def map_response_to_strategy(self, text: str) -> str:
        if self.response_mapper is not None and getattr(self.response_mapper, "is_fitted", lambda: False)():
            try:
                pred = self.response_mapper.predict(text)
                if pred:
                    return str(pred)
            except Exception:
                pass
        t = (text or "").lower()
        if "?" in t or any(x in t for x in ["what ", "how ", "could you", "would you", "can you"]):
            return "questions"
        if any(x in t for x in ["i know how you feel", "i've been", "i have been", "as someone who", "from personal experience"]):
            return "self-disclosure"
        if any(x in t for x in ["sorry", "you are not alone", "it will be okay", "you're going through", "that sounds hard", "i'm glad", "i am sorry"]):
            return "affirmation and reassurance"
        if any(x in t for x in ["you feel", "it sounds like", "seems like", "you seem", "must feel"]):
            return "reflection of feelings"
        if any(x in t for x in ["try", "consider", "maybe", "perhaps", "you could", "it might help", "suggest"]):
            return "providing suggestions"
        if any(x in t for x in ["means", "because", "often", "this is", "it is normal", "typically"]):
            return "information"
        if any(x in t for x in ["what i hear", "so you", "in other words", "you mean"]):
            return "restatement or paraphrasing"
        return "other"

    def retrieve(self, expanded_query: str, emotion: str = "", top_k_stressors: int = 3, top_k_expectations: int = 3, top_k_responses: int = 5) -> Dict[str, Any]:
        query_tokens = _tokenize(expanded_query)

        scored_stressors = []
        for node in self.stressors:
            score = self._lexical_score(query_tokens, _safe_text(node), float(node.get("value", 0.0)))
            score += self._emotion_bonus(str(node.get("id")), emotion)
            if score > 0:
                scored_stressors.append({
                    "id": node.get("id"),
                    "label": node.get("label"),
                    "story": node.get("story", ""),
                    "score": round(float(score), 4),
                    "affective_states": [a.get("label") for a in self.affective_by_stressor.get(str(node.get("id")), [])[:5]],
                })
        scored_stressors.sort(key=lambda x: x["score"], reverse=True)
        top_stressors = scored_stressors[:top_k_stressors]

        scored_expectations = []
        for node in self.expectations:
            score = self._lexical_score(query_tokens, _safe_text(node), float(node.get("value", 0.0)))
            if score > 0:
                scored_expectations.append({
                    "id": node.get("id"),
                    "label": node.get("label"),
                    "score": round(float(score), 4),
                })
        scored_expectations.sort(key=lambda x: x["score"], reverse=True)
        top_expectations = scored_expectations[:top_k_expectations]

        response_scores: Dict[str, Dict[str, Any]] = {}
        for st in top_stressors:
            for edge in self.stressor_to_response.get(st["id"], []):
                rid = edge["id"]
                node = self.response_by_id.get(rid)
                if not node:
                    continue
                base = st["score"] * (1.0 + min(edge.get("value", 0.0), 50.0) / 50.0)
                response_scores.setdefault(rid, {
                    "id": rid,
                    "text": self._response_text(node),
                    "score": 0.0,
                    "sources": [],
                })
                response_scores[rid]["score"] += base
                response_scores[rid]["sources"].append({"type": "stressor", "id": st["id"], "label": st["label"], "edge_value": edge.get("value", 0.0)})

        for ex in top_expectations:
            for edge in self.expectation_to_response.get(ex["id"], []):
                rid = edge["id"]
                node = self.response_by_id.get(rid)
                if not node:
                    continue
                base = ex["score"] * (1.0 + min(edge.get("value", 0.0), 20.0) / 20.0)
                response_scores.setdefault(rid, {
                    "id": rid,
                    "text": self._response_text(node),
                    "score": 0.0,
                    "sources": [],
                })
                response_scores[rid]["score"] += base
                response_scores[rid]["sources"].append({"type": "expectation", "id": ex["id"], "label": ex["label"], "edge_value": edge.get("value", 0.0)})

        scored_responses = list(response_scores.values())
        for item in scored_responses:
            item["strategy"] = self.map_response_to_strategy(item["text"])
            item["score"] = round(float(item["score"]), 4)
        scored_responses.sort(key=lambda x: x["score"], reverse=True)
        top_responses = scored_responses[:top_k_responses]

        strategy_priors: Dict[str, float] = defaultdict(float)
        for resp in top_responses:
            strategy_priors[resp["strategy"]] += resp["score"]
        max_prior = max(strategy_priors.values()) if strategy_priors else 0.0
        if max_prior > 0:
            for k in list(strategy_priors.keys()):
                strategy_priors[k] = round(strategy_priors[k] / max_prior, 4)

        subgraphs = []
        for st in top_stressors:
            linked_expectations = []
            for edge in self.stressor_to_expectation.get(st["id"], [])[:5]:
                node = self.expectation_by_id.get(edge["id"])
                if node:
                    linked_expectations.append({"id": edge["id"], "label": node.get("label"), "edge_value": edge.get("value", 0.0)})
            linked_responses = []
            for edge in self.stressor_to_response.get(st["id"], [])[:5]:
                node = self.response_by_id.get(edge["id"])
                if node:
                    text = self._response_text(node)
                    linked_responses.append({"id": edge["id"], "text": text, "strategy": self.map_response_to_strategy(text), "edge_value": edge.get("value", 0.0)})
            subgraphs.append({
                "stressor": st,
                "expectations": linked_expectations,
                "responses": linked_responses,
            })

        heal_text = self.build_heal_text(top_stressors, top_expectations, top_responses)
        return {
            "expanded_query": expanded_query,
            "top_stressors": top_stressors,
            "top_expectations": top_expectations,
            "top_responses": top_responses,
            "subgraphs": subgraphs,
            "strategy_priors": dict(strategy_priors),
            "heal_text": heal_text,
        }

    def build_heal_text(self, stressors: List[Dict[str, Any]], expectations: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> str:
        parts = []
        for st in stressors[:3]:
            aff = ", ".join(st.get("affective_states", [])[:3])
            parts.append(f"HEAL stressor: {st.get('label')} | affective states: {aff}")
        for ex in expectations[:3]:
            parts.append(f"HEAL expectation: {ex.get('label')}")
        for resp in responses[:5]:
            parts.append(f"HEAL response: {resp.get('text')} | strategy={resp.get('strategy')}")
        return " [SEP] ".join(parts)

    def build_strategy_specific_knowledge(self, heal_result: Dict[str, Any], category: str, limit: int = 3) -> str:
        chosen = [r for r in heal_result.get("top_responses", []) if r.get("strategy") == category][:limit]
        if not chosen:
            chosen = heal_result.get("top_responses", [])[:limit]
        return " [SEP] ".join(
            f"HEAL response for {category}: {r.get('text')}" for r in chosen
        )
