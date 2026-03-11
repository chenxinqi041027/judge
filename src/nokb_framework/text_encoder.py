from __future__ import annotations

import hashlib
import importlib
import re
from typing import List, Sequence, Tuple


def _load_sentence_transformer():
    try:
        module = importlib.import_module("sentence_transformers")
        return getattr(module, "SentenceTransformer", None)
    except Exception:
        return None


class TextEncoder:
    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hash_dim: int = 2048,
    ):
        self.requested_backend = backend
        self.model_name = model_name
        self.hash_dim = hash_dim
        self.backend = backend
        self.model = None
        self.output_dim = hash_dim
        self._resolve_backend()

    def _resolve_backend(self) -> None:
        if self.requested_backend == "hash":
            self.backend = "hash"
            self.output_dim = self.hash_dim
            return

        sentence_transformer_cls = _load_sentence_transformer()
        if sentence_transformer_cls is None:
            if self.requested_backend == "sbert":
                raise ImportError("sentence-transformers is required when encoder backend is set to 'sbert'")
            self.backend = "hash"
            self.output_dim = self.hash_dim
            return

        try:
            self.model = sentence_transformer_cls(self.model_name)
            self.backend = "sbert"
            self.output_dim = int(self.model.get_sentence_embedding_dimension())
            return
        except Exception as exc:
            self.model = None
            if self.requested_backend == "sbert":
                raise RuntimeError(
                    f"Failed to load sentence-transformer model '{self.model_name}'. "
                    "Ensure the model is accessible from the configured Hugging Face endpoint."
                ) from exc

        self.backend = "hash"
        self.output_dim = self.hash_dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _stable_hash_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % self.hash_dim

    def _encode_hash(self, text: str) -> List[float]:
        vector = [0.0] * self.hash_dim
        for token in self._tokenize(text):
            vector[self._stable_hash_index(token)] += 1.0
        return vector

    def encode_many(self, texts: Sequence[str]) -> Tuple[List[List[float]], int]:
        if self.backend == "sbert" and self.model is not None:
            embeddings = self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist(), self.output_dim
        return [self._encode_hash(text) for text in texts], self.hash_dim

    def encode_one(self, text: str) -> Tuple[List[float], int]:
        vectors, dim = self.encode_many([text])
        return vectors[0], dim