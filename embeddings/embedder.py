from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddingMeta:
    backend: str
    model_name: str
    dim: int


class TextEmbedder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        prefer_hf: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.prefer_hf = prefer_hf
        self._model = None

    def _load_sentence_transformer(self):
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            if self.device is None:
                self._model = SentenceTransformer(self.model_name)
            else:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> tuple[np.ndarray, EmbeddingMeta]:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32), EmbeddingMeta(
                backend="none",
                model_name=self.model_name,
                dim=0,
            )

        if self.prefer_hf:
            try:
                model = self._load_sentence_transformer()
                emb = model.encode(
                    text_list,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                ).astype(np.float32)
                return emb, EmbeddingMeta(backend="sentence-transformers", model_name=self.model_name, dim=int(emb.shape[1]))
            except Exception:
                pass

        emb = _hashing_vectorizer_embeddings(text_list, dim=256)
        if normalize:
            emb = _l2_normalize(emb)
        return emb, EmbeddingMeta(backend="hashing-vectorizer", model_name="sklearn.HashingVectorizer", dim=int(emb.shape[1]))


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _hashing_vectorizer_embeddings(texts: List[str], *, dim: int) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None)
    x = hv.transform(texts)
    return x.astype(np.float32).toarray()

