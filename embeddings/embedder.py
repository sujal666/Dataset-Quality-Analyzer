from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
        cache_dir: str | Path = "embeddings/cache",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.prefer_hf = prefer_hf
        self._model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        cache_key: Optional[str] = None,
    ) -> tuple[np.ndarray, EmbeddingMeta]:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32), EmbeddingMeta(
                backend="none",
                model_name=self.model_name,
                dim=0,
            )

        cache_path, cache_meta_path = self._cache_paths(cache_key=cache_key, normalize=normalize)
        if cache_path is not None and cache_path.exists():
            loaded = np.load(cache_path)
            meta = EmbeddingMeta(backend="cache", model_name=self.model_name, dim=int(loaded.shape[1]))
            if cache_meta_path is not None and cache_meta_path.exists():
                try:
                    payload = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                    meta = EmbeddingMeta(
                        backend=str(payload.get("backend", "cache")),
                        model_name=str(payload.get("model_name", self.model_name)),
                        dim=int(payload.get("dim", loaded.shape[1])),
                    )
                except Exception:
                    pass
            return loaded.astype(np.float32), meta

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
                meta = EmbeddingMeta(backend="sentence-transformers", model_name=self.model_name, dim=int(emb.shape[1]))
                self._write_cache(cache_path, cache_meta_path, emb, meta)
                return emb, meta
            except Exception:
                pass

        emb = _hashing_vectorizer_embeddings(text_list, dim=256)
        if normalize:
            emb = _l2_normalize(emb)
        meta = EmbeddingMeta(backend="hashing-vectorizer", model_name="sklearn.HashingVectorizer", dim=int(emb.shape[1]))
        self._write_cache(cache_path, cache_meta_path, emb, meta)
        return emb, meta

    def _cache_paths(self, *, cache_key: Optional[str], normalize: bool) -> tuple[Optional[Path], Optional[Path]]:
        if not cache_key:
            return None, None
        safe_model = "".join(ch if ch.isalnum() else "_" for ch in self.model_name.lower()).strip("_")
        norm_tag = "norm" if normalize else "raw"
        stem = f"{safe_model}__{cache_key}__{norm_tag}"
        return self.cache_dir / f"{stem}.npy", self.cache_dir / f"{stem}.meta.json"

    def _write_cache(
        self,
        cache_path: Optional[Path],
        cache_meta_path: Optional[Path],
        embeddings: np.ndarray,
        meta: EmbeddingMeta,
    ) -> None:
        if cache_path is None or cache_meta_path is None:
            return
        try:
            np.save(cache_path, embeddings.astype(np.float32))
            cache_meta_path.write_text(
                json.dumps({"backend": meta.backend, "model_name": meta.model_name, "dim": int(meta.dim)}),
                encoding="utf-8",
            )
        except Exception:
            return


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _hashing_vectorizer_embeddings(texts: List[str], *, dim: int) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None)
    x = hv.transform(texts)
    return x.astype(np.float32).toarray()
