from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def run_domain_check(
    df: pd.DataFrame,
    *,
    embeddings: Optional[np.ndarray] = None,
    max_k: int = 6,
    silhouette_threshold: float = 0.20,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if embeddings is None or len(df) < 10:
        return {
            "metrics": {"rows": int(len(df)), "ran": False},
            "issues": issues,
            "row_flags": row_flags,
        }

    if embeddings.shape[0] != len(df):
        issues.append(
            {
                "severity": "warning",
                "code": "embedding_mismatch",
                "message": "Embeddings do not align with dataframe rows; domain check skipped.",
                "recommendation": "Regenerate embeddings after preprocessing and ensure row order is consistent.",
            }
        )
        return {"metrics": {"rows": int(len(df)), "ran": False}, "issues": issues, "row_flags": row_flags}

    emb = _ensure_normalized(embeddings)
    n = emb.shape[0]
    max_k = int(min(max_k, max(2, n // 10)))

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k = 1
    best_score = -1.0
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(emb)
        score = float(silhouette_score(emb, labels, metric="cosine"))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    mixing_flag = bool(best_k >= 2 and best_score >= silhouette_threshold)
    if mixing_flag:
        issues.append(
            {
                "severity": "warning",
                "code": "possible_domain_mixing",
                "message": f"Embedding clusters suggest multiple domains (k={best_k}, silhouette={best_score:.2f}).",
                "recommendation": "Inspect clusters; consider splitting by domain or filtering inconsistent sources.",
            }
        )

    cluster_sizes = {}
    if best_labels is not None:
        unique, counts = np.unique(best_labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist(), strict=True)}

    metrics = {
        "rows": int(len(df)),
        "ran": True,
        "best_k": int(best_k),
        "silhouette": float(best_score),
        "mixing_flag": mixing_flag,
        "cluster_sizes": cluster_sizes,
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _ensure_normalized(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)

