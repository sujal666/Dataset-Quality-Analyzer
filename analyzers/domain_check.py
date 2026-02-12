from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def run_domain_check(
    df: pd.DataFrame,
    *,
    embeddings: Optional[np.ndarray] = None,
    modality: str = "text",
    category_col: Optional[str] = None,
    max_k: int = 6,
    silhouette_threshold: float = 0.20,
    entropy_threshold: float = 0.65,
) -> Dict[str, object]:
    if modality == "tabular":
        return _run_tabular_domain_check(
            df=df,
            category_col=category_col,
            max_k=max_k,
            entropy_threshold=entropy_threshold,
        )
    return _run_text_domain_check(
        df=df,
        embeddings=embeddings,
        max_k=max_k,
        silhouette_threshold=silhouette_threshold,
    )


def _run_text_domain_check(
    *,
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    max_k: int,
    silhouette_threshold: float,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if embeddings is None or len(df) < 10:
        return {
            "metrics": {"rows": int(len(df)), "ran": False, "mode": "text"},
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
        return {"metrics": {"rows": int(len(df)), "ran": False, "mode": "text"}, "issues": issues, "row_flags": row_flags}

    emb = _ensure_normalized(embeddings)
    row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()
    n = emb.shape[0]
    max_k = int(min(max_k, max(2, n // 10)))

    best_k, best_score, best_labels = _best_kmeans_labels(emb, max_k=max_k, metric="cosine")
    if best_labels is None:
        return {
            "metrics": {"rows": int(len(df)), "ran": False, "mode": "text"},
            "issues": issues,
            "row_flags": row_flags,
        }

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

    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist(), strict=True)}
    cluster_samples = []
    for cluster_id in unique.tolist()[:5]:
        idxs = np.where(best_labels == cluster_id)[0][:5]
        cluster_samples.append(
            {
                "cluster": int(cluster_id),
                "row_ids": [int(row_ids[int(i)]) for i in idxs.tolist()],
            }
        )

    metrics = {
        "rows": int(len(df)),
        "ran": True,
        "mode": "text",
        "best_k": int(best_k),
        "silhouette": float(best_score),
        "mixing_flag": mixing_flag,
        "cluster_sizes": cluster_sizes,
        "cluster_samples": cluster_samples,
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _run_tabular_domain_check(
    *,
    df: pd.DataFrame,
    category_col: Optional[str],
    max_k: int,
    entropy_threshold: float,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()
    chosen_category = category_col if category_col in df.columns else _pick_category_col(df)
    numeric_cols = [c for c in df.columns if c != "_row_id" and pd.api.types.is_numeric_dtype(df[c].dtype)]

    if chosen_category is None or len(numeric_cols) < 1 or len(df) < 20:
        return {
            "metrics": {
                "rows": int(len(df)),
                "ran": False,
                "mode": "tabular",
                "reason": "Missing category or numeric features for entropy-based domain check.",
            },
            "issues": issues,
            "row_flags": row_flags,
        }

    x = df[numeric_cols].copy().fillna(df[numeric_cols].median(numeric_only=True))
    if x.isna().any().any():
        x = x.fillna(0.0)

    from sklearn.preprocessing import StandardScaler

    x_std = StandardScaler().fit_transform(x.to_numpy())
    n = x_std.shape[0]
    max_k = int(min(max_k, max(2, n // 20)))

    best_k, best_score, best_labels = _best_kmeans_labels(x_std, max_k=max_k, metric="euclidean")
    if best_labels is None:
        return {
            "metrics": {"rows": int(len(df)), "ran": False, "mode": "tabular"},
            "issues": issues,
            "row_flags": row_flags,
        }

    categories = df[chosen_category].fillna("unknown").astype(str).tolist()
    entropy = _weighted_cluster_entropy(best_labels, categories)
    mixing_flag = bool(entropy > entropy_threshold)

    if mixing_flag:
        severity = _tabular_mixing_severity(entropy=entropy, threshold=entropy_threshold)
        issues.append(
            {
                "severity": severity,
                "code": "possible_domain_mixing",
                "message": (
                    f"Tabular clusters do not align well with category '{chosen_category}' "
                    f"(cluster_label_entropy={entropy:.2f})."
                ),
                "recommendation": "Inspect mixed clusters and verify whether records from different domains are intentionally combined.",
            }
        )

    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist(), strict=True)}
    cluster_samples = []
    for cluster_id in unique.tolist()[:5]:
        idxs = np.where(best_labels == cluster_id)[0][:5]
        categories_in_cluster = [categories[int(i)] for i in idxs.tolist()][:3]
        cluster_samples.append(
            {
                "cluster": int(cluster_id),
                "row_ids": [int(row_ids[int(i)]) for i in idxs.tolist()],
                "categories": categories_in_cluster,
            }
        )

    metrics = {
        "rows": int(len(df)),
        "ran": True,
        "mode": "tabular",
        "category_column": chosen_category,
        "best_k": int(best_k),
        "silhouette": float(best_score),
        "cluster_label_entropy": float(entropy),
        "entropy_threshold": float(entropy_threshold),
        "mixing_severity": _tabular_mixing_severity(entropy=entropy, threshold=entropy_threshold) if mixing_flag else "none",
        "mixing_flag": mixing_flag,
        "cluster_sizes": cluster_sizes,
        "cluster_samples": cluster_samples,
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _best_kmeans_labels(x: np.ndarray, *, max_k: int, metric: str) -> tuple[int, float, Optional[np.ndarray]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k = 1
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(x)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(x, labels, metric=metric))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    return best_k, best_score, best_labels


def _weighted_cluster_entropy(cluster_labels: np.ndarray, categories: List[str]) -> float:
    entropy_total = 0.0
    n = len(categories)
    for cluster in np.unique(cluster_labels).tolist():
        idxs = np.where(cluster_labels == cluster)[0]
        cluster_size = len(idxs)
        if cluster_size == 0:
            continue

        counts: Dict[str, int] = {}
        for i in idxs.tolist():
            cat = categories[int(i)]
            counts[cat] = counts.get(cat, 0) + 1

        probs = [v / cluster_size for v in counts.values()]
        raw_entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        norm_entropy = raw_entropy / max(1e-12, math.log(max(2, len(counts))))
        entropy_total += (cluster_size / max(1, n)) * norm_entropy

    return float(entropy_total)


def _tabular_mixing_severity(*, entropy: float, threshold: float, extreme_margin: float = 0.20) -> str:
    extreme_threshold = max(0.85, float(threshold) + float(extreme_margin))
    return "warning" if float(entropy) >= extreme_threshold else "info"


def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    candidate_tokens = ["city", "locality", "state", "region", "category", "segment", "market", "type"]
    cols = [c for c in df.columns if c != "_row_id"]
    norm_map = {_norm(c): c for c in cols}
    for token in candidate_tokens:
        if token in norm_map:
            return norm_map[token]
    for key, col in norm_map.items():
        if any(token in key for token in candidate_tokens):
            return col
    return None


def _norm(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).lower()).strip("_")


def _ensure_normalized(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)
