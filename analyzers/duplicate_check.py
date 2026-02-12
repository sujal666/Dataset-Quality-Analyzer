from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def run_duplicate_check(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    embeddings: Optional[np.ndarray] = None,
    semantic_threshold: float = 0.92,
    semantic_top_k: int = 5,
    max_semantic_samples: int = 2000,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if text_col not in df.columns:
        issues.append(
            {
                "severity": "critical",
                "code": "missing_text_column",
                "message": f"Missing required text column '{text_col}'.",
                "recommendation": "Ensure your dataset includes a text column and map it to 'text'.",
            }
        )
        return {"metrics": {}, "issues": issues, "row_flags": row_flags}

    text_series = df[text_col].fillna("").astype(str)
    row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()

    hashes = [_md5(t) for t in text_series.tolist()]
    groups: Dict[str, List[int]] = {}
    for rid, h in zip(row_ids, hashes, strict=True):
        groups.setdefault(h, []).append(int(rid))
    exact_groups = [ids for ids in groups.values() if len(ids) > 1]
    exact_dupe_rows = sorted({rid for grp in exact_groups for rid in grp})
    for rid in exact_dupe_rows:
        row_flags.setdefault(rid, []).append("duplicate_exact")

    semantic_groups: List[List[int]] = []
    semantic_pairs: List[Tuple[int, int, float]] = []

    if embeddings is not None and len(df) >= 2:
        idx = np.arange(len(df))
        if len(df) > max_semantic_samples:
            idx = np.linspace(0, len(df) - 1, max_semantic_samples).astype(int)
        emb = embeddings[idx]
        sub_row_ids = [row_ids[i] for i in idx.tolist()]

        pairs = _near_duplicate_pairs(emb, sub_row_ids, top_k=semantic_top_k, threshold=semantic_threshold)
        semantic_pairs = pairs[:2000]
        semantic_groups = _pairs_to_connected_components(pairs)

        for grp in semantic_groups:
            for rid in grp:
                row_flags.setdefault(int(rid), []).append("duplicate_semantic")

    if exact_groups:
        issues.append(
            {
                "severity": "warning",
                "code": "exact_duplicates",
                "message": f"Found {len(exact_groups)} exact-duplicate groups ({len(exact_dupe_rows)} rows involved).",
                "recommendation": "Deduplicate exact copies to reduce overfitting and misleading metrics.",
            }
        )
    if semantic_groups:
        issues.append(
            {
                "severity": "warning",
                "code": "semantic_duplicates",
                "message": f"Found {len(semantic_groups)} near-duplicate clusters using embeddings.",
                "recommendation": "Review and remove near-duplicates to improve dataset diversity.",
            }
        )

    metrics = {
        "rows": int(len(df)),
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_rows": int(len(exact_dupe_rows)),
        "semantic_duplicate_clusters": int(len(semantic_groups)),
        "semantic_threshold": float(semantic_threshold),
        "semantic_top_k": int(semantic_top_k),
        "semantic_pairs_sample": [
            {"row_id_a": int(a), "row_id_b": int(b), "similarity": float(s)} for a, b, s in semantic_pairs[:50]
        ],
        "exact_groups_sample": [grp[:10] for grp in exact_groups[:20]],
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _near_duplicate_pairs(
    embeddings: np.ndarray,
    row_ids: List[int],
    *,
    top_k: int,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    from sklearn.neighbors import NearestNeighbors

    if embeddings.ndim != 2 or embeddings.shape[0] != len(row_ids):
        return []

    k = max(2, min(top_k + 1, embeddings.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)

    pairs: List[Tuple[int, int, float]] = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            nbr = int(indices[i, j])
            sim = float(1.0 - distances[i, j])
            if sim < threshold:
                continue
            a = int(row_ids[i])
            b = int(row_ids[nbr])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            pairs.append((a, b, sim))

    pairs.sort(key=lambda t: t[2], reverse=True)
    dedup: Dict[Tuple[int, int], float] = {}
    for a, b, s in pairs:
        dedup[(a, b)] = max(s, dedup.get((a, b), 0.0))
    return [(a, b, s) for (a, b), s in dedup.items()]


def _pairs_to_connected_components(pairs: Iterable[Tuple[int, int, float]]) -> List[List[int]]:
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _ in pairs:
        union(int(a), int(b))

    groups: Dict[int, List[int]] = {}
    for x in list(parent.keys()):
        groups.setdefault(find(x), []).append(x)

    return [sorted(g) for g in groups.values() if len(g) > 1]

