from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def run_leakage_check(
    df: pd.DataFrame,
    *,
    embeddings: Optional[np.ndarray] = None,
    split_col: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    leakage_threshold: float = 0.92,
    top_k: int = 1,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if split_col not in df.columns:
        return {"metrics": {"ran": False, "reason": f"Missing '{split_col}' column."}, "issues": issues, "row_flags": row_flags}
    if embeddings is None or embeddings.shape[0] != len(df):
        return {"metrics": {"ran": False, "reason": "Embeddings missing or misaligned."}, "issues": issues, "row_flags": row_flags}

    split = df[split_col].fillna("").astype(str)
    train_mask = split == train_value
    test_mask = split == test_value
    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        return {
            "metrics": {"ran": False, "reason": "Train/test split not found in split column."},
            "issues": issues,
            "row_flags": row_flags,
        }

    train_emb = embeddings[train_mask.to_numpy()]
    test_emb = embeddings[test_mask.to_numpy()]

    train_row_ids = (
        df.loc[train_mask, "_row_id"].astype(int).tolist()
        if "_row_id" in df.columns
        else df.index[train_mask].astype(int).tolist()
    )
    test_row_ids = (
        df.loc[test_mask, "_row_id"].astype(int).tolist()
        if "_row_id" in df.columns
        else df.index[test_mask].astype(int).tolist()
    )

    pairs = _leakage_pairs(train_emb, test_emb, train_row_ids, test_row_ids, threshold=leakage_threshold, top_k=top_k)
    leaking_test_rows = sorted({b for _, b, _ in pairs})
    for rid in leaking_test_rows:
        row_flags.setdefault(int(rid), []).append("possible_leakage")

    risk = float(len(leaking_test_rows) / max(1, len(test_row_ids)))
    if leaking_test_rows:
        issues.append(
            {
                "severity": "critical" if risk >= 0.02 else "warning",
                "code": "train_test_leakage",
                "message": f"Potential leakage: {len(leaking_test_rows)} test rows are highly similar to train rows (risk={risk:.1%}).",
                "recommendation": "Remove overlapping samples and ensure strict separation of train/test sources.",
            }
        )

    metrics = {
        "ran": True,
        "train_rows": int(len(train_row_ids)),
        "test_rows": int(len(test_row_ids)),
        "leakage_threshold": float(leakage_threshold),
        "leaking_test_rows": int(len(leaking_test_rows)),
        "leakage_risk": risk,
        "pairs_sample": [{"train_row_id": int(a), "test_row_id": int(b), "similarity": float(s)} for a, b, s in pairs[:50]],
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _leakage_pairs(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_row_ids: List[int],
    test_row_ids: List[int],
    *,
    threshold: float,
    top_k: int,
) -> List[Tuple[int, int, float]]:
    from sklearn.neighbors import NearestNeighbors

    if train_emb.ndim != 2 or test_emb.ndim != 2:
        return []

    k = max(1, min(int(top_k), train_emb.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(train_emb)
    dists, idxs = nn.kneighbors(test_emb, return_distance=True)

    pairs: List[Tuple[int, int, float]] = []
    for i in range(test_emb.shape[0]):
        for j in range(k):
            sim = float(1.0 - dists[i, j])
            if sim < threshold:
                continue
            a = int(train_row_ids[int(idxs[i, j])])
            b = int(test_row_ids[i])
            pairs.append((a, b, sim))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

