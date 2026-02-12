from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def run_label_check(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    embeddings: Optional[np.ndarray] = None,
    imbalance_warn_ratio: float = 10.0,
    min_class_fraction_warn: float = 0.05,
    suspect_margin: float = 0.05,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if label_col not in df.columns:
        issues.append(
            {
                "severity": "warning",
                "code": "missing_label_column",
                "message": f"Label column '{label_col}' not found; label-based checks skipped.",
                "recommendation": "Provide a label column to enable class imbalance and label-noise analysis.",
            }
        )
        return {
            "metrics": {
                "rows": int(len(df)),
                "labeled_rows": 0,
                "num_classes": 0,
                "label_distribution": {},
            },
            "issues": issues,
            "row_flags": row_flags,
        }

    labels = df[label_col].fillna("").astype(str)
    valid_mask = labels.str.len() > 0
    labels = labels.where(valid_mask, other=None)

    counts = labels.dropna().value_counts()
    total = int(counts.sum())
    label_dist = {k: int(v) for k, v in counts.to_dict().items()}

    imbalance_ratio = None
    min_fraction = None
    if len(counts) >= 2:
        min_count = int(counts.min())
        max_count = int(counts.max())
        imbalance_ratio = float(max_count / max(1, min_count))
        min_fraction = float(min_count / max(1, total))

        if imbalance_ratio >= imbalance_warn_ratio or (min_fraction is not None and min_fraction <= min_class_fraction_warn):
            issues.append(
                {
                    "severity": "warning",
                    "code": "class_imbalance",
                    "message": f"Class imbalance detected (ratio={imbalance_ratio:.2f}, min_fraction={min_fraction:.3f}).",
                    "recommendation": "Consider re-sampling, collecting more data for minority classes, or using class-weighted training.",
                }
            )

    outliers: List[Dict[str, object]] = []
    suspects: List[Dict[str, object]] = []

    if embeddings is not None and len(df) == embeddings.shape[0] and len(counts) >= 2:
        emb = _ensure_normalized(embeddings)
        row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()

        centroids: Dict[str, np.ndarray] = {}
        for lab in counts.index.tolist():
            mask = labels == lab
            if int(mask.sum()) < 2:
                continue
            centroids[str(lab)] = _normalize_vec(emb[mask.to_numpy()].mean(axis=0))

        centroid_labels = list(centroids.keys())
        if len(centroid_labels) >= 2:
            centroid_mat = np.stack([centroids[l] for l in centroid_labels], axis=0)
            sims = emb @ centroid_mat.T
            for i in range(len(df)):
                lab = labels.iloc[i]
                if lab is None or str(lab) not in centroids:
                    continue
                own_idx = centroid_labels.index(str(lab))
                own_sim = float(sims[i, own_idx])
                best_idx = int(np.argmax(sims[i]))
                best_label = centroid_labels[best_idx]
                best_sim = float(sims[i, best_idx])

                if best_label != str(lab) and best_sim - own_sim >= suspect_margin:
                    rid = int(row_ids[i])
                    row_flags.setdefault(rid, []).append("label_suspect")
                    suspects.append(
                        {
                            "row_id": rid,
                            "label": str(lab),
                            "suggested_label": best_label,
                            "own_similarity": own_sim,
                            "best_similarity": best_sim,
                        }
                    )

            for lab in centroid_labels:
                mask = labels == lab
                if int(mask.sum()) < 10:
                    continue
                idxs = np.where(mask.to_numpy())[0]
                own_idx = centroid_labels.index(lab)
                dists = 1.0 - sims[idxs, own_idx]
                mean_dist = float(np.mean(dists))
                std_dist = float(np.std(dists))
                if std_dist <= 1e-12:
                    continue
                for local_i, dist in zip(idxs.tolist(), dists.tolist(), strict=True):
                    z_score = float((float(dist) - mean_dist) / std_dist)
                    if z_score >= 2.0:
                        rid = int(row_ids[local_i])
                        row_flags.setdefault(rid, []).append("label_outlier")
                        outliers.append(
                            {
                                "row_id": rid,
                                "label": lab,
                                "distance_to_centroid": float(dist),
                                "z_score": z_score,
                            }
                        )

            if suspects:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "possible_mislabeled_samples",
                        "message": f"Found {len(suspects)} samples closer to another class centroid.",
                        "recommendation": "Manually review suspected samples; fix labels or remove ambiguous data.",
                    }
                )
            if outliers:
                issues.append(
                    {
                        "severity": "info",
                        "code": "label_outliers",
                        "message": f"Found {len(outliers)} embedding outliers within classes.",
                        "recommendation": "Review outliers for noise, edge-cases, or inconsistent labeling.",
                    }
                )

    metrics = {
        "rows": int(len(df)),
        "labeled_rows": int(total),
        "num_classes": int(len(counts)),
        "label_distribution": label_dist,
        "imbalance_ratio": imbalance_ratio,
        "min_class_fraction": min_fraction,
        "suspected_mislabeled_samples": int(len(suspects)),
        "outlier_samples": int(len(outliers)),
        "suspects_sample": suspects[:50],
        "outliers_sample": outliers[:50],
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, eps)


def _ensure_normalized(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)
