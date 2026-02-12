from __future__ import annotations

from typing import Dict


WEIGHTS = {
    "missing": 20,
    "exact_duplicates": 20,
    "label_issues": 10,
    "toxicity": 20,
    "semantic_warnings": 10,
    "domain_mixing": 10,
    "modality_warning": 10,
}


def compute_quality_score(*, module_metrics: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    schema = module_metrics.get("schema", {})
    dup = module_metrics.get("duplicates", {})
    labels = module_metrics.get("labels", {})
    toxicity = module_metrics.get("toxicity", {})
    text_quality = module_metrics.get("text_quality", {})
    domain = module_metrics.get("domain", {})
    modality = module_metrics.get("modality", {})

    subscores = {
        "missing": _missing_score(schema),
        "exact_duplicates": _exact_duplicate_score(dup),
        "label_issues": _label_issue_score(labels),
        "toxicity": _toxicity_score(toxicity),
        "semantic_warnings": _semantic_warning_score(dup, text_quality, modality),
        "domain_mixing": _domain_mixing_score(domain),
        "modality_warning": _modality_score(modality),
    }

    overall = 0.0
    for key, weight in WEIGHTS.items():
        overall += float(subscores.get(key, 100.0)) * (float(weight) / 100.0)

    overall = max(0.0, min(100.0, overall))
    return {
        "overall": round(overall, 2),
        "verdict": _verdict(overall),
        "weights": {k: round(v / 100.0, 2) for k, v in WEIGHTS.items()},
        "subscores": {k: round(float(v), 2) for k, v in subscores.items()},
    }


def _verdict(score: float) -> str:
    if score >= 80:
        return "production_ready"
    if score >= 60:
        return "needs_cleanup"
    return "high_risk"


def _missing_score(metrics: Dict[str, object]) -> float:
    rows = int(metrics.get("rows") or 0)
    missing = metrics.get("missing_values_per_column") or {}
    empty_text = int(metrics.get("empty_text_rows") or 0)

    if rows <= 0:
        return 100.0

    if isinstance(missing, dict) and missing:
        col_count = max(1, len(missing))
        total_missing = float(sum(float(v) for v in missing.values()))
        missing_rate = total_missing / max(1.0, rows * col_count)
    else:
        missing_rate = 0.0

    empty_rate = float(empty_text / max(1, rows))
    penalty = min(100.0, (missing_rate * 220.0) + (empty_rate * 180.0))
    return max(0.0, 100.0 - penalty)


def _exact_duplicate_score(metrics: Dict[str, object]) -> float:
    rows = int(metrics.get("rows") or 0)
    if rows <= 0:
        return 100.0

    exact_rows = int(metrics.get("exact_duplicate_rows") or 0)
    key_rows = int(metrics.get("key_duplicate_rows") or 0)
    exact_rate = float(exact_rows / max(1, rows))
    key_rate = float(key_rows / max(1, rows))

    penalty = min(100.0, (exact_rate * 260.0) + (key_rate * 140.0))
    return max(0.0, 100.0 - penalty)


def _label_issue_score(metrics: Dict[str, object]) -> float:
    rows = int(metrics.get("rows") or 0)
    num_classes = int(metrics.get("num_classes") or 0)
    if rows <= 0:
        return 60.0
    if num_classes <= 0:
        return 60.0

    imbalance_ratio = metrics.get("imbalance_ratio")
    suspects = int(metrics.get("suspected_mislabeled_samples") or 0)
    outliers = int(metrics.get("outlier_samples") or 0)

    penalty = 0.0
    if isinstance(imbalance_ratio, (int, float)) and imbalance_ratio is not None:
        penalty += min(45.0, max(0.0, float(imbalance_ratio) - 1.0) * 2.8)
    penalty += min(30.0, (suspects / max(1, rows)) * 300.0)
    penalty += min(25.0, (outliers / max(1, rows)) * 200.0)
    return max(0.0, 100.0 - penalty)


def _toxicity_score(metrics: Dict[str, object]) -> float:
    ran = metrics.get("ran")
    if ran is False:
        return 100.0
    frac = metrics.get("toxic_fraction")
    if not isinstance(frac, (int, float)) or frac is None:
        return 100.0
    penalty = min(100.0, float(frac) * 500.0)
    return max(0.0, 100.0 - penalty)


def _semantic_warning_score(
    duplicate_metrics: Dict[str, object],
    text_quality_metrics: Dict[str, object],
    modality_metrics: Dict[str, object],
) -> float:
    modality = str(modality_metrics.get("modality") or "")
    if modality == "tabular":
        return 100.0

    semantic_clusters = int(duplicate_metrics.get("semantic_duplicate_clusters") or 0)
    readability = text_quality_metrics.get("flesch_reading_ease")

    penalty = min(70.0, semantic_clusters * 2.5)
    if isinstance(readability, (int, float)) and readability is not None and float(readability) < 30.0:
        penalty += min(20.0, (30.0 - float(readability)) * 0.8)
    return max(0.0, 100.0 - penalty)


def _domain_mixing_score(metrics: Dict[str, object]) -> float:
    ran = metrics.get("ran")
    if ran is False:
        return 100.0

    mixing_flag = bool(metrics.get("mixing_flag"))
    if not mixing_flag:
        return 100.0

    entropy = metrics.get("cluster_label_entropy")
    if isinstance(entropy, (int, float)) and entropy is not None:
        penalty = min(100.0, 25.0 + float(entropy) * 70.0)
        return max(0.0, 100.0 - penalty)

    silhouette = metrics.get("silhouette")
    if isinstance(silhouette, (int, float)) and silhouette is not None:
        penalty = min(100.0, 30.0 + float(silhouette) * 70.0)
        return max(0.0, 100.0 - penalty)

    return 70.0


def _modality_score(modality_metrics: Dict[str, object]) -> float:
    modality = str(modality_metrics.get("modality") or "mixed")
    confidence = float(modality_metrics.get("confidence") or 0.6)

    if modality == "mixed":
        return max(40.0, min(85.0, confidence * 100.0))
    if confidence < 0.65:
        return 85.0
    return 100.0

