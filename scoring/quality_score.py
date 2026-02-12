from __future__ import annotations

from typing import Dict, List, Optional, Tuple


WEIGHTS = {
    "duplicates": 0.25,
    "label_quality": 0.25,
    "bias_toxicity": 0.20,
    "text_quality": 0.15,
    "data_leakage": 0.15,
}


def compute_quality_score(*, module_metrics: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    dup = module_metrics.get("duplicates", {})
    lab = module_metrics.get("labels", {})
    tox = module_metrics.get("toxicity", {})
    txt = module_metrics.get("text_quality", {})
    leak = module_metrics.get("leakage", {})

    subscores = {
        "duplicates": _duplicates_score(dup),
        "label_quality": _label_quality_score(lab),
        "bias_toxicity": _toxicity_score(tox),
        "text_quality": _text_quality_score(txt),
        "data_leakage": _leakage_score(leak),
    }

    overall = 0.0
    for k, w in WEIGHTS.items():
        overall += float(subscores.get(k, 100.0)) * float(w)

    overall = max(0.0, min(100.0, overall))
    verdict = _verdict(overall)
    return {
        "overall": round(overall, 2),
        "verdict": verdict,
        "weights": WEIGHTS,
        "subscores": {k: round(float(v), 2) for k, v in subscores.items()},
    }


def _verdict(score: float) -> str:
    if score >= 80:
        return "production_ready"
    if score >= 60:
        return "needs_cleanup"
    return "high_risk"


def _duplicates_score(metrics: Dict[str, object]) -> float:
    rows = int(metrics.get("rows") or 0)
    if rows <= 0:
        return 100.0
    exact_rows = int(metrics.get("exact_duplicate_rows") or 0)
    sem_clusters = int(metrics.get("semantic_duplicate_clusters") or 0)
    dup_frac = float(exact_rows / rows)

    penalty = 0.0
    penalty += min(70.0, dup_frac * 200.0)  # 10% exact dupes -> -20 points
    penalty += min(30.0, sem_clusters * 2.0)  # many clusters -> extra penalty
    return max(0.0, 100.0 - penalty)


def _label_quality_score(metrics: Dict[str, object]) -> float:
    rows = int(metrics.get("rows") or 0)
    if rows <= 0:
        return 100.0
    ratio = metrics.get("imbalance_ratio")
    suspects = int(metrics.get("suspected_mislabeled_samples") or 0)
    outliers = int(metrics.get("outlier_samples") or 0)

    penalty = 0.0
    if isinstance(ratio, (int, float)) and ratio is not None:
        penalty += min(50.0, max(0.0, float(ratio) - 1.0) * 3.0)  # ratio 10 -> -27

    penalty += min(40.0, (suspects / max(1, rows)) * 400.0)  # 5% suspects -> -20
    penalty += min(20.0, (outliers / max(1, rows)) * 200.0)  # 10% outliers -> -20
    return max(0.0, 100.0 - penalty)


def _toxicity_score(metrics: Dict[str, object]) -> float:
    frac = metrics.get("toxic_fraction")
    if not isinstance(frac, (int, float)) or frac is None:
        return 100.0
    frac = float(frac)
    penalty = min(100.0, frac * 500.0)  # 1% toxic -> -5
    return max(0.0, 100.0 - penalty)


def _text_quality_score(metrics: Dict[str, object]) -> float:
    readability = metrics.get("flesch_reading_ease")
    repetition = metrics.get("avg_repetition_ratio")
    vocab_richness = metrics.get("vocabulary_richness")

    penalty = 0.0
    if isinstance(readability, (int, float)) and readability:
        r = float(readability)
        if r < 30:
            penalty += min(40.0, (30.0 - r) * 1.5)
        if r > 90:
            penalty += min(10.0, (r - 90.0) * 1.0)

    if isinstance(repetition, (int, float)) and repetition is not None:
        rep = float(repetition)
        if rep > 0.5:
            penalty += min(30.0, (rep - 0.5) * 60.0)

    if isinstance(vocab_richness, (int, float)) and vocab_richness is not None:
        vr = float(vocab_richness)
        if vr < 0.05:
            penalty += min(20.0, (0.05 - vr) * 400.0)

    return max(0.0, 100.0 - penalty)


def _leakage_score(metrics: Dict[str, object]) -> float:
    ran = metrics.get("ran")
    if ran is False:
        return 100.0
    risk = metrics.get("leakage_risk")
    if not isinstance(risk, (int, float)) or risk is None:
        return 100.0
    risk = float(risk)
    penalty = min(100.0, risk * 2000.0)  # 1% -> -20
    return max(0.0, 100.0 - penalty)

