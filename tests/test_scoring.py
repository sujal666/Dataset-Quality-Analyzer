from __future__ import annotations

from scoring.quality_score import compute_quality_score


def test_quality_score_uses_new_weighted_categories() -> None:
    report = compute_quality_score(
        module_metrics={
            "schema": {"rows": 100, "missing_values_per_column": {"a": 0, "b": 0}, "empty_text_rows": 0},
            "duplicates": {"rows": 100, "exact_duplicate_rows": 20, "key_duplicate_rows": 0, "semantic_duplicate_clusters": 0},
            "labels": {"rows": 100, "num_classes": 2, "imbalance_ratio": 1.2, "suspected_mislabeled_samples": 0, "outlier_samples": 0},
            "toxicity": {"ran": True, "toxic_fraction": 0.0},
            "text_quality": {"flesch_reading_ease": 60.0},
            "domain": {"ran": True, "mixing_flag": False},
            "modality": {"modality": "text", "confidence": 0.9},
        }
    )

    assert 0 <= report["overall"] <= 100
    assert "exact_duplicates" in report["subscores"]
    assert report["subscores"]["exact_duplicates"] < 100

