from __future__ import annotations

import pandas as pd

from analyzers.tabular_duplicate_check import run_tabular_duplicate_check


def test_tabular_duplicate_detector_flags_exact_and_key_duplicates() -> None:
    df = pd.DataFrame(
        {
            "city": ["mumbai", "mumbai", "mumbai", "pune"],
            "locality": ["powai", "powai", "powai", "kothrud"],
            "area": [700, 700, 710, 600],
            "rent": [45000, 45000, 45000, 30000],
            "_row_id": [0, 1, 2, 3],
        }
    )

    result = run_tabular_duplicate_check(df)
    metrics = result["metrics"]
    issues = {i["code"] for i in result["issues"]}

    assert metrics["exact_duplicate_rows"] == 2
    assert metrics["key_duplicate_rows"] == 2
    assert metrics["duplicate_rate_percentage"] == 50.0
    assert metrics["key_duplicate_rate_percentage"] == 50.0
    assert "exact_duplicates" in issues
    assert "tabular_key_duplicates" in issues


def test_tabular_duplicate_detector_ignores_artifact_columns() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": [10, 11, 12, 13],
            "date": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02"],
            "time": ["18:00", "18:00", "18:00", "20:00"],
            "comp": ["A", "A", "A", "B"],
            "value": [1.0, 1.2, 1.1, 3.0],
            "_row_id": [0, 1, 2, 3],
        }
    )

    result = run_tabular_duplicate_check(df)
    metrics = result["metrics"]

    assert "Unnamed: 0" in metrics["ignored_artifact_columns"]
    assert "Unnamed: 0" not in metrics["key_columns_used"]
    assert metrics["key_duplicate_rows"] == 3
