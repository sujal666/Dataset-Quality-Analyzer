from __future__ import annotations

from pathlib import Path

import pandas as pd

from reports.report_generator import analyze_csv


def test_analyze_csv_generates_report(tmp_path: Path) -> None:
    p = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "text": ["a", "a", "b"],
            "label": ["x", "x", "y"],
        }
    ).to_csv(p, index=False)

    report = analyze_csv(p, prefer_hf_models=False)
    assert report["report_version"] == 1
    assert 0 <= report["score"]["overall"] <= 100
    assert report["meta"]["rows"] == 3

    dup_rows = report["modules"]["duplicates"]["exact_duplicate_rows"]
    assert dup_rows == 2

