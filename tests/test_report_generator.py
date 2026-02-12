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
    assert isinstance(report.get("modality_explanation"), str)

    dup_rows = report["modules"]["duplicates"]["exact_duplicate_rows"]
    assert dup_rows == 2

    exact_issue = next(i for i in report["issues"] if i["code"] == "exact_duplicates")
    assert "plain_explanation" in exact_issue
    assert "why_it_matters" in exact_issue
    assert "headline" in exact_issue
    assert "what_we_found" in exact_issue
    assert "what_you_can_do" in exact_issue
    assert "technical_details" in exact_issue
    assert "severity_label" in exact_issue
    assert isinstance(exact_issue.get("examples"), list) and len(exact_issue["examples"]) > 0
    assert isinstance(report.get("score_breakdown", {}).get("cards"), list)
    assert isinstance(report.get("issue_groups", {}).get("needs_attention"), list)
    assert isinstance(report.get("good_to_go"), list)


def test_analyze_csv_uses_tabular_pipeline_for_structured_data(tmp_path: Path) -> None:
    p = tmp_path / "tabular.csv"
    pd.DataFrame(
        {
            "city": ["mumbai", "mumbai", "pune", "pune"],
            "locality": ["powai", "powai", "kothrud", "kothrud"],
            "area": [700, 700, 600, 600],
            "rent": [45000, 45000, 30000, 30000],
            "beds": [2, 2, 1, 1],
            "bathrooms": [2, 2, 1, 1],
            "text": ["2 BHK Powai", "2 BHK Powai", "1 BHK Kothrud", "1 BHK Kothrud"],
        }
    ).to_csv(p, index=False)

    report = analyze_csv(p, prefer_hf_models=False)
    assert report["meta"]["modality"]["modality"] == "tabular"
    assert report["modules"]["duplicates"]["mode"] == "tabular"
    assert report["modules"]["text_quality"]["ran"] is False
    assert "treated as tabular dataset" in report["modality_explanation"]
