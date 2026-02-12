from __future__ import annotations

import pandas as pd

from analyzers.modality_detector import detect_modality, detect_modality_details


def test_detects_tabular_modality_on_numeric_heavy_dataset() -> None:
    df = pd.DataFrame(
        {
            "city": ["mumbai", "mumbai", "pune", "pune"],
            "rent": [40000, 42000, 28000, 30000],
            "area": [750, 780, 620, 650],
            "bhk": [2, 2, 1, 1],
            "text": ["2bhk", "2bhk", "1bhk", "1bhk"],
        }
    )

    modality = detect_modality(df, text_col="text")
    details = detect_modality_details(df, text_col="text")
    assert modality == "tabular"
    assert details["numeric_ratio"] > 0.5


def test_detects_text_modality_on_long_text_dataset() -> None:
    df = pd.DataFrame(
        {
            "text": [
                "This is a long product review discussing quality and reliability in detail.",
                "Another detailed review with multiple words and sentence structure for NLP analysis.",
                "The customer explains pros, cons, and service outcomes in narrative form.",
            ],
            "label": ["pos", "neg", "pos"],
        }
    )

    modality = detect_modality(df, text_col="text")
    assert modality == "text"

