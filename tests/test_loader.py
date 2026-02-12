from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.loader import load_csv_dataset


def test_load_csv_normalizes_text_and_label_columns(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    pd.DataFrame(
        {
            "Text": ["hello", "world"],
            "Category": ["a", "b"],
        }
    ).to_csv(p, index=False)

    loaded = load_csv_dataset(p)
    assert "text" in loaded.df.columns
    assert "label" in loaded.df.columns
    assert loaded.text_col == "text"
    assert loaded.label_col == "label"

