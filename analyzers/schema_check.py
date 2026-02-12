from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def run_schema_check(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    short_text_chars: int = 8,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if text_col not in df.columns:
        issues.append(
            {
                "severity": "critical",
                "code": "missing_text_column",
                "message": f"Missing required text column '{text_col}'.",
                "recommendation": "Ensure your dataset includes a text column and map it to 'text'.",
            }
        )
        return {"metrics": {}, "issues": issues, "row_flags": row_flags}

    missing = df.isna().sum().to_dict()

    text_series = df[text_col].fillna("").astype(str)
    empty_mask = text_series.str.strip().str.len() == 0
    short_mask = (~empty_mask) & (text_series.str.len() < short_text_chars)

    if "_row_id" in df.columns:
        empty_row_ids = df.loc[empty_mask, "_row_id"].astype(int).tolist()
        short_row_ids = df.loc[short_mask, "_row_id"].astype(int).tolist()
    else:
        empty_row_ids = df.index[empty_mask].astype(int).tolist()
        short_row_ids = df.index[short_mask].astype(int).tolist()

    for rid in empty_row_ids:
        row_flags.setdefault(int(rid), []).append("empty_text")
    for rid in short_row_ids:
        row_flags.setdefault(int(rid), []).append("short_text")

    if empty_row_ids:
        issues.append(
            {
                "severity": "critical",
                "code": "empty_text_rows",
                "message": f"Found {len(empty_row_ids)} empty text rows.",
                "recommendation": "Drop or fix empty rows before training.",
                "count": len(empty_row_ids),
            }
        )
    if short_row_ids:
        issues.append(
            {
                "severity": "warning",
                "code": "short_text_rows",
                "message": f"Found {len(short_row_ids)} very short text rows (<{short_text_chars} chars).",
                "recommendation": "Review short rows; they may be low-signal or mislabeled.",
                "count": len(short_row_ids),
            }
        )

    if label_col is not None and label_col in df.columns:
        label_unique = df[label_col].dropna().astype(str).nunique()
        if label_unique >= max(50, int(0.5 * len(df))):
            issues.append(
                {
                    "severity": "warning",
                    "code": "high_label_cardinality",
                    "message": f"Label cardinality is high ({label_unique} unique labels across {len(df)} rows).",
                    "recommendation": "Verify that the label column is correct and not an ID or free-text field.",
                }
            )
    elif label_col is not None:
        issues.append(
            {
                "severity": "warning",
                "code": "missing_label_column",
                "message": f"Label column '{label_col}' not found; label-based checks will be skipped.",
                "recommendation": "Provide a label column to enable class imbalance and label-noise analysis.",
            }
        )

    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values_per_column": {k: int(v) for k, v in missing.items()},
        "empty_text_rows": int(len(empty_row_ids)),
        "short_text_rows": int(len(short_row_ids)),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}

