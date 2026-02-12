from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype


def detect_modality(df: pd.DataFrame, *, text_col: Optional[str] = "text") -> str:
    return str(detect_modality_details(df, text_col=text_col).get("modality", "mixed"))


def detect_modality_details(df: pd.DataFrame, *, text_col: Optional[str] = "text") -> Dict[str, Any]:
    feature_cols = [c for c in df.columns if c != "_row_id"]
    total_cols = max(1, len(feature_cols))

    numeric_cols = [c for c in feature_cols if is_numeric_dtype(df[c].dtype)]
    numeric_ratio = float(len(numeric_cols) / total_cols)

    string_like_cols = [
        c
        for c in feature_cols
        if is_string_dtype(df[c].dtype) or is_object_dtype(df[c].dtype) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]

    dominant_text_col = _choose_dominant_text_column(df, string_like_cols, preferred=text_col)
    dominant_avg_len = 0.0
    dominant_space_ratio = 0.0
    if dominant_text_col is not None:
        series = df[dominant_text_col].fillna("").astype(str)
        if len(series) > 0:
            lengths = series.map(lambda s: len(s.strip()))
            dominant_avg_len = float(lengths.mean())
            dominant_space_ratio = float(series.str.contains(r"\s", regex=True).mean())

    has_dominant_text = bool(dominant_text_col is not None and dominant_avg_len >= 30 and dominant_space_ratio >= 0.4)
    is_wide_structured_table = bool(total_cols >= 6 and numeric_ratio >= 0.5 and len(string_like_cols) >= 3)

    if is_wide_structured_table:
        modality = "tabular"
    elif numeric_ratio > 0.5 and dominant_avg_len < 30:
        modality = "tabular"
    elif numeric_ratio >= 0.7:
        modality = "tabular"
    elif has_dominant_text and numeric_ratio < 0.45:
        modality = "text"
    else:
        modality = "mixed"

    confidence = _confidence(modality, numeric_ratio, dominant_avg_len, has_dominant_text)
    return {
        "modality": modality,
        "confidence": confidence,
        "numeric_ratio": numeric_ratio,
        "numeric_columns": numeric_cols,
        "string_like_columns": string_like_cols,
        "dominant_text_column": dominant_text_col,
        "dominant_text_avg_length": dominant_avg_len,
        "dominant_text_space_ratio": dominant_space_ratio,
        "rules_applied": {
            "avg_string_len_tabular_cutoff": 30,
            "numeric_ratio_tabular_cutoff": 0.50,
            "numeric_ratio_hard_tabular_cutoff": 0.70,
            "wide_structured_table_cutoff_cols": 6,
        },
    }


def _choose_dominant_text_column(
    df: pd.DataFrame, string_cols: list[str], *, preferred: Optional[str]
) -> Optional[str]:
    if preferred and preferred in string_cols:
        return preferred
    if not string_cols:
        return None

    best_col = string_cols[0]
    best_score = float("-inf")
    for col in string_cols:
        series = df[col].fillna("").astype(str)
        if len(series) == 0:
            continue
        lengths = series.map(lambda s: len(s.strip()))
        avg_len = float(lengths.mean())
        space_ratio = float(series.str.contains(r"\s", regex=True).mean())
        nonempty_ratio = float((lengths > 0).mean())
        score = avg_len + (20.0 * space_ratio) + (10.0 * nonempty_ratio)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _confidence(modality: str, numeric_ratio: float, avg_len: float, has_dominant_text: bool) -> float:
    if modality == "tabular":
        conf = 0.55 + min(0.35, max(0.0, numeric_ratio - 0.5))
        if avg_len < 15:
            conf += 0.08
        return round(min(conf, 0.98), 3)
    if modality == "text":
        conf = 0.55
        if has_dominant_text:
            conf += 0.25
        if avg_len > 80:
            conf += 0.10
        return round(min(conf, 0.98), 3)
    return 0.6
