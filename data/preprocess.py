from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PreprocessResult:
    df: pd.DataFrame
    dropped_row_ids: List[int]
    meta: Dict[str, object]


def preprocess_dataset(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> PreprocessResult:
    if text_col not in df.columns:
        raise ValueError(f"Missing required text column '{text_col}'.")

    df_out = df.copy()
    if "_row_id" not in df_out.columns:
        df_out["_row_id"] = list(range(len(df_out)))

    df_out[text_col] = df_out[text_col].fillna("").astype(str).map(lambda s: s.strip())
    empty_mask = df_out[text_col].str.len() == 0
    dropped_row_ids = df_out.loc[empty_mask, "_row_id"].astype(int).tolist()
    df_out = df_out.loc[~empty_mask].reset_index(drop=True)

    if label_col is not None and label_col in df_out.columns:
        def _normalize_label(v: object) -> Optional[str]:
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            s = str(v).strip()
            return s if s != "" else None

        df_out[label_col] = df_out[label_col].map(_normalize_label)

    meta = {
        "rows_before": int(len(df)),
        "rows_after": int(len(df_out)),
        "dropped_empty_text": int(len(dropped_row_ids)),
    }
    return PreprocessResult(df=df_out, dropped_row_ids=dropped_row_ids, meta=meta)
