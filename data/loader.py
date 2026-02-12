from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype


@dataclass(frozen=True)
class LoadedDataset:
    df: pd.DataFrame
    text_col: str
    label_col: Optional[str]
    meta: Dict[str, Any]


_TEXT_COL_CANDIDATES = {
    "text",
    "sentence",
    "review",
    "content",
    "message",
    "body",
    "document",
    "article",
    "title",
    "description",
    "comment",
    "prompt",
    "question",
    "query",
    "tweet",
    "utterance",
    "dialogue",
}

_LABEL_COL_CANDIDATES = {"label", "labels", "class", "category", "target", "y"}


def _normalize_col_key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


def _choose_text_and_label_columns(
    df: pd.DataFrame, *, text_col: Optional[str], label_col: Optional[str]
) -> Tuple[str, Optional[str]]:
    if text_col is not None:
        if text_col not in df.columns:
            raise ValueError(f"text_col '{text_col}' not found in columns: {list(df.columns)}")
        chosen_text = text_col
    else:
        norm_map = {_normalize_col_key(c): c for c in df.columns}
        chosen_text = None
        for candidate in _TEXT_COL_CANDIDATES:
            if candidate in norm_map:
                chosen_text = norm_map[candidate]
                break
        if chosen_text is None:
            text_like_cols = [
                c
                for c in df.columns
                if is_string_dtype(df[c].dtype) or is_object_dtype(df[c].dtype) or is_categorical_dtype(df[c].dtype)
            ]
            if not text_like_cols:
                dtypes = {c: str(df[c].dtype) for c in df.columns}
                raise ValueError(
                    "No obvious text column found; provide text_col explicitly. "
                    f"Columns={list(df.columns)} dtypes={dtypes}"
                )
            chosen_text = _select_best_text_column(df, text_like_cols)

    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in columns: {list(df.columns)}")
        chosen_label: Optional[str] = label_col
    else:
        norm_map = {_normalize_col_key(c): c for c in df.columns}
        chosen_label = None
        for candidate in _LABEL_COL_CANDIDATES:
            if candidate in norm_map and norm_map[candidate] != chosen_text:
                chosen_label = norm_map[candidate]
                break

    return chosen_text, chosen_label


def _select_best_text_column(df: pd.DataFrame, cols: list[str], *, sample_rows: int = 200) -> str:
    best_col = cols[0]
    best_score = float("-inf")

    for col in cols:
        key = _normalize_col_key(col)
        name_score = 0.0
        if key in _TEXT_COL_CANDIDATES:
            name_score += 100.0
        else:
            for tok in _TEXT_COL_CANDIDATES:
                if tok in key:
                    name_score += 35.0
                    break

        s = df[col]
        sample = s.dropna()
        if len(sample) > sample_rows:
            sample = sample.head(sample_rows)

        sample_str = sample.astype(str)
        lengths = sample_str.map(lambda x: len(x.strip()))
        avg_len = float(lengths.mean()) if len(lengths) else 0.0
        whitespace_frac = float(sample_str.str.contains(r"\s", regex=True).mean()) if len(sample_str) else 0.0

        score = name_score + avg_len + 15.0 * whitespace_frac
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _rename_to_standard(df: pd.DataFrame, *, text_col: str, label_col: Optional[str]) -> LoadedDataset:
    df_out = df.copy()
    rename: Dict[str, str] = {}

    if text_col != "text":
        rename[text_col] = "text"
    if label_col is not None and label_col != "label":
        rename[label_col] = "label"

    if rename:
        df_out = df_out.rename(columns=rename)

    meta = {
        "original_columns": list(df.columns),
        "renamed_columns": rename,
        "rows": int(len(df_out)),
    }
    return LoadedDataset(df=df_out, text_col="text", label_col=("label" if label_col is not None else None), meta=meta)


def load_csv_dataset(
    csv_path: str | Path,
    *,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    encoding: Optional[str] = None,
) -> LoadedDataset:
    path = Path(csv_path)
    df = pd.read_csv(path, encoding=encoding)
    chosen_text, chosen_label = _choose_text_and_label_columns(df, text_col=text_col, label_col=label_col)
    return _rename_to_standard(df, text_col=chosen_text, label_col=chosen_label)


def load_csv_bytes_dataset(
    csv_bytes: bytes,
    *,
    filename: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    encoding: Optional[str] = None,
) -> LoadedDataset:
    df = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
    chosen_text, chosen_label = _choose_text_and_label_columns(df, text_col=text_col, label_col=label_col)
    loaded = _rename_to_standard(df, text_col=chosen_text, label_col=chosen_label)
    if filename:
        loaded.meta["filename"] = filename
    return loaded


def load_hf_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    subset: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> LoadedDataset:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)
    if max_rows is not None and max_rows > 0:
        ds = ds.select(range(min(max_rows, len(ds))))
    df = ds.to_pandas()
    chosen_text, chosen_label = _choose_text_and_label_columns(df, text_col=text_col, label_col=label_col)
    loaded = _rename_to_standard(df, text_col=chosen_text, label_col=chosen_label)
    loaded.meta.update(
        {
            "hf_dataset": dataset_name,
            "hf_subset": subset,
            "hf_split": split,
        }
    )
    return loaded
