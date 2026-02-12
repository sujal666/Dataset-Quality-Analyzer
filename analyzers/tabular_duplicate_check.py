from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def run_tabular_duplicate_check(
    df: pd.DataFrame,
    *,
    key_columns: Optional[List[str]] = None,
) -> Dict[str, object]:
    issues: List[Dict[str, object]] = []
    row_flags: Dict[int, List[str]] = {}

    if len(df) == 0:
        return {"metrics": {"rows": 0}, "issues": issues, "row_flags": row_flags}

    row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()
    source_columns = [c for c in df.columns if c != "_row_id"]
    ignored_artifact_columns = [c for c in source_columns if _is_artifact_column(c)]
    feature_columns = [c for c in source_columns if c not in ignored_artifact_columns]

    exact_mask = df.duplicated(subset=feature_columns, keep=False) if feature_columns else pd.Series([False] * len(df))
    exact_row_ids = [int(row_ids[i]) for i in range(len(df)) if bool(exact_mask.iloc[i])]
    exact_groups = _duplicate_groups(df, row_ids, subset=feature_columns) if feature_columns else []

    for rid in exact_row_ids:
        row_flags.setdefault(rid, []).append("duplicate_exact")

    chosen_keys = [c for c in (key_columns or _choose_key_columns(df)) if c in feature_columns]
    key_dup_rows: List[int] = []
    key_groups: List[List[int]] = []
    if chosen_keys and len(chosen_keys) >= 2:
        key_mask = df.duplicated(subset=chosen_keys, keep=False)
        key_dup_rows = [int(row_ids[i]) for i in range(len(df)) if bool(key_mask.iloc[i])]
        key_groups = _duplicate_groups(df, row_ids, subset=chosen_keys)

        for rid in key_dup_rows:
            if rid not in exact_row_ids:
                row_flags.setdefault(rid, []).append("duplicate_key")

    if exact_groups:
        issues.append(
            {
                "severity": "warning",
                "code": "exact_duplicates",
                "message": f"Found {len(exact_groups)} exact-duplicate groups ({len(exact_row_ids)} rows involved).",
                "recommendation": "Remove exact duplicate rows to avoid biased training.",
            }
        )

    if key_groups:
        issues.append(
            {
                "severity": "warning",
                "code": "tabular_key_duplicates",
                "message": f"Found {len(key_groups)} duplicate groups on key columns {chosen_keys}.",
                "recommendation": "Review repeated listings on key business fields and keep only one canonical row where appropriate.",
            }
        )

    metrics = {
        "rows": int(len(df)),
        "mode": "tabular",
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_rows": int(len(exact_row_ids)),
        "duplicate_rate_percentage": round((len(exact_row_ids) / max(1, len(df))) * 100.0, 2),
        "semantic_duplicate_clusters": 0,
        "key_columns_used": chosen_keys,
        "ignored_artifact_columns": ignored_artifact_columns,
        "key_duplicate_groups": int(len(key_groups)),
        "key_duplicate_rows": int(len(key_dup_rows)),
        "key_duplicate_rate_percentage": round((len(key_dup_rows) / max(1, len(df))) * 100.0, 2),
        "exact_groups_sample": [group[:10] for group in exact_groups[:20]],
        "key_groups_sample": [group[:10] for group in key_groups[:20]],
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _duplicate_groups(df: pd.DataFrame, row_ids: List[int], *, subset: List[str]) -> List[List[int]]:
    if not subset:
        return []

    grouped = (
        df.assign(__row_id=row_ids)
        .groupby(subset, dropna=False, sort=False)["__row_id"]
        .apply(list)
        .tolist()
    )
    return [sorted(int(r) for r in g) for g in grouped if len(g) > 1]


def _choose_key_columns(df: pd.DataFrame) -> List[str]:
    columns = [c for c in df.columns if c != "_row_id" and not _is_artifact_column(c)]
    norm_map = {_norm(c): c for c in columns}

    city = _find_col(norm_map, {"city"})
    locality = _find_col(norm_map, {"locality", "location", "area_name"})
    area = _find_col(norm_map, {"area", "carpet_area", "super_area", "sqft", "area_sqft"})
    rent = _find_col(norm_map, {"rent", "price", "monthly_rent"})

    keys = [c for c in [city, locality, area, rent] if c is not None]
    if len(keys) >= 2:
        return keys

    if len(columns) >= 2:
        rows = max(1, len(df))
        candidate_cols = [c for c in columns if float(df[c].nunique(dropna=False)) / float(rows) < 0.9]
        if len(candidate_cols) >= 2:
            candidate_cols.sort(key=lambda c: float(df[c].nunique(dropna=False)) / float(rows))
            return candidate_cols[: min(4, len(candidate_cols))]

    return columns[: min(4, len(columns))]


def _find_col(norm_map: Dict[str, str], names: set[str]) -> Optional[str]:
    for n in names:
        if n in norm_map:
            return norm_map[n]
    for key, original in norm_map.items():
        if any(n in key for n in names):
            return original
    return None


def _norm(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).lower()).strip("_")


def _is_artifact_column(name: str) -> bool:
    lowered = str(name).strip().lower()
    if lowered.startswith("unnamed"):
        return True

    normalized = _norm(name)
    index_like = {
        "index",
        "_index",
        "__index_level_0__",
        "unnamed_0",
        "unnamed_1",
    }
    return normalized in index_like
