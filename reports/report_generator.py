from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzers import (
    run_domain_check,
    run_duplicate_check,
    run_label_check,
    run_leakage_check,
    run_schema_check,
    run_text_quality_check,
    run_toxicity_check,
)
from data.loader import load_csv_dataset, load_hf_dataset
from data.preprocess import preprocess_dataset
from embeddings.embedder import TextEmbedder
from scoring.quality_score import compute_quality_score


def generate_report(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    prefer_hf_models: bool = True,
) -> Dict[str, Any]:
    schema = run_schema_check(df, text_col=text_col, label_col=label_col)
    text_quality = run_text_quality_check(df, text_col=text_col)

    embedder = TextEmbedder(prefer_hf=prefer_hf_models)
    embeddings, emb_meta = embedder.embed_texts(df[text_col].fillna("").astype(str).tolist())

    duplicates = run_duplicate_check(df, text_col=text_col, embeddings=embeddings)
    labels = run_label_check(df, label_col=(label_col or "label"), embeddings=embeddings)
    toxicity = run_toxicity_check(
        df,
        text_col=text_col,
        label_col=label_col,
        prefer_hf=prefer_hf_models,
    )
    domain = run_domain_check(df, embeddings=embeddings)
    leakage = run_leakage_check(df, embeddings=embeddings)

    module_outputs = {
        "schema": schema,
        "text_quality": text_quality,
        "duplicates": duplicates,
        "labels": labels,
        "toxicity": toxicity,
        "domain": domain,
        "leakage": leakage,
    }

    score = compute_quality_score(
        module_metrics={
            "duplicates": duplicates.get("metrics", {}),
            "labels": labels.get("metrics", {}),
            "toxicity": toxicity.get("metrics", {}),
            "text_quality": text_quality.get("metrics", {}),
            "leakage": leakage.get("metrics", {}),
        }
    )

    issues = _collect_issues(module_outputs)
    row_flags = _merge_row_flags([o.get("row_flags", {}) for o in module_outputs.values()])
    row_flag_list = _row_flags_to_list(row_flags)
    recommendations = _collect_recommendations(issues)

    summary = {
        "quality_score": score["overall"],
        "verdict": score["verdict"],
        "critical_issues": int(sum(1 for i in issues if i.get("severity") == "critical")),
        "warnings": int(sum(1 for i in issues if i.get("severity") == "warning")),
        "info": int(sum(1 for i in issues if i.get("severity") == "info")),
        "flagged_rows": int(len(row_flag_list)),
    }

    return {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "text_col": text_col,
            "label_col": label_col,
            "embedding": {
                "backend": emb_meta.backend,
                "model_name": emb_meta.model_name,
                "dim": emb_meta.dim,
            },
        },
        "summary": summary,
        "score": score,
        "issues": issues,
        "recommendations": recommendations,
        "row_flags": row_flag_list,
        "modules": {k: v.get("metrics", {}) for k, v in module_outputs.items()},
    }


def analyze_csv(
    csv_path: str | Path,
    *,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    prefer_hf_models: bool = True,
) -> Dict[str, Any]:
    loaded = load_csv_dataset(csv_path, text_col=text_col, label_col=label_col)
    pre = preprocess_dataset(loaded.df, text_col=loaded.text_col, label_col=loaded.label_col)
    report = generate_report(pre.df, text_col=loaded.text_col, label_col=loaded.label_col, prefer_hf_models=prefer_hf_models)
    report["meta"].update({"input": {"type": "csv", "path": str(Path(csv_path))}, "loader": loaded.meta, "preprocess": pre.meta})
    return report


def analyze_hf(
    dataset_name: str,
    *,
    split: str = "train",
    subset: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    max_rows: Optional[int] = None,
    prefer_hf_models: bool = True,
) -> Dict[str, Any]:
    loaded = load_hf_dataset(
        dataset_name,
        split=split,
        subset=subset,
        text_col=text_col,
        label_col=label_col,
        max_rows=max_rows,
    )
    pre = preprocess_dataset(loaded.df, text_col=loaded.text_col, label_col=loaded.label_col)
    report = generate_report(pre.df, text_col=loaded.text_col, label_col=loaded.label_col, prefer_hf_models=prefer_hf_models)
    report["meta"].update({"input": {"type": "huggingface", "dataset": dataset_name, "subset": subset, "split": split}, "loader": loaded.meta, "preprocess": pre.meta})
    return report


def _collect_issues(module_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for module_name, payload in module_outputs.items():
        for issue in payload.get("issues", []):
            item = dict(issue)
            item.setdefault("module", module_name)
            out.append(item)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    out.sort(key=lambda i: (severity_order.get(str(i.get("severity")), 99), str(i.get("module")), str(i.get("code"))))
    return out


def _collect_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    recs: List[str] = []
    for i in issues:
        rec = i.get("recommendation")
        if not rec or not isinstance(rec, str):
            continue
        if rec in seen:
            continue
        seen.add(rec)
        recs.append(rec)
    return recs


def _merge_row_flags(row_flags_list: List[Dict[int, List[str]]]) -> Dict[int, List[str]]:
    merged: Dict[int, set[str]] = {}
    for rf in row_flags_list:
        for rid, flags in rf.items():
            merged.setdefault(int(rid), set()).update(str(f) for f in flags)
    return {rid: sorted(flags) for rid, flags in merged.items()}


def _row_flags_to_list(row_flags: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    critical_flags = {"empty_text", "toxic_text", "possible_leakage"}
    rows = []
    for rid, flags in sorted(row_flags.items(), key=lambda t: int(t[0])):
        severity = "warning"
        if any(f in critical_flags for f in flags):
            severity = "critical"
        rows.append({"row_id": int(rid), "flags": flags, "severity": severity})
    return rows


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset Quality Analyzer - JSON report generator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=str, help="Path to a CSV file")
    g.add_argument("--hf", type=str, help="Hugging Face dataset name, e.g. 'imdb'")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default=None)
    p.add_argument("--text-col", type=str, default=None)
    p.add_argument("--label-col", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out", type=str, default="reports/report.json")
    p.add_argument("--no-hf-models", action="store_true", help="Disable HF models (use lightweight fallbacks).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prefer = not bool(args.no_hf_models)
    if args.csv:
        report = analyze_csv(args.csv, text_col=args.text_col, label_col=args.label_col, prefer_hf_models=prefer)
    else:
        report = analyze_hf(
            args.hf,
            split=args.split,
            subset=args.subset,
            text_col=args.text_col,
            label_col=args.label_col,
            max_rows=args.max_rows,
            prefer_hf_models=prefer,
        )
    _write_json(args.out, report)
    print(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()

