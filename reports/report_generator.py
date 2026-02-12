from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzers import (
    detect_modality_details,
    run_domain_check,
    run_duplicate_check,
    run_label_check,
    run_leakage_check,
    run_schema_check,
    run_tabular_duplicate_check,
    run_text_quality_check,
    run_toxicity_check,
)
from data.loader import load_csv_dataset, load_hf_dataset
from data.preprocess import preprocess_dataset
from embeddings.embedder import EmbeddingMeta, TextEmbedder
from scoring.quality_score import compute_quality_score

SEVERITY_LABELS = {
    "critical": "High risk",
    "warning": "Needs attention",
    "info": "FYI",
}

SEVERITY_GROUPS = {
    "critical": "needs_attention",
    "warning": "needs_attention",
    "info": "optional_improvements",
}


def generate_report(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    prefer_hf_models: bool = True,
) -> Dict[str, Any]:
    schema = run_schema_check(df, text_col=text_col, label_col=label_col)
    modality_info = detect_modality_details(df, text_col=text_col)
    modality = str(modality_info.get("modality", "mixed"))

    modality_issues: List[Dict[str, Any]] = []
    if modality == "mixed":
        modality_issues.append(
            {
                "severity": "warning",
                "code": "modality_ambiguous",
                "message": "Dataset modality is mixed; both text and tabular patterns were detected.",
                "recommendation": "Review column mapping and run a modality-specific analysis profile for more precise checks.",
            }
        )

    embeddings: Optional[np.ndarray] = None
    emb_meta = EmbeddingMeta(backend="disabled", model_name="none", dim=0)

    if modality in {"text", "mixed"} and text_col in df.columns:
        embedder = TextEmbedder(prefer_hf=prefer_hf_models)
        signature = _dataset_signature(df, text_col=text_col, label_col=label_col)
        embeddings, emb_meta = embedder.embed_texts(
            df[text_col].fillna("").astype(str).tolist(),
            cache_key=signature,
        )

    if modality == "tabular":
        duplicates = run_tabular_duplicate_check(df)
        text_quality = _skipped_module("Text quality checks are disabled for tabular modality.")
        toxicity = _skipped_module("Toxicity checks are disabled for tabular modality.")
    else:
        duplicates = run_duplicate_check(df, text_col=text_col, embeddings=embeddings)
        text_quality = run_text_quality_check(df, text_col=text_col)
        toxicity = run_toxicity_check(
            df,
            text_col=text_col,
            label_col=label_col,
            prefer_hf=prefer_hf_models,
        )

    labels = run_label_check(df, label_col=(label_col or "label"), embeddings=embeddings)
    domain = run_domain_check(df, embeddings=embeddings, modality=modality)
    leakage = run_leakage_check(df, embeddings=embeddings) if embeddings is not None else _skipped_module(
        "Leakage check requires text embeddings and was skipped."
    )

    module_outputs = {
        "modality": {"metrics": modality_info, "issues": modality_issues, "row_flags": {}},
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
            "schema": schema.get("metrics", {}),
            "duplicates": duplicates.get("metrics", {}),
            "labels": labels.get("metrics", {}),
            "toxicity": toxicity.get("metrics", {}),
            "text_quality": text_quality.get("metrics", {}),
            "domain": domain.get("metrics", {}),
            "modality": modality_info,
        }
    )

    row_flags = _merge_row_flags([o.get("row_flags", {}) for o in module_outputs.values()])
    issues = _collect_issues(module_outputs)
    issues = _enrich_issues(
        issues=issues,
        module_outputs=module_outputs,
        df=df,
        text_col=text_col,
        label_col=label_col,
        row_flags=row_flags,
    )
    issues = _add_issue_display_fields(issues)
    row_flag_list = _row_flags_to_list(row_flags)
    recommendations = _collect_recommendations(issues)
    issue_groups = _group_issues(issues)
    good_to_go = _build_good_to_go(module_outputs)
    score_breakdown = _build_score_breakdown(
        score=score,
        module_outputs=module_outputs,
        has_label_column=bool(label_col and label_col in df.columns),
    )

    summary = {
        "quality_score": score["overall"],
        "verdict": score["verdict"],
        "critical_issues": int(sum(1 for i in issues if i.get("severity") == "critical")),
        "warnings": int(sum(1 for i in issues if i.get("severity") == "warning")),
        "info": int(sum(1 for i in issues if i.get("severity") == "info")),
        "status_counts": {
            "high_risk": int(sum(1 for i in issues if i.get("severity") == "critical")),
            "needs_attention": int(sum(1 for i in issues if i.get("severity") == "warning")),
            "fyi": int(sum(1 for i in issues if i.get("severity") == "info")),
        },
        "flagged_rows": int(len(row_flag_list)),
    }

    modality_explanation = _build_modality_explanation(modality_info)

    return {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "text_col": text_col,
            "label_col": label_col,
            "modality": modality_info,
            "embedding": {
                "backend": emb_meta.backend,
                "model_name": emb_meta.model_name,
                "dim": emb_meta.dim,
            },
        },
        "modality_explanation": modality_explanation,
        "summary": summary,
        "score": score,
        "score_breakdown": score_breakdown,
        "issues": issues,
        "issue_groups": issue_groups,
        "good_to_go": good_to_go,
        "recommendations": recommendations,
        "row_flags": row_flag_list,
        "modules": {k: v.get("metrics", {}) for k, v in module_outputs.items()},
    }


def _skipped_module(reason: str) -> Dict[str, Any]:
    return {"metrics": {"ran": False, "reason": reason}, "issues": [], "row_flags": {}}


def _dataset_signature(df: pd.DataFrame, *, text_col: str, label_col: Optional[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(len(df)).encode("utf-8"))
    hasher.update("|".join(str(c) for c in df.columns).encode("utf-8"))
    if text_col in df.columns:
        for value in df[text_col].fillna("").astype(str).head(5000).tolist():
            hasher.update(value.encode("utf-8", errors="ignore"))
            hasher.update(b"\n")
    if label_col and label_col in df.columns:
        for value in df[label_col].fillna("").astype(str).head(5000).tolist():
            hasher.update(value.encode("utf-8", errors="ignore"))
            hasher.update(b"|")
    return hasher.hexdigest()[:24]


def _build_modality_explanation(modality_info: Dict[str, Any]) -> str:
    modality = str(modality_info.get("modality", "mixed"))
    numeric_ratio = float(modality_info.get("numeric_ratio", 0.0))
    dominant_col = modality_info.get("dominant_text_column")
    avg_len = float(modality_info.get("dominant_text_avg_length", 0.0))

    numeric_pct = round(numeric_ratio * 100.0, 1)
    if modality == "tabular":
        return (
            f"{numeric_pct}% numeric columns and dominant text column '{dominant_col}' with average length "
            f"{avg_len:.1f} detected; treated as tabular dataset."
        )
    if modality == "text":
        return (
            f"Dominant text column '{dominant_col}' with average length {avg_len:.1f} and "
            f"{numeric_pct}% numeric columns detected; treated as text dataset."
        )
    return (
        f"{numeric_pct}% numeric columns with mixed text signals in '{dominant_col}' "
        f"(avg length {avg_len:.1f}); treated as mixed dataset."
    )


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


def _enrich_issues(
    *,
    issues: List[Dict[str, Any]],
    module_outputs: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str],
    row_flags: Dict[int, List[str]],
) -> List[Dict[str, Any]]:
    duplicate_metrics = module_outputs.get("duplicates", {}).get("metrics", {})
    label_metrics = module_outputs.get("labels", {}).get("metrics", {})
    text_quality_metrics = module_outputs.get("text_quality", {}).get("metrics", {})
    domain_metrics = module_outputs.get("domain", {}).get("metrics", {})
    leakage_metrics = module_outputs.get("leakage", {}).get("metrics", {})
    schema_metrics = module_outputs.get("schema", {}).get("metrics", {})

    row_texts, row_labels = _build_row_maps(df, text_col=text_col, label_col=label_col)
    columns = [str(c) for c in schema_metrics.get("columns", list(df.columns))]

    enriched: List[Dict[str, Any]] = []
    for issue in issues:
        item = dict(issue)
        code = str(item.get("code") or "")

        headline = ""
        plain = ""
        why = ""
        action = str(item.get("recommendation") or "").strip()
        examples: List[Dict[str, Any]] = []
        technical_details: Dict[str, Any] = {}

        if code == "exact_duplicates":
            duplicate_rows = int(duplicate_metrics.get("exact_duplicate_rows") or 0)
            duplicate_groups = int(duplicate_metrics.get("exact_duplicate_groups") or 0)
            duplicate_rate = float(duplicate_metrics.get("duplicate_rate_percentage") or 0.0)
            if str(duplicate_metrics.get("mode")) == "tabular":
                headline = "Exact rows are repeated"
                plain = f"About {duplicate_rate:.2f}% of rows are exact repeats across all compared columns."
            else:
                headline = "Same text appears multiple times"
                plain = f"{duplicate_rows} rows contain exactly repeated text entries."
            why = "Repeated examples can bias model learning and make evaluation look better than it really is."
            action = "Keep one canonical copy per exact record and remove accidental duplicates before training."
            technical_details = {
                "duplicate_groups": duplicate_groups,
                "rows_involved": duplicate_rows,
                "duplicate_rate_percentage": round(duplicate_rate, 2),
            }
            groups = duplicate_metrics.get("exact_groups_sample", [])
            for group in groups[:3]:
                row_ids = [int(r) for r in group[:5]]
                details = []
                for rid in row_ids[:2]:
                    details.append(_row_detail(rid, row_texts, row_labels))
                examples.append({"title": f"Duplicate group rows {row_ids}", "details": " | ".join(details)})

        elif code == "tabular_key_duplicates":
            key_cols = [str(c) for c in duplicate_metrics.get("key_columns_used", [])]
            key_rate = float(duplicate_metrics.get("key_duplicate_rate_percentage") or 0.0)
            key_groups = int(duplicate_metrics.get("key_duplicate_groups") or 0)
            key_rows = int(duplicate_metrics.get("key_duplicate_rows") or 0)
            headline = "Many records share the same key details"
            if key_cols:
                plain = (
                    f"About {key_rate:.2f}% of rows share the same key fields "
                    f"({', '.join(key_cols)}), even when other values differ."
                )
            else:
                plain = f"About {key_rate:.2f}% of rows repeat on key business fields."
            why = "When the same entity appears many times with slight differences, metrics and model behavior can be skewed."
            action = (
                "Check repeated key combinations, keep one canonical row per entity, and merge or remove noisy duplicates."
            )
            technical_details = {
                "key_columns": key_cols,
                "duplicate_groups": key_groups,
                "rows_involved": key_rows,
                "key_duplicate_rate_percentage": round(key_rate, 2),
            }
            groups = duplicate_metrics.get("key_groups_sample", [])
            if key_cols:
                examples.append({"title": "Key columns used", "details": ", ".join(key_cols)})
            for group in groups[:3]:
                row_ids = [int(r) for r in group[:5]]
                details = []
                for rid in row_ids[:2]:
                    details.append(_row_detail(rid, row_texts, row_labels))
                examples.append({"title": f"Key-duplicate rows {row_ids}", "details": " | ".join(details)})

        elif code == "semantic_duplicates":
            semantic_clusters = int(duplicate_metrics.get("semantic_duplicate_clusters") or 0)
            threshold = float(duplicate_metrics.get("semantic_threshold") or 0.0)
            headline = "Very similar text appears repeatedly"
            plain = f"{semantic_clusters} near-duplicate text clusters were detected based on semantic similarity."
            why = "Near-duplicates reduce data diversity, so the model sees less variation during training."
            action = "Keep one representative sample from each near-duplicate cluster and remove repetitive wording."
            technical_details = {
                "semantic_duplicate_clusters": semantic_clusters,
                "similarity_threshold": round(threshold, 3),
            }
            pairs = duplicate_metrics.get("semantic_pairs_sample", [])
            for pair in pairs[:3]:
                a = int(pair.get("row_id_a", -1))
                b = int(pair.get("row_id_b", -1))
                sim = float(pair.get("similarity", 0.0))
                examples.append(
                    {
                        "title": f"Near-duplicate pair rows {a} and {b}",
                        "details": f"similarity={sim:.3f}; {_row_detail(a, row_texts, row_labels)} | {_row_detail(b, row_texts, row_labels)}",
                    }
                )

        elif code == "possible_domain_mixing":
            best_k = int(domain_metrics.get("best_k") or 0)
            silhouette = float(domain_metrics.get("silhouette") or 0.0)
            entropy = domain_metrics.get("cluster_label_entropy")
            entropy_threshold = domain_metrics.get("entropy_threshold")
            if str(domain_metrics.get("mode")) == "tabular":
                headline = "Record groups may be mixed"
                plain = "Some clusters contain mixed categories instead of cleanly grouping similar records."
                why = "Mixed clusters can hide inconsistent sources and reduce downstream model stability."
                action = (
                    "Verify whether mixed categories are intentional; if not, split the dataset by domain key and retrain."
                )
            else:
                headline = "Dataset may combine multiple content domains"
                plain = "The text clusters indicate more than one writing style or topic distribution."
                why = "Domain drift inside one dataset can produce unstable model behavior."
                action = "Split by domain where possible, or balance samples so each domain is represented intentionally."
            cluster_sizes = domain_metrics.get("cluster_sizes", {})
            technical_details = {
                "best_k": best_k,
                "silhouette": round(silhouette, 3),
                "cluster_sizes": cluster_sizes,
            }
            if entropy is not None:
                technical_details["cluster_label_entropy"] = round(float(entropy), 3)
            if entropy_threshold is not None:
                technical_details["entropy_threshold"] = round(float(entropy_threshold), 3)
            if entropy is not None:
                examples.append(
                    {
                        "title": "Cluster/category alignment",
                        "details": f"cluster_label_entropy={float(entropy):.3f}, threshold={float(entropy_threshold or 0.0):.3f}",
                    }
                )
            examples.append(
                {
                    "title": "Detected cluster pattern",
                    "details": f"k={best_k}, silhouette={silhouette:.3f}, cluster_sizes={cluster_sizes}",
                }
            )
            for sample in domain_metrics.get("cluster_samples", [])[:3]:
                cluster_id = int(sample.get("cluster", -1))
                sample_rows = [int(r) for r in sample.get("row_ids", [])]
                if not sample_rows:
                    continue
                cats = sample.get("categories")
                cat_hint = f"; categories={cats}" if cats else ""
                examples.append(
                    {
                        "title": f"Cluster {cluster_id} sample rows",
                        "details": ", ".join(_row_detail(rid, row_texts, row_labels) for rid in sample_rows[:2]) + cat_hint,
                    }
                )

        elif code == "modality_ambiguous":
            modality_metrics = module_outputs.get("modality", {}).get("metrics", {})
            headline = "Dataset type is ambiguous"
            plain = "The dataset has both tabular and text-like patterns, so some checks are less confident."
            why = "When modality is mixed, thresholds are harder to tune and false positives are more likely."
            action = "Choose explicit text and label columns, then run the analyzer in the modality that matches your use case."
            technical_details = {
                "numeric_ratio": round(float(modality_metrics.get("numeric_ratio", 0.0)), 3),
                "dominant_text_column": modality_metrics.get("dominant_text_column"),
                "confidence": round(float(modality_metrics.get("confidence", 0.0)), 3),
            }
            examples.append(
                {
                    "title": "Detected modality metrics",
                    "details": (
                        f"numeric_ratio={float(modality_metrics.get('numeric_ratio', 0.0)):.3f}, "
                        f"dominant_text_column={modality_metrics.get('dominant_text_column')}"
                    ),
                }
            )

        elif code == "missing_label_column":
            headline = "No target column found"
            plain = "A label/target column was not detected, so class balance and label-quality checks were skipped."
            why = "Without a target column, the report cannot validate class imbalance or suspicious labels."
            action = (
                "Add the column you want to predict (for example `match_result`, `price_category`, or `sentiment`) "
                "and rerun analysis for deeper diagnostics."
            )
            technical_details = {"detected_columns_sample": columns[:20]}
            examples.append({"title": "Detected columns", "details": ", ".join(columns[:12])})

        elif code == "class_imbalance":
            imbalance_ratio = float(label_metrics.get("imbalance_ratio") or 0.0)
            headline = "Some classes are underrepresented"
            plain = "One or more labels have much fewer rows than the majority class."
            why = "Class imbalance often causes poor accuracy on rare classes."
            action = "Collect more minority-class examples or use class-balanced sampling/weights during training."
            technical_details = {"imbalance_ratio": round(imbalance_ratio, 3)}
            label_dist = label_metrics.get("label_distribution", {})
            top_labels = sorted(label_dist.items(), key=lambda t: int(t[1]), reverse=True)[:5]
            examples.append({"title": "Label distribution sample", "details": ", ".join(f"{k}: {v}" for k, v in top_labels)})

        elif code == "possible_mislabeled_samples":
            suspects = int(label_metrics.get("suspected_mislabeled_samples") or 0)
            headline = "Some labels may be incorrect"
            plain = f"{suspects} rows look closer to another class than their current label."
            why = "Incorrect labels teach the model conflicting patterns and reduce accuracy."
            action = "Review suspected rows, correct labels where needed, and remove ambiguous records."
            technical_details = {"suspected_mislabeled_samples": suspects}
            for sample in label_metrics.get("suspects_sample", [])[:3]:
                rid = int(sample.get("row_id", -1))
                label = str(sample.get("label", ""))
                suggested = str(sample.get("suggested_label", ""))
                examples.append(
                    {
                        "title": f"Row {rid} may be mislabeled",
                        "details": f"current={label}, suggested={suggested}; {_row_detail(rid, row_texts, row_labels)}",
                    }
                )

        elif code == "label_outliers":
            outliers = int(label_metrics.get("outlier_samples") or 0)
            headline = "Some rows are unusual for their class"
            plain = f"{outliers} rows are far from their class centroid and may be outliers."
            why = "Extreme outliers can be noise, edge cases, or incorrect labels."
            action = "Manually review these rows and decide whether to relabel, clean, or keep as intentional edge cases."
            technical_details = {"outlier_samples": outliers, "method": "class-centroid z-score"}
            for sample in label_metrics.get("outliers_sample", [])[:3]:
                rid = int(sample.get("row_id", -1))
                dist = float(sample.get("distance_to_centroid", 0.0))
                z_score = float(sample.get("z_score", 0.0))
                examples.append(
                    {
                        "title": f"Outlier row {rid}",
                        "details": f"distance_to_class_centroid={dist:.3f}, z_score={z_score:.2f}; {_row_detail(rid, row_texts, row_labels)}",
                    }
                )

        elif code == "toxic_content":
            toxic_fraction = float(module_outputs.get("toxicity", {}).get("metrics", {}).get("toxic_fraction", 0.0))
            headline = "Potentially unsafe language detected"
            plain = f"About {toxic_fraction * 100.0:.2f}% of rows were flagged for toxic content."
            why = "Toxic content can introduce safety, moderation, and compliance risks."
            action = "Review flagged rows and remove or isolate unsafe text before model training."
            technical_details = {"toxic_fraction": round(toxic_fraction, 4)}
            toxic_rows = [rid for rid, flags in row_flags.items() if "toxic_text" in flags][:3]
            for rid in toxic_rows:
                examples.append({"title": f"Toxic row {rid}", "details": _row_detail(rid, row_texts, row_labels)})

        elif code == "empty_text_rows":
            empty_rows_count = int(schema_metrics.get("empty_text_rows") or 0)
            headline = "Some rows are empty"
            plain = f"{empty_rows_count} rows have no usable text."
            why = "Empty rows add noise and reduce effective training data."
            action = "Drop empty rows or fill them with valid content before training."
            technical_details = {"empty_text_rows": empty_rows_count}
            empty_rows = [rid for rid, flags in row_flags.items() if "empty_text" in flags][:5]
            if empty_rows:
                examples.append({"title": "Empty text row IDs", "details": ", ".join(str(r) for r in empty_rows)})

        elif code == "short_text_rows":
            short_rows_count = int(schema_metrics.get("short_text_rows") or 0)
            headline = "Some rows are very short"
            plain = f"{short_rows_count} rows are very short and may not carry enough training signal."
            why = "Very short text can behave like low-information noise."
            action = "Review very short rows and remove or enrich them where possible."
            technical_details = {"short_text_rows": short_rows_count}
            short_rows = [rid for rid, flags in row_flags.items() if "short_text" in flags][:3]
            for rid in short_rows:
                examples.append({"title": f"Short row {rid}", "details": _row_detail(rid, row_texts, row_labels)})

        elif code == "low_readability":
            readability = float(text_quality_metrics.get("flesch_reading_ease") or 0.0)
            headline = "Text readability is low"
            plain = "The dataset text is hard to read or overly complex."
            why = "Low readability can reduce model quality and make annotation harder."
            action = "Normalize casing/punctuation and simplify overly complex phrasing where feasible."
            technical_details = {"flesch_reading_ease": round(readability, 2)}
            readability = float(text_quality_metrics.get("flesch_reading_ease") or 0.0)
            examples.append({"title": "Readability score", "details": f"Flesch reading ease={readability:.2f}"})

        elif code == "train_test_leakage":
            leak_pairs = int(leakage_metrics.get("leak_pair_count") or 0)
            leak_fraction = float(leakage_metrics.get("leak_fraction") or 0.0)
            headline = "Possible train-test leakage found"
            plain = f"{leak_pairs} train/test pairs are unusually similar ({leak_fraction * 100.0:.2f}% leakage risk)."
            why = "Leakage inflates test performance and hides real-world failure modes."
            action = "Remove overlap between train and test splits and rebuild splits from independent sources."
            technical_details = {"leak_pair_count": leak_pairs, "leak_fraction": round(leak_fraction, 4)}
            for pair in leakage_metrics.get("pairs_sample", [])[:3]:
                train_rid = int(pair.get("train_row_id", -1))
                test_rid = int(pair.get("test_row_id", -1))
                sim = float(pair.get("similarity", 0.0))
                examples.append(
                    {
                        "title": f"Leakage pair train={train_rid}, test={test_rid}",
                        "details": f"similarity={sim:.3f}; train: {_row_detail(train_rid, row_texts, row_labels)} | test: {_row_detail(test_rid, row_texts, row_labels)}",
                    }
                )

        elif code == "high_label_cardinality":
            headline = "Selected label column looks like an ID field"
            plain = "The label column has too many unique values for standard classification."
            why = "This usually means an ID-like column was used as label by mistake."
            action = "Map the label column to a true target field with a small, meaningful set of classes."
            technical_details = {"detected_columns_sample": columns[:20]}
            examples.append({"title": "Detected columns", "details": ", ".join(columns[:12])})

        elif code == "missing_text_column":
            headline = "Text column is missing"
            plain = "The selected text column was not found in the dataset."
            why = "Without text content, text-quality and semantic analysis cannot run."
            action = "Select an existing text column or provide `text_col` explicitly in the request."
            technical_details = {"requested_text_col": text_col, "detected_columns_sample": columns[:20]}
            examples.append({"title": "Detected columns", "details": ", ".join(columns[:12])})

        if not headline:
            headline = _friendly_code_title(code, fallback_message=str(item.get("message") or ""))
        if plain:
            item["explanation"] = plain
            item["plain_explanation"] = plain
            item["message"] = plain
        else:
            plain = str(item.get("message") or "").strip()
            item["message"] = plain
            item["explanation"] = plain
            item["plain_explanation"] = plain
        if why:
            item["why_it_matters"] = why
        else:
            why = "This can reduce data quality or model reliability if left unresolved."
            item["why_it_matters"] = why
        if not action:
            action = "Review affected rows and apply cleanup before model training."
        if not technical_details:
            technical_details = {
                "module": item.get("module"),
                "code": code,
            }
        item["headline"] = headline
        item["what_we_found"] = plain
        item["how_serious"] = why
        item["what_you_can_do"] = action
        item["technical_details"] = technical_details
        item["recommendation"] = action
        if examples:
            item["examples"] = examples[:5]

        enriched.append(item)

    return enriched


def _build_row_maps(
    df: pd.DataFrame, *, text_col: str, label_col: Optional[str]
) -> Tuple[Dict[int, str], Dict[int, Optional[str]]]:
    if "_row_id" in df.columns:
        row_ids = df["_row_id"].astype(int).tolist()
    else:
        row_ids = df.index.astype(int).tolist()

    if text_col in df.columns:
        texts = df[text_col].fillna("").astype(str).tolist()
    else:
        texts = [""] * len(df)

    labels: List[Optional[str]]
    if label_col is not None and label_col in df.columns:
        labels = [
            (str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else None)
            for v in df[label_col].tolist()
        ]
    else:
        labels = [None] * len(df)

    row_texts = {int(rid): text for rid, text in zip(row_ids, texts, strict=True)}
    row_labels = {int(rid): label for rid, label in zip(row_ids, labels, strict=True)}
    return row_texts, row_labels


def _row_detail(row_id: int, row_texts: Dict[int, str], row_labels: Dict[int, Optional[str]]) -> str:
    text = _short_text(row_texts.get(int(row_id), ""))
    label = row_labels.get(int(row_id))
    if label:
        return f'row {int(row_id)} (label="{label}") text="{text}"'
    return f'row {int(row_id)} text="{text}"'


def _short_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _friendly_code_title(code: str, *, fallback_message: str = "") -> str:
    mapping = {
        "exact_duplicates": "Exact rows are repeated",
        "tabular_key_duplicates": "Many records share the same key details",
        "semantic_duplicates": "Very similar text appears repeatedly",
        "possible_domain_mixing": "Record groups may be mixed",
        "modality_ambiguous": "Dataset type is ambiguous",
        "missing_label_column": "No target column found",
        "class_imbalance": "Some classes are underrepresented",
        "possible_mislabeled_samples": "Some labels may be incorrect",
        "label_outliers": "Some rows are unusual for their class",
        "toxic_content": "Potentially unsafe language detected",
        "empty_text_rows": "Some rows are empty",
        "short_text_rows": "Some rows are very short",
        "low_readability": "Text readability is low",
        "train_test_leakage": "Possible train-test leakage found",
        "high_label_cardinality": "Label column may be wrong",
        "missing_text_column": "Text column is missing",
    }
    if code in mapping:
        return mapping[code]
    if fallback_message:
        return fallback_message
    if not code:
        return "Dataset issue detected"
    return code.replace("_", " ").strip().title()


def _add_issue_display_fields(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for issue in issues:
        item = dict(issue)
        severity = str(item.get("severity") or "warning")
        item["severity_label"] = SEVERITY_LABELS.get(severity, "Needs attention")
        item["severity_group"] = SEVERITY_GROUPS.get(severity, "needs_attention")
        out.append(item)
    return out


def _collect_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    recs: List[str] = []
    for i in issues:
        action = i.get("what_you_can_do") or i.get("recommendation")
        if not action or not isinstance(action, str):
            continue
        action = action.strip()
        if not action:
            continue
        headline = str(i.get("headline") or "").strip()
        rec = f"{headline}: {action}" if headline else action
        if rec in seen:
            continue
        seen.add(rec)
        recs.append(rec)
    return recs


def _group_issues(issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    needs_attention = [i for i in issues if str(i.get("severity_group")) == "needs_attention"]
    optional_improvements = [i for i in issues if str(i.get("severity_group")) == "optional_improvements"]
    return {
        "needs_attention": needs_attention,
        "optional_improvements": optional_improvements,
    }


def _build_good_to_go(module_outputs: Dict[str, Dict[str, Any]]) -> List[str]:
    checks: List[str] = []
    schema = module_outputs.get("schema", {}).get("metrics", {})
    duplicates = module_outputs.get("duplicates", {}).get("metrics", {})
    toxicity = module_outputs.get("toxicity", {}).get("metrics", {})
    domain = module_outputs.get("domain", {}).get("metrics", {})

    missing = schema.get("missing_values_per_column")
    if isinstance(missing, dict) and sum(float(v) for v in missing.values()) == 0:
        checks.append("No missing values detected.")

    if int(schema.get("empty_text_rows") or 0) == 0:
        checks.append("No empty text rows detected.")

    exact_rows = int(duplicates.get("exact_duplicate_rows") or 0)
    key_rows = int(duplicates.get("key_duplicate_rows") or 0)
    semantic_clusters = int(duplicates.get("semantic_duplicate_clusters") or 0)
    if exact_rows == 0 and key_rows == 0 and semantic_clusters == 0:
        checks.append("No duplicate patterns detected.")

    if toxicity.get("ran") is True and float(toxicity.get("toxic_fraction") or 0.0) == 0.0:
        checks.append("No toxic content was flagged.")

    if domain.get("ran") is True and not bool(domain.get("mixing_flag")):
        checks.append("Domain consistency looks stable.")

    return checks[:6]


def _build_score_breakdown(
    *,
    score: Dict[str, Any],
    module_outputs: Dict[str, Dict[str, Any]],
    has_label_column: bool,
) -> Dict[str, Any]:
    subscores = score.get("subscores", {})
    structure_health = float(subscores.get("missing", 100.0))
    duplicate_health = (float(subscores.get("exact_duplicates", 100.0)) + float(subscores.get("semantic_warnings", 100.0))) / 2.0
    label_health = float(subscores.get("label_issues", 100.0)) if has_label_column else None
    safety_health = float(subscores.get("toxicity", 100.0))
    domain_health = float(subscores.get("domain_mixing", 100.0))
    modality_health = float(subscores.get("modality_warning", 100.0))

    cards = [
        {
            "key": "structure_health",
            "title": "Structure Health",
            "score": round(structure_health, 2),
            "status": _health_status(structure_health),
            "details": "Missing values and empty-row quality.",
        },
        {
            "key": "duplicate_risk",
            "title": "Duplicate Risk",
            "score": round(duplicate_health, 2),
            "status": _risk_status_from_health(duplicate_health),
            "details": "Exact and near-duplicate signal.",
        },
        {
            "key": "label_readiness",
            "title": "Label Readiness",
            "score": None if label_health is None else round(label_health, 2),
            "status": "Not applicable" if label_health is None else _health_status(label_health),
            "details": "Class balance and label noise checks.",
        },
        {
            "key": "safety_quality",
            "title": "Safety Quality",
            "score": round(safety_health, 2),
            "status": _health_status(safety_health),
            "details": "Toxicity and safety risk profile.",
        },
        {
            "key": "domain_consistency",
            "title": "Domain Consistency",
            "score": round(domain_health, 2),
            "status": _health_status(domain_health),
            "details": "Topic/category cohesion across records.",
        },
        {
            "key": "modality_fit",
            "title": "Modality Fit",
            "score": round(modality_health, 2),
            "status": _health_status(modality_health),
            "details": "Confidence that analysis mode matches dataset type.",
        },
    ]

    overall = float(score.get("overall") or 0.0)
    return {
        "overall": round(overall, 2),
        "overall_status": _health_status(overall),
        "cards": cards,
        "modality": module_outputs.get("modality", {}).get("metrics", {}).get("modality"),
    }


def _health_status(value: float) -> str:
    if value >= 90:
        return "Excellent"
    if value >= 80:
        return "Good"
    if value >= 65:
        return "Moderate"
    return "Needs work"


def _risk_status_from_health(value: float) -> str:
    if value >= 90:
        return "Low"
    if value >= 75:
        return "Moderate"
    if value >= 60:
        return "Elevated"
    return "High"


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
        rows.append(
            {
                "row_id": int(rid),
                "flags": flags,
                "severity": severity,
                "severity_label": SEVERITY_LABELS.get(severity, "Needs attention"),
            }
        )
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
