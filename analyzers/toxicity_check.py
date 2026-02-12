from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_TOXICITY_MODEL = "unitary/toxic-bert"


_HEURISTIC_BAD_WORDS = {
    "idiot",
    "stupid",
    "moron",
    "hate",
    "kill",
    "dumb",
}


def run_toxicity_check(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    model_name: str = DEFAULT_TOXICITY_MODEL,
    toxic_threshold: float = 0.80,
    batch_size: int = 16,
    prefer_hf: bool = True,
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

    texts = df[text_col].fillna("").astype(str).tolist()
    row_ids = df["_row_id"].astype(int).tolist() if "_row_id" in df.columns else df.index.astype(int).tolist()

    scores: List[float] = []
    backend = "heuristic"
    if prefer_hf:
        try:
            from transformers import pipeline

            clf = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
            outputs = clf(texts, batch_size=batch_size, truncation=True)
            scores = [_toxic_score(o) for o in outputs]
            backend = "transformers"
        except Exception:
            scores = []

    if not scores:
        scores = [_heuristic_toxicity_score(t) for t in texts]
        backend = "heuristic"

    toxic_rows = []
    for rid, s in zip(row_ids, scores, strict=True):
        if float(s) >= toxic_threshold:
            row_flags.setdefault(int(rid), []).append("toxic_text")
            toxic_rows.append(int(rid))

    toxic_pct = float(len(toxic_rows) / max(1, len(df)))
    if toxic_rows:
        issues.append(
            {
                "severity": "critical" if toxic_pct >= 0.05 else "warning",
                "code": "toxic_content",
                "message": f"Detected toxic/unsafe content in {len(toxic_rows)} rows ({toxic_pct:.1%}).",
                "recommendation": "Remove or filter toxic samples; consider separate moderation workflows.",
            }
        )

    label_toxicity = None
    if label_col is not None and label_col in df.columns:
        label_vals = df[label_col].fillna("").astype(str).tolist()
        agg: Dict[str, List[float]] = {}
        for lab, s in zip(label_vals, scores, strict=True):
            if lab:
                agg.setdefault(lab, []).append(float(s))
        label_toxicity = {lab: float(sum(v) / max(1, len(v))) for lab, v in agg.items()}

    metrics = {
        "rows": int(len(df)),
        "backend": backend,
        "model_name": model_name if backend == "transformers" else None,
        "toxic_threshold": float(toxic_threshold),
        "toxic_rows": int(len(toxic_rows)),
        "toxic_fraction": toxic_pct,
        "label_mean_toxicity": label_toxicity,
        "toxicity_scores_sample": [{"row_id": int(r), "score": float(s)} for r, s in list(zip(row_ids, scores, strict=True))[:50]],
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _toxic_score(output: object) -> float:
    if isinstance(output, dict) and "score" in output:
        return float(output.get("score", 0.0))
    if isinstance(output, list):
        lower = [(str(o.get("label", "")).lower(), float(o.get("score", 0.0))) for o in output if isinstance(o, dict)]
        for lbl, score in lower:
            if "toxic" in lbl and "non" not in lbl:
                return score
        for lbl, score in lower:
            if lbl in {"label_1", "1", "toxic"}:
                return score
        if lower:
            return max(lower, key=lambda t: t[1])[1]
    return 0.0


def _heuristic_toxicity_score(text: str) -> float:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z']+", text)}
    hits = len(tokens & _HEURISTIC_BAD_WORDS)
    return min(1.0, hits / 2.0)

