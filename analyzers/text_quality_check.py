from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"[A-Za-z']+")


def run_text_quality_check(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
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

    word_counts: List[int] = []
    sentence_counts: List[int] = []
    syllable_counts: List[int] = []
    repeat_ratios: List[float] = []

    total_words = 0
    vocab: set[str] = set()

    for rid, t in zip(row_ids, texts, strict=True):
        words = [w.lower() for w in _WORD_RE.findall(t)]
        wc = len(words)
        if wc == 0:
            row_flags.setdefault(int(rid), []).append("no_words")
        total_words += wc
        vocab.update(words)

        uniq = len(set(words)) if wc else 0
        rep_ratio = float(1.0 - (uniq / wc)) if wc else 0.0
        repeat_ratios.append(rep_ratio)

        sc = _sentence_count(t)
        sy = sum(_syllables(w) for w in words)

        word_counts.append(wc)
        sentence_counts.append(sc)
        syllable_counts.append(sy)

    avg_words = float(np.mean(word_counts)) if word_counts else 0.0
    avg_repeat = float(np.mean(repeat_ratios)) if repeat_ratios else 0.0
    vocab_richness = float(len(vocab) / max(1, total_words))
    readability = _flesch_reading_ease(sum(word_counts), sum(sentence_counts), sum(syllable_counts))

    if readability > 0 and readability < 30:
        issues.append(
            {
                "severity": "warning",
                "code": "low_readability",
                "message": f"Low average readability (Flesch={readability:.1f}).",
                "recommendation": "Check for overly complex, noisy, or malformed text; consider cleaning.",
            }
        )

    metrics = {
        "rows": int(len(df)),
        "avg_words_per_row": avg_words,
        "vocabulary_richness": vocab_richness,
        "avg_repetition_ratio": avg_repeat,
        "flesch_reading_ease": readability,
    }
    return {"metrics": metrics, "issues": issues, "row_flags": row_flags}


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    count = sum(1 for p in parts if p.strip())
    return max(1, count)


def _syllables(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(words: int, sentences: int, syllables: int) -> float:
    if words <= 0 or sentences <= 0:
        return 0.0
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

