"""Extract complaint and praise themes from review text without an LLM."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.behavior.theme_lexicon import COMPLAINT_PHRASES, PRAISE_PHRASES, THEME_LABELS
from app.types.behavior import ThemeSignal


def _hits_for_phrases(text_lower: str, phrases: list[str]) -> int:
    """Return how many distinct phrase needles appear at least once in `text_lower`."""
    count = 0
    for phrase in phrases:
        if phrase in text_lower:
            count += 1
    return count


def _first_snippet(text: str, phrase: str, window: int = 120) -> str | None:
    """Return a short verbatim window around the first occurrence of `phrase`."""
    lower = text.lower()
    idx = lower.find(phrase)
    if idx == -1:
        return None
    start = max(0, idx - 40)
    end = min(len(text), idx + len(phrase) + 40)
    snippet = text[start:end].replace("\n", " ").strip()
    if len(snippet) > window:
        snippet = snippet[: window - 3] + "..."
    return snippet


def _collect_theme_scores(
    reviews: Iterable[dict[str, Any]],
    phrase_map: dict[str, list[str]],
    polarity: str,
) -> tuple[dict[str, float], dict[str, int], dict[str, list[str]]]:
    """Aggregate weighted scores per theme across reviews.

    `polarity` is ``complaint`` (weight higher on low stars) or ``praise``.
    """
    weighted: dict[str, float] = defaultdict(float)
    hits: dict[str, int] = defaultdict(int)
    snippets: dict[str, list[str]] = defaultdict(list)

    for review in reviews:
        text = str(review.get("text") or "")
        if not text.strip():
            continue
        text_lower = text.lower()
        stars = review.get("stars")
        try:
            star_f = float(stars) if stars is not None else 3.0
        except (TypeError, ValueError):
            star_f = 3.0

        if polarity == "complaint":
            weight = 1.0 + max(0.0, 3.5 - star_f) * 0.35
        else:
            weight = 1.0 + max(0.0, star_f - 3.5) * 0.35

        for theme_id, phrases in phrase_map.items():
            hit = _hits_for_phrases(text_lower, phrases)
            if hit == 0:
                continue
            weighted[theme_id] += float(hit) * weight
            hits[theme_id] += 1
            for phrase in phrases:
                if phrase in text_lower:
                    snip = _first_snippet(text, phrase)
                    if snip and len(snippets[theme_id]) < 3:
                        snippets[theme_id].append(snip)
                    break

    return dict(weighted), dict(hits), {k: list(v) for k, v in snippets.items()}


def _normalise_scores(raw: dict[str, float], review_count: int) -> dict[str, float]:
    """Turn raw weighted hits into roughly comparable scores in [0, 1]."""
    if review_count <= 0 or not raw:
        return {}
    max_w = max(raw.values()) if raw.values() else 0.0
    if max_w <= 0:
        return {}
    return {k: min(1.0, v / max_w) for k, v in raw.items()}


def extract_themes(
    reviews: list[dict[str, Any]],
    top_k: int = 5,
) -> tuple[list[ThemeSignal], list[ThemeSignal]]:
    """Return `(praise_themes, complaint_themes)` sorted by strength."""
    n = max(1, len(reviews))

    comp_w, comp_h, comp_snip = _collect_theme_scores(reviews, COMPLAINT_PHRASES, "complaint")
    praise_w, praise_h, praise_snip = _collect_theme_scores(reviews, PRAISE_PHRASES, "praise")

    comp_scores = _normalise_scores(comp_w, n)
    praise_scores = _normalise_scores(praise_w, n)

    complaints = [
        ThemeSignal(
            theme=tid,
            label=THEME_LABELS.get(tid, tid),
            score=float(comp_scores.get(tid, 0.0)),
            hit_count=int(comp_h.get(tid, 0)),
            snippets=comp_snip.get(tid, [])[:3],
        )
        for tid in comp_scores
    ]
    complaints.sort(key=lambda x: x.score, reverse=True)
    complaints = complaints[:top_k]

    praises = [
        ThemeSignal(
            theme=tid,
            label=THEME_LABELS.get(tid, tid),
            score=float(praise_scores.get(tid, 0.0)),
            hit_count=int(praise_h.get(tid, 0)),
            snippets=praise_snip.get(tid, [])[:3],
        )
        for tid in praise_scores
    ]
    praises.sort(key=lambda x: x.score, reverse=True)
    praises = praises[:top_k]

    return praises, complaints
