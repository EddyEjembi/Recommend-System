"""Deterministic scoring for recommendation candidates."""

from __future__ import annotations

import math
import re
from typing import Any

from app.models.schemas import Persona
from app.types.behavior import BusinessBehaviorProfile, UserBehaviorProfile
from app.types.recommendation import RankedCandidate

_WEIGHT_SEMANTIC = 0.25
_WEIGHT_CATEGORY = 0.20
_WEIGHT_PRICE = 0.15
_WEIGHT_SERVICE = 0.20
_WEIGHT_PERSONA = 0.10
_WEIGHT_REPUTATION = 0.10

_BUDGET_TO_PRICE: dict[str, int] = {
    "low": 1,
    "low-to-mid": 2,
    "mid": 2,
    "medium": 2,
    "mid-to-high": 3,
    "high": 4,
}

_SLOW_SERVICE_THEMES = frozenset(
    {"slow_service", "long_wait", "wait_time", "crowded", "noise", "noisy"}
)
_PORTION_THEMES = frozenset({"portion_size", "large_portions", "generous_portions", "portions"})
_FAST_SERVICE_THEMES = frozenset({"fast_service", "quick_service", "efficient_service"})


def _normalize_faiss_score(score: float) -> float:
    """Map inner-product cosine scores into [0, 1]."""
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def _user_budget_level(profile: UserBehaviorProfile) -> int | None:
    """Return target price_range 1–4 from cold seed or inferred preferences."""
    if profile.source == "cold_seed":
        cs = profile.cold_seed or {}
        prefs = cs.get("preferences") or {}
        budget = str(prefs.get("budget") or "").strip().lower()
        return _BUDGET_TO_PRICE.get(budget)
    return 2


def _user_cuisine_tokens(profile: UserBehaviorProfile, persona: Persona) -> set[str]:
    """Collect lowercase cuisine/category tokens for overlap scoring."""
    tokens: set[str] = set()
    for pref in persona.preferences:
        for part in re.split(r"[,;/|]+", str(pref)):
            t = part.strip().lower()
            if len(t) > 2:
                tokens.add(t)
    if profile.source == "cold_seed":
        cs = profile.cold_seed or {}
        prefs = cs.get("preferences") or {}
        cuisines = prefs.get("cuisines")
        if isinstance(cuisines, list):
            for c in cuisines:
                t = str(c).strip().lower()
                if t:
                    tokens.add(t)
    return tokens


def _category_overlap(categories: str | None, tokens: set[str]) -> float:
    if not categories or not tokens:
        return 0.5
    cat_lower = categories.lower()
    hits = sum(1 for t in tokens if t in cat_lower)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.65
    return 0.25


def _price_fit(user_price: int | None, business_price: int | None) -> float:
    if user_price is None or business_price is None:
        return 0.55
    diff = abs(user_price - business_price)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.75
    if diff == 2:
        return 0.45
    return 0.2


def _wait_tolerance(profile: UserBehaviorProfile) -> str:
    if profile.source != "cold_seed":
        return "medium"
    cs = profile.cold_seed or {}
    svc = cs.get("service_expectations") or {}
    return str(svc.get("wait_time_tolerance") or "medium").strip().lower()


def _theme_ids(themes: list[Any]) -> set[str]:
    out: set[str] = set()
    for t in themes:
        tid = getattr(t, "theme", None) or (t.get("theme") if isinstance(t, dict) else None)
        if tid:
            out.add(str(tid).lower())
    return out


def _service_fit(profile: UserBehaviorProfile, bb: BusinessBehaviorProfile | None) -> float:
    if bb is None:
        return 0.5
    complaints = _theme_ids(bb.complaint_themes)
    praise = _theme_ids(bb.praise_themes)
    tolerance = _wait_tolerance(profile)
    score = 0.55
    if tolerance in ("low", "short"):
        if complaints & _SLOW_SERVICE_THEMES:
            score -= 0.35
        if praise & _FAST_SERVICE_THEMES:
            score += 0.2
    elif tolerance in ("high", "long"):
        if praise & _PORTION_THEMES:
            score += 0.15
    cs = profile.cold_seed or {}
    svc = (cs.get("service_expectations") or {}) if profile.source == "cold_seed" else {}
    portion_imp = str(svc.get("portion_size_importance") or "").lower()
    if "high" in portion_imp or "very" in portion_imp:
        if praise & _PORTION_THEMES:
            score += 0.2
        if complaints & _PORTION_THEMES:
            score -= 0.15
    return max(0.0, min(1.0, score))


def _persona_theme_fit(persona: Persona, bb: BusinessBehaviorProfile | None) -> float:
    if bb is None:
        return 0.5
    praise_labels = {t.label.lower() for t in bb.praise_themes}
    complaint_labels = {t.label.lower() for t in bb.complaint_themes}
    score = 0.5
    for pref in persona.preferences:
        pl = pref.lower()
        if any(pl in lbl or lbl in pl for lbl in praise_labels):
            score += 0.12
    for deal in persona.dealbreakers:
        dl = deal.lower()
        if any(dl in lbl or lbl in dl for lbl in complaint_labels):
            score -= 0.18
    return max(0.0, min(1.0, score))


def _reputation_score(stars: float | None, review_count: int | None) -> float:
    if stars is None:
        return 0.45
    star_part = max(0.0, min(1.0, float(stars) / 5.0))
    rc = int(review_count or 0)
    count_part = min(1.0, math.log1p(rc) / math.log1p(200))
    return 0.65 * star_part + 0.35 * count_part


def score_candidate(
    *,
    business_id: str,
    name: str,
    row: dict[str, Any],
    semantic_raw: float,
    persona: Persona,
    profile: UserBehaviorProfile,
    business_behavior: BusinessBehaviorProfile | None,
) -> RankedCandidate:
    """Compute component scores and weighted retrieval_score for one business."""
    categories = str(row.get("categories") or "") or None
    stars_val = row.get("stars")
    stars = float(stars_val) if stars_val is not None else None
    rc_val = row.get("review_count")
    review_count = int(rc_val) if rc_val is not None else None
    pr_val = row.get("price_range")
    price_range = int(pr_val) if pr_val is not None and str(pr_val) != "nan" else None

    semantic = _normalize_faiss_score(semantic_raw)
    category = _category_overlap(categories, _user_cuisine_tokens(profile, persona))
    price = _price_fit(_user_budget_level(profile), price_range)
    service = _service_fit(profile, business_behavior)
    persona_fit = _persona_theme_fit(persona, business_behavior)
    reputation = _reputation_score(stars, review_count)

    retrieval = (
        _WEIGHT_SEMANTIC * semantic
        + _WEIGHT_CATEGORY * category
        + _WEIGHT_PRICE * price
        + _WEIGHT_SERVICE * service
        + _WEIGHT_PERSONA * persona_fit
        + _WEIGHT_REPUTATION * reputation
    )
    retrieval = max(0.0, min(1.0, retrieval))

    praise_snip = ""
    if business_behavior and business_behavior.praise_themes:
        t0 = business_behavior.praise_themes[0]
        praise_snip = t0.label
    complaint_snip = ""
    if business_behavior and business_behavior.complaint_themes:
        t0 = business_behavior.complaint_themes[0]
        complaint_snip = t0.label

    summary = (
        f"{name} | {categories or 'n/a'} | "
        f"stars={stars} reviews={review_count} price={price_range} | "
        f"retrieval_score={retrieval:.3f} | praise={praise_snip} complaint={complaint_snip}"
    )

    return RankedCandidate(
        business_id=business_id,
        name=name,
        city=str(row.get("city") or "") or None,
        state=str(row.get("state") or "") or None,
        categories=categories,
        stars=stars,
        review_count=review_count,
        price_range=price_range,
        semantic_score=semantic,
        category_score=category,
        price_score=price,
        service_score=service,
        persona_score=persona_fit,
        reputation_score=reputation,
        retrieval_score=retrieval,
        summary=summary,
    )
