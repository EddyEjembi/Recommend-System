"""List businesses in the demo processed subset (`businesses.parquet`) with pagination."""

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.constants import BUSINESS_LIST_DEFAULT_LIMIT, BUSINESS_LIST_MAX_LIMIT
from app.models.demo_list import BusinessListItem, BusinessesListResponse
from app.retrieval.registry import default_processed_dir

router = APIRouter()


@lru_cache(maxsize=1)
def _businesses_dataframe() -> pd.DataFrame:
    """Load `businesses.parquet` once per process."""
    path = default_processed_dir() / "businesses.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing businesses subset: {path}")
    return pd.read_parquet(path)


def _normalize_categories(value: object) -> list[str] | str | None:
    """Turn Parquet category cell into JSON-friendly `list[str]` or `str`."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    return str(value)


def _row_to_item(row: pd.Series) -> BusinessListItem | None:
    """Map one Parquet row to a list item, or `None` if `business_id` is missing."""
    bid = row.get("business_id")
    if bid is None or (isinstance(bid, float) and pd.isna(bid)):
        return None
    stars = row.get("stars")
    if stars is not None and not (isinstance(stars, float) and pd.isna(stars)):
        stars_f = float(stars)
    else:
        stars_f = None
    rc = row.get("review_count")
    if rc is not None and not (isinstance(rc, float) and pd.isna(rc)):
        rc_i = int(rc)
    else:
        rc_i = None
    pr = row.get("price_range")
    price_range = None if pr is None or (isinstance(pr, float) and pd.isna(pr)) else str(pr)
    nm = row.get("name")
    name = None if nm is None or (isinstance(nm, float) and pd.isna(nm)) else str(nm)
    city = row.get("city")
    city_s = None if city is None or (isinstance(city, float) and pd.isna(city)) else str(city)
    state = row.get("state")
    state_s = None if state is None or (isinstance(state, float) and pd.isna(state)) else str(state)
    return BusinessListItem(
        business_id=str(bid),
        name=name,
        city=city_s,
        state=state_s,
        categories=_normalize_categories(row.get("categories")),
        stars=stars_f,
        review_count=rc_i,
        price_range=price_range,
    )


@router.get("", response_model=BusinessesListResponse)
def list_demo_businesses(
    limit: int = Query(
        BUSINESS_LIST_DEFAULT_LIMIT,
        ge=1,
        le=BUSINESS_LIST_MAX_LIMIT,
        description="Page size (default keeps responses small).",
    ),
    offset: int = Query(0, ge=0, description="Skip this many rows after optional filters and stable sort."),
    state: str | None = Query(
        None,
        min_length=2,
        max_length=2,
        description="Optional exact US state code (e.g. PA, FL, IN) to narrow the universe.",
    ),
    city: str | None = Query(
        None,
        min_length=1,
        max_length=120,
        description="Optional exact city match after trimming (e.g. Philadelphia).",
    ),
) -> BusinessesListResponse:
    """Return a **page** of businesses; the Parquet holds thousands of rows so we never return all at once.

    Rows are sorted by `state`, `city`, `business_id` for stable pagination. Combine `state` / `city`
    filters with `limit` / `offset` to browse the demo subset without loading everything into memory
    for JSON serialization.
    """
    try:
        df = _businesses_dataframe()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    state_norm = state.strip().upper() if state else None
    city_norm = city.strip() if city else None
    filtered = df
    if state_norm:
        st_col = filtered["state"].astype(str).str.strip().str.upper()
        filtered = filtered[st_col == state_norm]
    if city_norm:
        ct_col = filtered["city"].astype(str).str.strip()
        filtered = filtered[ct_col == city_norm]
    sort_cols = [c for c in ("state", "city", "business_id") if c in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)
    total_matching = int(len(filtered))
    page = filtered.iloc[offset : offset + limit]
    items: list[BusinessListItem] = []
    for _, row in page.iterrows():
        item = _row_to_item(row)
        if item is not None:
            items.append(item)
    return BusinessesListResponse(
        businesses=items,
        total_matching=total_matching,
        limit=limit,
        offset=offset,
        state_filter=state_norm,
        city_filter=city_norm,
    )
