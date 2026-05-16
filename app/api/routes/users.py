"""List demo users from `test_users.json`."""

from fastapi import APIRouter, HTTPException

from app.models.demo_list import TestUserListItem, TestUsersListResponse
from app.persona.test_users_store import load_test_users_payload
from app.retrieval.registry import default_processed_dir

router = APIRouter()


def _opt_int(value: object) -> int | None:
    """Coerce manifest numeric fields to `int` for the response model."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@router.get("", response_model=TestUsersListResponse)
def list_demo_users() -> TestUsersListResponse:
    """Return every `user_id` registered for this demo (`existing` + `cold_start`)."""
    path = default_processed_dir() / "test_users.json"
    try:
        payload = load_test_users_payload(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    users: list[TestUserListItem] = []
    for slot in ("existing", "cold_start"):
        for row in payload.get(slot) or []:
            uid = str(row.get("user_id") or "")
            if not uid:
                continue
            has_persona = row.get("persona") is not None
            src = _opt_int(row.get("subset_review_count"))
            users.append(
                TestUserListItem(
                    user_id=uid,
                    slot=slot,
                    has_persona=has_persona,
                    kind=row.get("kind") if isinstance(row.get("kind"), str) else None,
                    archetype=row.get("archetype") if isinstance(row.get("archetype"), str) else None,
                    name=row.get("name") if isinstance(row.get("name"), str) else None,
                    subset_review_count=src,
                )
            )
    return TestUsersListResponse(users=users)
