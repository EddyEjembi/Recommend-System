"""Schema discovery for cold-start users (`GET /new-user`)."""

from fastapi import APIRouter

from app.models.requests import (
    NEW_DEMO_USER_EXAMPLE,
    NEW_DEMO_USER_NESTED_HINTS,
    NewDemoUserPayload,
)
from app.models.responses import NewUserSchemaResponse, NewUserUsageHint

router = APIRouter()


@router.get("", response_model=NewUserSchemaResponse)
def get_new_user_schema() -> NewUserSchemaResponse:
    """Return the JSON Schema and example for `new_user` accepted by `POST /recommend`."""
    schema = NewDemoUserPayload.model_json_schema(ref_template="#/components/schemas/{model}")
    required = [str(x) for x in schema.get("required", [])]
    return NewUserSchemaResponse(
        description=(
            "Define a cold-start demo user before recommendations. "
            "The server assigns a new `user_id`, appends the row to `test_users.json`, "
            "and registers behaviour for retrieval."
        ),
        json_schema=schema,
        required=required,
        example=NEW_DEMO_USER_EXAMPLE,
        nested_field_hints=NEW_DEMO_USER_NESTED_HINTS,
        usage=NewUserUsageHint(
            endpoint="POST /recommend",
            method="POST",
            body_field="new_user",
            also_required=[],
            mutually_exclusive_with=["user_id"],
            auth="Authorization: Bearer <openai_api_key>",
        ),
    )
