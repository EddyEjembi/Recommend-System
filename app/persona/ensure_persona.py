"""Ensure a test user has a persisted `persona` before downstream generation."""

from pathlib import Path

from app.generation.llm_client import LLMClient
from app.models.schemas import Persona
from app.persona.persona_service import PersonaBuildService
from app.persona.test_users_store import (
    atomic_write_json,
    find_test_user_bucket,
    load_test_users_payload,
)


def ensure_persona_for_test_user(
    user_id: str,
    processed_dir: Path,
    embeddings_dir: Path | None = None,
    *,
    force: bool = False,
    llm: LLMClient | None = None,
) -> Persona:
    """Return `persona` from `test_users.json`, building and saving it if missing.

    When `force` is True, always rebuild via the LLM and overwrite the stored persona.
    When `llm` is set, it is used for that build; otherwise `PersonaBuildService` uses its
    default client (environment API key).
    """
    test_path = processed_dir / "test_users.json"
    payload = load_test_users_payload(test_path)
    bucket, index = find_test_user_bucket(payload, user_id)
    entry = payload[bucket][index]

    if entry.get("persona") is not None and not force:
        return Persona.model_validate(entry["persona"])

    service = PersonaBuildService.from_default_paths(
        llm=llm,
        processed_dir=processed_dir,
        embeddings_dir=embeddings_dir,
    )
    persona = service.build_for_test_slot(user_id, bucket)
    payload[bucket][index]["persona"] = persona.model_dump(mode="json")
    atomic_write_json(test_path, payload)
    return persona
