"""Endpoint for behavioral recommendations (`POST /recommend`)."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AuthenticationError

from app.generation.openai_client import OpenAILLMClient
from app.models.requests import RecommendRequest
from app.models.responses import RecommendResponse
from app.persona.api_user_registration import persist_new_cold_start_demo_user
from app.recommendation.recommendation_service import RecommendationService
from app.retrieval.registry import default_processed_dir

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _recommendation_service() -> RecommendationService:
    """Load retrieval data once per process (no API key at startup)."""
    return RecommendationService.from_default_paths()


def _llm_client_for_request(credentials: HTTPAuthorizationCredentials | None) -> OpenAILLMClient:
    """Build LLM client: Bearer key when sent, otherwise AI_API_KEY / OPENAI_API_KEY from env."""
    header_key: str | None = None
    if credentials is not None:
        header_key = (credentials.credentials or "").strip() or None
    if header_key:
        return OpenAILLMClient(api_key=header_key, allow_env_fallback=False)
    try:
        return OpenAILLMClient(allow_env_fallback=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=(
                "No API key. Send Authorization: Bearer <key>, "
                "or set AI_API_KEY / OPENAI_API_KEY in the environment."
            ),
        ) from exc


@router.post("", response_model=RecommendResponse)
def make_recommendations(
    payload: RecommendRequest,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> RecommendResponse:
    """Generate personalized recommendations Requires `Authorization: Bearer <openai_api_key>`."""
    llm = _llm_client_for_request(credentials)
    service = _recommendation_service()
    resolved_user_id: str
    if payload.new_user is not None:
        nu = payload.new_user
        try:
            new_id, profile = persist_new_cold_start_demo_user(
                default_processed_dir(),
                archetype=nu.archetype,
                demographics=nu.demographics,
                preferences=nu.preferences,
                service_expectations=nu.service_expectations,
                notes=nu.notes,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service.register_runtime_user_profile(profile)
        resolved_user_id = new_id
    else:
        resolved_user_id = (payload.user_id or "").strip()

    limit = payload.resolved_limit()
    try:
        result = service.generate(resolved_user_id, limit=limit, llm=llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Unknown user_id="):
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=502, detail=message) from exc
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RecommendResponse.from_result(resolved_user_id, result)
