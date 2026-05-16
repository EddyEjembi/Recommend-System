"""Health-check endpoints used by uptime probes and orchestrators."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check() -> dict[str, str]:
    """Return a simple liveness payload."""
    return {"status": "ok"}


@router.get("/ready")
def readiness_check() -> dict[str, str]:
    """Return readiness state.

    Extend this to verify downstream dependencies (vector store, LLM client, etc.).
    """
    return {"status": "ready"}
