"""Application configuration.

Centralised settings powered by environment variables. Using `lru_cache`
ensures the settings object is built once per process.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    """Strongly typed application configuration."""

    app_name: str
    app_version: str
    log_level: str
    data_dir: Path
    embeddings_dir: Path
    raw_dir: Path
    processed_dir: Path
    llm_model: str
    embedding_model: str
    vector_store_path: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance for the current process."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"

    return Settings(
        app_name=os.getenv("APP_NAME", "recommendation-system"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        data_dir=data_dir,
        embeddings_dir=data_dir / "embeddings",
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        vector_store_path=data_dir / "embeddings" / "index",
    )
