"""CLI: build a persona for a `user_id` listed in `test_users.json` and save it there.

Flow:
  1. Resolve `user_id` under `existing` or `cold_start` in `app/data/processed/test_users.json`.
     Unknown ids **error** (not in either list).
  2. If that entry already has a `persona` object and `--force` was not passed, print it and exit
     (no LLM call, no file rewrite).
  3. Otherwise load retrieval + `user_behavior.jsonl`, call the LLM:
     - `existing` -> warm pipeline (review history + behaviour)
     - `cold_start` -> cold pipeline (seed + similar users from vector DB + neighbour behaviour)
  4. Write the persona back onto that user object in `test_users.json` (atomic replace).

Requires `OPENAI_API_KEY`, Phase 2 embeddings, Phase 3 `user_behavior.jsonl`.

Usage:

    uv run python -m app.persona.build_persona --user-id Hi10sGSZNxQH3NLyWSZ1oA

    uv run python -m app.persona.build_persona --user-id cold_ng_001_lagos_budget_diner

    uv run python -m app.persona.build_persona --user-id ... --force

Optional `--output path.json` copies the persona JSON to another file for debugging only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.models.schemas import Persona
from app.persona.persona_cache import PersonaCache
from app.persona.persona_service import PersonaBuildService
from app.persona.test_users_store import (
    atomic_write_json,
    find_test_user_bucket,
    load_test_users_payload,
)
from app.retrieval.registry import default_processed_dir
from app.utils.json_utils import write_json

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--user-id",
        required=True,
        help="Must appear under 'existing' or 'cold_start' in test_users.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional extra copy of the persona JSON (debug only). test_users.json is always updated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate with the LLM even if `persona` is already stored on this user.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override app/data/processed (default: package-relative).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Override app/data/embeddings (default: package-relative).",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve manifest, reuse cached persona, or call LLM and persist to test_users.json."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    processed = args.processed_dir or default_processed_dir()
    test_path = processed / "test_users.json"

    payload = load_test_users_payload(test_path)
    bucket, index = find_test_user_bucket(payload, args.user_id)
    entry = payload[bucket][index]

    if entry.get("persona") is not None and not args.force:
        logger.info("Persona already on file for %s - use --force to regenerate.", args.user_id)
        persona = Persona.model_validate(entry["persona"])
        _emit(persona, args.output)
        return

    cache = PersonaCache()
    if not args.force:
        cached = cache.get(args.user_id)
        if cached is not None:
            logger.info("In-memory cache hit for %s - use --force to regenerate.", args.user_id)
            _emit(cached, args.output)
            _persist_persona(payload, bucket, index, cached, test_path)
            return

    logger.info("Loading retrieval + behaviour (first run can take ~30s)...")
    service = PersonaBuildService.from_default_paths(
        processed_dir=processed,
        embeddings_dir=args.embeddings_dir,
    )
    logger.info("Calling LLM for persona: %s (%s)", args.user_id, bucket)
    persona = service.build_for_test_slot(args.user_id, bucket)
    cache.set(args.user_id, persona)

    _persist_persona(payload, bucket, index, persona, test_path)
    logger.info("Attached persona to %s in %s", args.user_id, test_path.name)
    _emit(persona, args.output)


def _persist_persona(
    payload: dict,
    bucket: str,
    index: int,
    persona: Persona,
    test_path: Path,
) -> None:
    """Write `persona` onto the manifest row and save atomically."""
    payload[bucket][index]["persona"] = persona.model_dump(mode="json")
    atomic_write_json(test_path, payload)


def _emit(persona: Persona, output: Path | None) -> None:
    """Print JSON to stdout and optionally write an extra debug copy."""
    text = persona.model_dump_json(indent=2)
    print(text)
    if output is not None:
        write_json(output, persona.model_dump())
        logger.info("Wrote extra copy to %s", output)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
