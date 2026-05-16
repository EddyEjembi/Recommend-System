"""Build FAISS indices for businesses, reviews, and users.

Shrink review artifacts only (leave businesses / users / test_users.json alone):

    uv run python -m app.ingestion.trim_reviews --demo
    uv run python -m app.retrieval.build_index --only reviews

Full rebuild from raw Yelp (all three parquet files) — use `app.ingestion.build_subset` instead.

Outputs land under `app/data/embeddings/`:
    businesses.faiss / businesses.meta.parquet
    reviews.faiss    / reviews.meta.parquet
    users.faiss      / users.meta.parquet
    manifest.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.retrieval.doc_builder import (
    build_business_doc,
    build_review_doc,
    build_user_doc,
)
from app.retrieval.embedder import SentenceTransformerEmbedder
from app.retrieval.registry import default_embeddings_dir, default_processed_dir
from app.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Per-namespace builders
# ---------------------------------------------------------------------------


def build_business_index(
    businesses_df: pd.DataFrame,
    embedder: SentenceTransformerEmbedder,
    out_path: Path,
) -> int:
    """Embed every business row and persist the FAISS index."""
    logger.info("Embedding %d businesses...", len(businesses_df))
    docs = [build_business_doc(row) for _, row in businesses_df.iterrows()]
    vectors = embedder.embed_batch(docs)

    store = FaissVectorStore(dim=embedder.dim)
    metadatas = [
        {
            "name": _safe(row.get("name")),
            "city": _safe(row.get("city")),
            "state": _safe(row.get("state")),
            "categories": _safe(row.get("categories")),
            "stars": _safe_float(row.get("stars")),
            "review_count": _safe_int(row.get("review_count")),
            "price_range": _safe_int(row.get("price_range")),
        }
        for _, row in businesses_df.iterrows()
    ]
    store.upsert(
        ids=businesses_df["business_id"].astype(str).tolist(),
        vectors=vectors,
        metadatas=metadatas,
    )
    store.save(out_path)
    logger.info("Wrote %s (%d vectors)", out_path.name, len(store))
    return len(store)


def build_review_index(
    reviews_df: pd.DataFrame,
    embedder: SentenceTransformerEmbedder,
    out_path: Path,
) -> int:
    """Embed every review and persist the FAISS index."""
    logger.info("Embedding %d reviews (this is the long step)...", len(reviews_df))
    docs = [build_review_doc(row) for _, row in reviews_df.iterrows()]
    vectors = embedder.embed_batch(docs)

    store = FaissVectorStore(dim=embedder.dim)
    metadatas = [
        {
            "user_id": _safe(row.get("user_id")),
            "business_id": _safe(row.get("business_id")),
            "stars": _safe_float(row.get("stars")),
            "date": _safe(row.get("date")),
        }
        for _, row in reviews_df.iterrows()
    ]
    store.upsert(
        ids=reviews_df["review_id"].astype(str).tolist(),
        vectors=vectors,
        metadatas=metadatas,
    )
    store.save(out_path)
    logger.info("Wrote %s (%d vectors)", out_path.name, len(store))
    return len(store)


def build_user_index(
    users_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    embedder: SentenceTransformerEmbedder,
    out_path: Path,
    sample_size: int = 3,
) -> int:
    """Embed every user (profile + sampled reviews) and persist the FAISS index."""
    logger.info(
        "Embedding %d users (with up to %d sampled reviews each)...",
        len(users_df),
        sample_size,
    )
    samples_by_user = _sample_reviews_by_user(reviews_df, sample_size)

    docs = [
        build_user_doc(row, samples_by_user.get(row["user_id"], []))
        for _, row in users_df.iterrows()
    ]
    vectors = embedder.embed_batch(docs)

    store = FaissVectorStore(dim=embedder.dim)
    metadatas = [
        {
            "name": _safe(row.get("name")),
            "review_count": _safe_int(row.get("review_count")),
            "average_stars": _safe_float(row.get("average_stars")),
            "is_elite": bool(row.get("is_elite", False)),
            "fans": _safe_int(row.get("fans")),
        }
        for _, row in users_df.iterrows()
    ]
    store.upsert(
        ids=users_df["user_id"].astype(str).tolist(),
        vectors=vectors,
        metadatas=metadatas,
    )
    store.save(out_path)
    logger.info("Wrote %s (%d vectors)", out_path.name, len(store))
    return len(store)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_all(
    processed_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Build all three indices and write `manifest.json`."""
    processed = processed_dir or default_processed_dir()
    out_dir = embeddings_dir or default_embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed Parquet files from %s", processed)
    businesses_df = pd.read_parquet(processed / "businesses.parquet")
    reviews_df = pd.read_parquet(processed / "reviews.parquet")
    users_df = pd.read_parquet(processed / "users.parquet")
    logger.info(
        "Loaded: %d businesses, %d reviews, %d users",
        len(businesses_df),
        len(reviews_df),
        len(users_df),
    )

    logger.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    logger.info("Embedding dim: %d", embedder.dim)

    counts: dict[str, int] = {}
    counts["businesses"] = build_business_index(
        businesses_df, embedder, out_dir / "businesses.faiss"
    )
    counts["reviews"] = build_review_index(
        reviews_df, embedder, out_dir / "reviews.faiss"
    )
    counts["users"] = build_user_index(
        users_df, reviews_df, embedder, out_dir / "users.faiss"
    )

    manifest = {
        "model_name": model_name,
        "embedding_dim": embedder.dim,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)

    for name in ("businesses", "reviews", "users"):
        faiss_path = out_dir / f"{name}.faiss"
        meta_path = out_dir / f"{name}.meta.parquet"
        if faiss_path.exists():
            faiss_mb = faiss_path.stat().st_size / (1024 * 1024)
            meta_mb = meta_path.stat().st_size / (1024 * 1024) if meta_path.exists() else 0.0
            logger.info("  %s.faiss: %.1f MB (+ meta %.1f MB)", name, faiss_mb, meta_mb)

    return manifest


def build_reviews_only(
    processed_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Rebuild only `reviews.faiss` / `reviews.meta.parquet` from `reviews.parquet`."""
    processed = processed_dir or default_processed_dir()
    out_dir = embeddings_dir or default_embeddings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews_df = pd.read_parquet(processed / "reviews.parquet")
    logger.info("Loaded %d reviews for index rebuild", len(reviews_df))

    logger.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformerEmbedder(model_name=model_name)

    count = build_review_index(reviews_df, embedder, out_dir / "reviews.faiss")

    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_name"] = model_name
    manifest["embedding_dim"] = embedder.dim
    manifest["built_at"] = datetime.now(timezone.utc).isoformat()
    if "counts" not in manifest or not isinstance(manifest.get("counts"), dict):
        manifest["counts"] = {}
    manifest["counts"]["reviews"] = count
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Updated %s (reviews count only)", manifest_path)

    for suffix in ("reviews.faiss", "reviews.meta.parquet"):
        path = out_dir / suffix
        if path.exists():
            logger.info("  %s: %.1f MB", suffix, path.stat().st_size / (1024 * 1024))

    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_reviews_by_user(
    reviews_df: pd.DataFrame, sample_size: int
) -> dict[str, list[str]]:
    """Return a dict mapping `user_id -> [up to `sample_size` review texts]`.

    Picks the highest-engagement reviews first (useful + funny + cool desc).
    """
    df = reviews_df.copy()
    df["_engagement"] = (
        df["useful"].fillna(0) + df["funny"].fillna(0) + df["cool"].fillna(0)
    )
    df = df.sort_values(
        by=["user_id", "_engagement", "date"],
        ascending=[True, False, False],
        kind="stable",
    )
    grouped = df.groupby("user_id", group_keys=False).head(sample_size)
    return (
        grouped.groupby("user_id")["text"]
        .apply(lambda series: [str(t) for t in series.tolist() if t])
        .to_dict()
    )


def _safe(value: Any) -> Any:
    """Convert NaN/None to None and pass everything else through."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for businesses, reviews, and users.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=("reviews",),
        default=None,
        help="Rebuild only the review index (after `trim_reviews`).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-Transformers model name (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    if args.only == "reviews":
        manifest = build_reviews_only(model_name=args.model)
        label = "Review index build complete"
    else:
        manifest = build_all(model_name=args.model)
        label = "Index build complete"

    print(f"\n=== {label} ===")
    print(f"  model      : {manifest['model_name']}")
    print(f"  dimension  : {manifest['embedding_dim']}")
    print(f"  built_at   : {manifest['built_at']}")
    counts = manifest.get("counts") or {}
    for name, count in counts.items():
        print(f"  {name:<10}: {count:>8} vectors")
    emb_dir = default_embeddings_dir()
    for suffix in ("reviews.faiss", "reviews.meta.parquet"):
        path = emb_dir / suffix
        if path.exists():
            print(f"  {suffix}: {path.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
