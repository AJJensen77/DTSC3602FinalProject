"""Canonical paths for the project.

Everything under ``data/`` is gitignored; see the README for how to recreate it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "ml-32m"
PROCESSED_DIR = ROOT / "data" / "processed"
RESERVED_TEST_DIR = ROOT / "data" / "reserved_test"

DOCS_DIR = ROOT / "docs"
FIGURES_DIR = ROOT / "figures"

RATINGS_CSV = RAW_DIR / "ratings.csv"
MOVIES_CSV = RAW_DIR / "movies.csv"
LINKS_CSV = RAW_DIR / "links.csv"
TAGS_CSV = RAW_DIR / "tags.csv"

# Reading 32M rows with explicit dtypes keeps ratings.csv around 0.5 GB in memory
# instead of the ~2 GB pandas would default to.
RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, RESERVED_TEST_DIR, DOCS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
