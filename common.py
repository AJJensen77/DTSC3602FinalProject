"""Shared Supabase connection handling for the dashboard and the uploader."""

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Aggregate tables the dashboard reads. The 22M-row ratings splits are
# deliberately NOT uploaded: the dashboard only ever needs these summaries.
TABLES = [
    "movies_dim",
    "rating_value_distribution",
    "rating_distribution_by_era",
    "monthly_activity",
    "user_activity_bins",
    "movie_popularity_bins",
    "genre_stats",
    "split_summary",
    "eval_cohort",
    "popularity_concentration",
    "data_dictionary",
]

TABLE_PREFIX = "d2_"


def table(name: str) -> str:
    """Namespace project tables so they cannot collide with other coursework."""
    return f"{TABLE_PREFIX}{name}"


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in your "
            "Supabase connection string (local), or set it as a Modal secret named "
            "'supabase-db-url' with a DATABASE_URL key (deployment)."
        )
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    return url


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)
