"""Shared Supabase connection handling for the dashboard and the uploader."""

import os
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
    url = url.strip().strip('"').strip("'")

    # Supabase's copyable string carries ?pgbouncer=true for Prisma. libpq has no
    # such option and rejects the whole connection, so drop it (and any other
    # query flags) before handing the URL to psycopg.
    parsed = urlsplit(url)
    if parsed.query:
        kept = [(k, v) for k, v in parse_qsl(parsed.query) if k != "pgbouncer"]
        url = urlunsplit(parsed._replace(query=urlencode(kept)))

    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    return url


def _is_transaction_pooler(url: str) -> bool:
    """Supabase serves transaction-mode pooling on 6543, session mode on 5432."""
    return ":6543/" in url


@lru_cache
def get_engine() -> Engine:
    url = get_database_url()
    connect_args: dict = {}
    if _is_transaction_pooler(url):
        # In transaction mode a connection is handed to a different backend per
        # statement, so server-side prepared statements raise DuplicatePreparedStatement.
        connect_args["prepare_threshold"] = None
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)
