"""Check that the deployed environment can actually reach Supabase.

The dashboard returning HTTP 200 only proves Streamlit booted; its data load
happens over a websocket afterwards, so a broken database connection still
looks healthy from outside. This runs the dashboard's query inside a Modal
container, with the same .env credentials, and prints the row counts it reads.

Deliberately self-contained: it defines its own small image and inlines the
connection logic, because anything it imported from the project would also have
to exist inside the container.

Usage:
    uv run modal run verify_deployment.py
"""

from pathlib import Path

import modal

LOCAL_DIR = Path(__file__).parent

app = modal.App("dtsc-3602-final-project-d2-verify")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sqlalchemy==2.1.6", "psycopg[binary]==3.3.6"
)

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


@app.function(image=image, secrets=[modal.Secret.from_dotenv(LOCAL_DIR)])
def check() -> dict:
    import os
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL", "").strip().strip('"').strip("'")
    if not url:
        return {"error": "DATABASE_URL never reached the container"}

    parsed = urlsplit(url)
    if parsed.query:
        kept = [(k, v) for k, v in parse_qsl(parsed.query) if k != "pgbouncer"]
        url = urlunsplit(parsed._replace(query=urlencode(kept)))
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]

    engine = create_engine(
        url, pool_pre_ping=True, connect_args={"prepare_threshold": None}
    )
    out: dict = {"counts": {}}
    with engine.connect() as conn:
        out["server"] = conn.execute(text("select version()")).scalar()[:38]
        for name in TABLES:
            out["counts"][name] = conn.execute(
                text(f"select count(*) from d2_{name}")
            ).scalar()
    return out


@app.local_entrypoint()
def main():
    out = check.remote()
    if "error" in out:
        print("FAILED:", out["error"])
        return
    print(f"Server reached from Modal: {out['server']}")
    total = 0
    for name, n in out["counts"].items():
        print(f"  d2_{name:<30} {n:>8,}")
        total += n
    print(f"\nTOTAL {total:,} rows readable from the deployed environment")
