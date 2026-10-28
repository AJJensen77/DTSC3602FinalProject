"""Step 5 of the D2 pipeline: push the aggregate tables to Supabase Postgres.

Only the small summary tables go up - never the 22M-row ratings splits, which
stay local as parquet. Tables are namespaced with a `d2_` prefix so this cannot
collide with anything else in the database.

Usage:
    uv run python upload_to_supabase.py

Requires DATABASE_URL in .env (see .env.example).
"""

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from common import TABLES, get_engine, table
from dtsc_3602_final_project.paths import PROCESSED_DIR

load_dotenv()


def main() -> None:
    engine = get_engine()
    missing = [t for t in TABLES if not (PROCESSED_DIR / f"{t}.parquet").exists()]
    if missing:
        raise SystemExit(
            f"Missing parquet files: {missing}. Run prepare_data.py and "
            "make_data_dictionary.py first."
        )

    total = 0
    for name in TABLES:
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")

        # Postgres has no pandas nullable Float64/Int64; cast to plain numpy types.
        for col in df.columns:
            if str(df[col].dtype) in {"Float64", "Int64"}:
                df[col] = df[col].astype("float64")

        df.to_sql(
            table(name),
            engine,
            if_exists="replace",
            index=False,
            chunksize=5_000,
            method="multi",
        )
        total += len(df)
        print(f"  {table(name):<34} {len(df):>8,} rows")

    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE INDEX IF NOT EXISTS idx_{table('movies_dim')}_nr "
                 f"ON {table('movies_dim')} (n_ratings_train DESC)")
        )

    print(f"\nUploaded {len(TABLES)} tables, {total:,} rows total.")


if __name__ == "__main__":
    main()
