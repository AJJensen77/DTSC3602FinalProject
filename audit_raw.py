"""Step 1 of the D2 pipeline: audit the raw MovieLens files before any cleaning.

Writes docs/audit_report.json so the data note quotes measured numbers rather
than numbers copied from the dataset README.

Usage:
    uv run python audit_raw.py
"""

import json
import re

import numpy as np
import pandas as pd

from dtsc_3602_final_project.paths import (
    DOCS_DIR,
    LINKS_CSV,
    MOVIES_CSV,
    RATINGS_CSV,
    RATINGS_DTYPES,
    TAGS_CSV,
    ensure_dirs,
)

# MovieLens encodes the release year in the title, e.g. "Toy Story (1995)".
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def audit_ratings() -> tuple[pd.DataFrame, dict]:
    print("Reading ratings.csv (877 MB, this takes a minute)...")
    ratings = pd.read_csv(RATINGS_CSV, dtype=RATINGS_DTYPES)

    ts = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    dup_mask = ratings.duplicated(subset=["userId", "movieId"], keep=False)
    value_counts = ratings["rating"].value_counts().sort_index()
    per_user = ratings.groupby("userId").size()

    report = {
        "n_rows": int(len(ratings)),
        "n_users": int(ratings["userId"].nunique()),
        "n_movies_rated": int(ratings["movieId"].nunique()),
        "nulls": {c: int(ratings[c].isna().sum()) for c in ratings.columns},
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_median": float(ratings["rating"].median()),
        "distinct_rating_values": [float(v) for v in value_counts.index],
        "rating_value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "half_star_share": float(
            ratings["rating"].isin([0.5, 1.5, 2.5, 3.5, 4.5]).mean()
        ),
        "n_duplicate_user_movie_pairs": int(dup_mask.sum()),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
        "n_timestamps_out_of_range": int(
            ((ts < "1995-01-01") | (ts > "2023-10-31")).sum()
        ),
        "ratings_per_user": {
            "min": int(per_user.min()),
            "p25": float(per_user.quantile(0.25)),
            "median": float(per_user.median()),
            "p75": float(per_user.quantile(0.75)),
            "p99": float(per_user.quantile(0.99)),
            "max": int(per_user.max()),
            "mean": float(per_user.mean()),
        },
        "n_users_below_20_ratings": int((per_user < 20).sum()),
    }
    return ratings, report


def audit_movies() -> tuple[pd.DataFrame, dict]:
    movies = pd.read_csv(MOVIES_CSV, dtype={"movieId": "int32"})
    year = movies["title"].str.extract(YEAR_RE)[0].astype("float")
    genres_split = movies["genres"].str.split("|")
    all_genres = sorted({g for gs in genres_split for g in gs})

    base_title = movies["title"].str.replace(YEAR_RE, "", regex=True).str.strip()
    dup_title_year = movies.duplicated(subset=["title"], keep=False)

    report = {
        "n_movies": int(len(movies)),
        "nulls": {c: int(movies[c].isna().sum()) for c in movies.columns},
        "n_duplicate_movie_ids": int(movies["movieId"].duplicated().sum()),
        "n_duplicate_titles": int(dup_title_year.sum()),
        "n_missing_year_in_title": int(year.isna().sum()),
        "year_min": None if year.isna().all() else float(np.nanmin(year)),
        "year_max": None if year.isna().all() else float(np.nanmax(year)),
        "n_no_genres_listed": int((movies["genres"] == "(no genres listed)").sum()),
        "n_distinct_genres": len(all_genres),
        "genres": all_genres,
        "example_duplicate_titles": base_title[dup_title_year].head(5).tolist(),
    }
    return movies, report


def audit_links() -> dict:
    links = pd.read_csv(LINKS_CSV, dtype={"movieId": "int32", "imdbId": "str"})
    return {
        "n_rows": int(len(links)),
        "nulls": {c: int(links[c].isna().sum()) for c in links.columns},
        "tmdb_coverage": float(links["tmdbId"].notna().mean()),
    }


def audit_tags() -> dict:
    tags = pd.read_csv(TAGS_CSV, dtype={"userId": "int32", "movieId": "int32"})
    return {
        "n_rows": int(len(tags)),
        "n_users": int(tags["userId"].nunique()),
        "n_movies": int(tags["movieId"].nunique()),
        "nulls": {c: int(tags[c].isna().sum()) for c in tags.columns},
    }


def main() -> None:
    ensure_dirs()
    ratings, ratings_report = audit_ratings()
    movies, movies_report = audit_movies()

    rated_ids = set(ratings["movieId"].unique())
    catalog_ids = set(movies["movieId"])

    cross = {
        "n_rated_movies_missing_from_movies_csv": len(rated_ids - catalog_ids),
        "n_catalog_movies_with_zero_ratings": len(catalog_ids - rated_ids),
    }

    # How concentrated are ratings? This is the popularity-bias risk from D1.
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False)
    total = per_movie.sum()
    cross["top_1pct_movies_share_of_ratings"] = float(
        per_movie.head(max(1, len(per_movie) // 100)).sum() / total
    )
    cross["top_10pct_movies_share_of_ratings"] = float(
        per_movie.head(max(1, len(per_movie) // 10)).sum() / total
    )
    cross["n_movies_with_fewer_than_20_ratings"] = int((per_movie < 20).sum())
    cross["median_ratings_per_rated_movie"] = float(per_movie.median())

    report = {
        "ratings": ratings_report,
        "movies": movies_report,
        "links": audit_links(),
        "tags": audit_tags(),
        "cross_file": cross,
    }

    out = DOCS_DIR / "audit_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
