"""Hybrid-model step 1: attach IMDb rating, votes, cast and director to each film.

Run after prepare_data.py. Needs these files in data/raw/ (see README):
title.ratings.tsv.gz, title.principals.tsv.gz, title.crew.tsv.gz

Writes data/processed/movie_features.parquet, one row per film kept in D2.

Usage:
    uv run python "Gradient boost model/prepare_imdb.py"
"""

import json
from pathlib import Path

import pandas as pd

from gbmodel.imdb_features import load_directors, load_ratings, load_top_actors
from dtsc_3602_final_project.paths import PROCESSED_DIR, RAW_DIR

IMDB_DIR = RAW_DIR.parent
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    catalog = pd.read_parquet(PROCESSED_DIR / "catalog.parquet")
    films = catalog[~catalog["is_tv"]]
    keep = set(films["tconst"].dropna())
    print(f"Matching {len(films):,} films to IMDb...", flush=True)

    ratings = load_ratings(IMDB_DIR / "title.ratings.tsv.gz", keep)
    print(f"  ratings:   {len(ratings):,}", flush=True)
    actors = load_top_actors(IMDB_DIR / "title.principals.tsv.gz", keep)
    print(f"  cast:      {len(actors):,}", flush=True)
    directors = load_directors(IMDB_DIR / "title.crew.tsv.gz", keep)
    print(f"  directors: {len(directors):,}", flush=True)

    out = (
        films[["movieId", "tconst", "clean_title", "year", "genres"]]
        .merge(ratings, on="tconst", how="left")
        .merge(actors, on="tconst", how="left")
        .merge(directors, on="tconst", how="left")
    )
    # Empty lists rather than NaN, so downstream code can always iterate.
    for col in ("actors", "directors"):
        out[col] = out[col].apply(lambda v: list(v) if isinstance(v, list) else [])
    out["year"] = out["year"].astype("float64")
    out.to_parquet(PROCESSED_DIR / "movie_features.parquet", index=False)

    report = {
        "n_films": int(len(out)),
        "pct_with_imdb_rating": float(out["averageRating"].notna().mean() * 100),
        "pct_with_actors": float((out["actors"].str.len() > 0).mean() * 100),
        "pct_with_director": float((out["directors"].str.len() > 0).mean() * 100),
    }
    (RESULTS_DIR / "imdb_features_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
