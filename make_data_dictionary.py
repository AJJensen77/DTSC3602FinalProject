"""Step 4 of the D2 pipeline: build the data dictionary.

Column names and dtypes are read back off the written parquet files, so the
dictionary cannot drift from what the pipeline actually produced. Descriptions
are authored here; a missing one is reported as a gap rather than left blank.

Usage:
    uv run python make_data_dictionary.py
"""

import pandas as pd

from dtsc_3602_final_project.paths import (
    DOCS_DIR,
    PROCESSED_DIR,
    RESERVED_TEST_DIR,
    ensure_dirs,
)

TABLE_PURPOSE = {
    "ratings_train": "One row per rating placed on or before 2016-10-01. Model fitting data.",
    "ratings_validation": "One row per rating from 2016-10-01 to 2019-11-02. Model selection data.",
    "ratings_test": "RESERVED. One row per rating after 2019-11-02. Opened once, in D6.",
    "movies_dim": "One row per film kept after television was removed. Dashboard catalog.",
    "rating_value_distribution": "Count of each star value within each split.",
    "rating_distribution_by_era": "Star values before vs. after half-stars existed (Feb 2003).",
    "monthly_activity": "One row per calendar month of the collection window.",
    "user_activity_bins": "Users bucketed by how many training ratings they placed.",
    "movie_popularity_bins": "Movies bucketed by how many training ratings they received.",
    "genre_stats": "One row per MovieLens genre label.",
    "split_summary": "One row per split, with its time span and size.",
    "eval_cohort": "How much of each evaluation split a personalized model can score.",
    "popularity_concentration": "Lorenz curve of rating concentration across movies.",
    "catalog": "Full MovieLens catalog joined to IMDb title types, before TV removal.",
}

COLUMN_DESCRIPTIONS = {
    "userId": "Anonymous MovieLens user identifier. Stable across files; no demographics.",
    "movieId": "MovieLens movie identifier. Primary key of movies_dim; joins every table.",
    "rating": "Star rating the user gave, 0.5-5.0 in 0.5 steps. The prediction target.",
    "timestamp": "Unix epoch seconds (UTC) when the rating was recorded. Drives the split.",
    "title": "Film title with the trailing release year stripped off.",
    "clean_title": "Film title with the trailing release year stripped off.",
    "year": "Release year parsed from the raw MovieLens title. Null where absent.",
    "genres": "Pipe-separated MovieLens genre labels, or '(no genres listed)'.",
    "genre": "A single genre label, after splitting the pipe-separated field.",
    "titleType": "IMDb title type used to drop television. 'unknown' where unmatched.",
    "imdbId": "IMDb identifier from links.csv, without the 'tt' prefix.",
    "tconst": "Zero-padded IMDb identifier ('tt0114709') used for the IMDb join.",
    "is_tv": "True where titleType is episodic television. These rows were dropped.",
    "is_film": "True where titleType is a film type (movie, tvMovie, video, short).",
    "n_ratings_train": "Ratings this movie received in the training split only.",
    "mean_rating_train": "Mean star rating in the training split only. Null if never rated.",
    "n_ratings": "Number of ratings in the group.",
    "mean_rating": "Mean star rating in the group.",
    "n_users": "Number of distinct users in the group.",
    "n_movies": "Number of distinct movies in the group.",
    "split": "Which chronological split the row belongs to: train, validation, or test.",
    "era": "Whether the rating predates the Feb 2003 introduction of half-stars.",
    "share_within_era": "Share of that era's ratings falling on this star value.",
    "month": "First day of the calendar month (UTC, tz-naive).",
    "bucket": "Label of the count range this group covers.",
    "share_of_ratings": "This split's share of all retained ratings.",
    "start": "Timestamp of the earliest rating in the split (UTC).",
    "end": "Timestamp of the latest rating in the split (UTC).",
    "pct_ratings_from_cold_users": "Percent of the split's ratings from users absent from train.",
    "pct_ratings_on_cold_movies": "Percent of the split's ratings on movies absent from train.",
    "n_users_total": "Distinct users appearing in the split.",
    "n_users_warm": "Users in the split that also appear in training data.",
    "n_users_scored": "Users with >=20 training and >=5 evaluation ratings.",
    "n_ratings_total": "All ratings in the split.",
    "n_ratings_scored": "Ratings belonging to scorable users.",
    "pct_ratings_scored": "n_ratings_scored as a percent of n_ratings_total.",
    "mean_rating_scored": "Mean star rating among scorable ratings.",
    "mean_rating_unscored": "Mean star rating among ratings that cannot be scored.",
    "pct_movies": "Share of movies, ordered most-rated first.",
    "pct_ratings": "Cumulative share of training ratings held by those movies.",
    "rated_at": "Rating timestamp as a UTC datetime.",
}


def main() -> None:
    ensure_dirs()
    sources = sorted(PROCESSED_DIR.glob("*.parquet")) + sorted(
        RESERVED_TEST_DIR.glob("*.parquet")
    )

    rows = []
    missing = []
    for path in sources:
        table = path.stem
        # Read an empty slice: schema only, no need to pull 22M rows into memory.
        head = pd.read_parquet(path).head(200)
        for col in head.columns:
            desc = COLUMN_DESCRIPTIONS.get(col)
            if desc is None:
                missing.append(f"{table}.{col}")
                desc = "(undocumented)"
            rows.append(
                {
                    "table_name": table,
                    "column_name": col,
                    "dtype": str(head[col].dtype),
                    "description": desc,
                    "table_purpose": TABLE_PURPOSE.get(table, "(undocumented)"),
                }
            )

    dd = pd.DataFrame(rows)
    dd.to_parquet(PROCESSED_DIR / "data_dictionary.parquet", index=False)

    lines = [
        "# Data dictionary",
        "",
        "Generated by `make_data_dictionary.py` from the parquet files the",
        "pipeline actually wrote. Do not edit by hand.",
        "",
        "Source: MovieLens 32M (GroupLens) joined to IMDb `title.basics.tsv`.",
        "",
    ]
    for table, grp in dd.groupby("table_name", sort=True):
        n_rows = len(pd.read_parquet(_path_for(table, sources)))
        lines += [
            f"## `{table}`",
            "",
            f"{TABLE_PURPOSE.get(table, '')}  ",
            f"Rows: {n_rows:,}",
            "",
            "| Column | Type | Description |",
            "|---|---|---|",
        ]
        for _, r in grp.iterrows():
            lines.append(f"| `{r.column_name}` | `{r.dtype}` | {r.description} |")
        lines.append("")

    out = DOCS_DIR / "data_dictionary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(dd)} columns across {dd['table_name'].nunique()} tables)")
    if missing:
        print(f"WARNING: {len(missing)} undocumented columns: {missing}")


def _path_for(table: str, sources: list) -> str:
    return next(p for p in sources if p.stem == table)


if __name__ == "__main__":
    main()
