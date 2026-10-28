"""Step 2 of the D2 pipeline: clean, drop television, split, and aggregate.

Reads the raw MovieLens files plus IMDb ``title.basics.tsv.gz`` and writes:

  data/processed/       train + validation ratings and the dashboard aggregates
  data/reserved_test/   the sealed final test split (not opened again until D6)
  docs/prep_report.json measured before/after numbers quoted by the data note

Usage:
    uv run python prepare_data.py
"""

import json

import numpy as np
import pandas as pd

from dtsc_3602_final_project.catalog import build_catalog, load_imdb_types
from dtsc_3602_final_project.paths import (
    DOCS_DIR,
    LINKS_CSV,
    MOVIES_CSV,
    PROCESSED_DIR,
    RATINGS_CSV,
    RATINGS_DTYPES,
    RAW_DIR,
    RESERVED_TEST_DIR,
    ensure_dirs,
)

IMDB_BASICS = RAW_DIR.parent / "title.basics.tsv.gz"

# Chronological split points. The question is "earlier ratings -> later ratings",
# so the split is by time, never at random: a random split would let the model
# learn from a user's future to predict their past.
TRAIN_END_Q = 0.70
VAL_END_Q = 0.85

report: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Loading catalog + IMDb title types...")
    movies = pd.read_csv(MOVIES_CSV, dtype={"movieId": "int32"})
    links = pd.read_csv(LINKS_CSV, dtype={"movieId": "int32"})
    imdb = load_imdb_types(IMDB_BASICS)
    catalog = build_catalog(movies, links, imdb)

    log("Loading ratings.csv (32M rows)...")
    ratings = pd.read_csv(RATINGS_CSV, dtype=RATINGS_DTYPES)
    return catalog, ratings


def drop_television(catalog: pd.DataFrame, ratings: pd.DataFrame):
    """Remove TV titles, keeping films (incl. made-for-TV and direct-to-video films).

    Titles with no IMDb match keep their rows: inspection showed they are
    ordinary films whose imdbId is simply stale, not television.
    """
    tv = catalog[catalog["is_tv"]]
    tv_ids = set(tv["movieId"])

    type_counts = catalog["titleType"].value_counts(dropna=False)
    ratings_before = len(ratings)

    kept_catalog = catalog[~catalog["is_tv"]].copy()
    tv_rating_mask = ratings["movieId"].isin(tv_ids)
    kept_ratings = ratings[~tv_rating_mask].copy()

    report["tv_removal"] = {
        "title_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "n_titles_before": int(len(catalog)),
        "n_tv_titles_dropped": int(len(tv_ids)),
        "n_titles_after": int(len(kept_catalog)),
        "n_unmatched_titles_kept": int((catalog["titleType"] == "unknown").sum()),
        "n_ratings_before": int(ratings_before),
        "n_ratings_dropped_with_tv": int(tv_rating_mask.sum()),
        "pct_ratings_dropped_with_tv": float(tv_rating_mask.mean() * 100),
        "n_ratings_after": int(len(kept_ratings)),
        "dropped_types": sorted({str(t) for t in tv["titleType"].unique()}),
        "most_rated_dropped_titles": (
            ratings[tv_rating_mask]
            .groupby("movieId")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .rename("n_ratings")
            .reset_index()
            .merge(catalog[["movieId", "clean_title", "titleType"]], on="movieId")
            .to_dict("records")
        ),
    }
    log(
        f"  dropped {len(tv_ids):,} TV titles and "
        f"{int(tv_rating_mask.sum()):,} ratings "
        f"({tv_rating_mask.mean() * 100:.2f}% of all ratings)"
    )
    return kept_catalog, kept_ratings


def clean_ratings(ratings: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, range-check, and drop ratings pointing at no catalog entry."""
    n_start = len(ratings)

    valid_ids = set(catalog["movieId"])
    orphan_mask = ~ratings["movieId"].isin(valid_ids)
    ratings = ratings[~orphan_mask]

    # A user should have at most one rating per movie. Where that is violated,
    # keep the most recent rating: it is the user's current opinion.
    ratings = ratings.sort_values("timestamp", kind="mergesort")
    dup_mask = ratings.duplicated(subset=["userId", "movieId"], keep="last")
    n_dups = int(dup_mask.sum())
    ratings = ratings[~dup_mask]

    bad_rating = ~ratings["rating"].between(0.5, 5.0)
    n_bad = int(bad_rating.sum())
    ratings = ratings[~bad_rating]

    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)

    per_user = ratings.groupby("userId").size()
    report["cleaning"] = {
        "n_ratings_in": int(n_start),
        "n_orphan_ratings_dropped": int(orphan_mask.sum()),
        "n_duplicate_user_movie_dropped": n_dups,
        "n_out_of_range_ratings_dropped": n_bad,
        "n_ratings_out": int(len(ratings)),
        "n_users_out": int(ratings["userId"].nunique()),
        "n_movies_out": int(ratings["movieId"].nunique()),
        # MovieLens guarantees >=20 ratings per user on the raw file; dropping TV
        # can push a user under that floor, which matters for personalization.
        "n_users_below_20_after_tv_drop": int((per_user < 20).sum()),
        "n_users_below_5_after_tv_drop": int((per_user < 5).sum()),
    }
    log(f"  cleaned to {len(ratings):,} ratings")
    return ratings


def chronological_split(ratings: pd.DataFrame) -> pd.DataFrame:
    """Label each rating train / validation / test by global timestamp quantile."""
    t = ratings["timestamp"].to_numpy()
    train_end = int(np.quantile(t, TRAIN_END_Q))
    val_end = int(np.quantile(t, VAL_END_Q))

    split = np.where(t <= train_end, "train", np.where(t <= val_end, "validation", "test"))
    ratings["split"] = pd.Categorical(split, categories=["train", "validation", "test"])

    train_users = set(ratings.loc[ratings["split"] == "train", "userId"])
    train_movies = set(ratings.loc[ratings["split"] == "train", "movieId"])

    summary = []
    for name, part in ratings.groupby("split", observed=True):
        cold_u = (~part["userId"].isin(train_users)).mean() if name != "train" else 0.0
        cold_m = (~part["movieId"].isin(train_movies)).mean() if name != "train" else 0.0
        summary.append(
            {
                "split": str(name),
                "n_ratings": int(len(part)),
                "share_of_ratings": float(len(part) / len(ratings)),
                "n_users": int(part["userId"].nunique()),
                "n_movies": int(part["movieId"].nunique()),
                "start": str(part["rated_at"].min()),
                "end": str(part["rated_at"].max()),
                "mean_rating": float(part["rating"].mean()),
                "pct_ratings_from_cold_users": float(cold_u * 100),
                "pct_ratings_on_cold_movies": float(cold_m * 100),
            }
        )

    report["split"] = {
        "method": "global chronological quantile split (70 / 15 / 15 by timestamp)",
        "train_end_utc": str(pd.to_datetime(train_end, unit="s", utc=True)),
        "val_end_utc": str(pd.to_datetime(val_end, unit="s", utc=True)),
        "train_end_timestamp": train_end,
        "val_end_timestamp": val_end,
        "splits": summary,
    }
    for s in summary:
        log(f"  {s['split']:<10} {s['n_ratings']:>10,}  {s['start'][:10]} -> {s['end'][:10]}")
    return ratings


# A personalized model needs history to personalize from. MovieLens users join
# over time, so most users appearing after the train cutoff are brand new and
# cannot be scored by any user-history model. Evaluation is therefore reported
# on an explicit warm cohort, and the cold remainder is reported separately
# rather than quietly averaged in.
MIN_TRAIN_RATINGS = 20
MIN_EVAL_RATINGS = 5


def build_eval_cohort(ratings: pd.DataFrame) -> pd.DataFrame:
    """Define and measure the warm-user cohort the D3 comparison will score."""
    train_counts = ratings[ratings["split"] == "train"].groupby("userId").size()
    warm_users = set(train_counts[train_counts >= MIN_TRAIN_RATINGS].index)

    rows = []
    for name in ("validation", "test"):
        part = ratings[ratings["split"] == name]
        warm = part[part["userId"].isin(warm_users)]
        counts = warm.groupby("userId").size()
        eligible = set(counts[counts >= MIN_EVAL_RATINGS].index)
        scored = warm[warm["userId"].isin(eligible)]

        rows.append(
            {
                "split": name,
                "n_users_total": int(part["userId"].nunique()),
                "n_users_warm": int(warm["userId"].nunique()),
                "n_users_scored": int(len(eligible)),
                "n_ratings_total": int(len(part)),
                "n_ratings_scored": int(len(scored)),
                "pct_ratings_scored": float(len(scored) / len(part) * 100),
                "mean_rating_scored": float(scored["rating"].mean()) if len(scored) else float("nan"),
                "mean_rating_unscored": float(
                    part[~part.index.isin(scored.index)]["rating"].mean()
                ),
            }
        )
        log(
            f"  {name:<11} scorable: {len(eligible):,} users / "
            f"{len(scored):,} ratings ({len(scored) / len(part) * 100:.1f}%)"
        )

    cohort = pd.DataFrame(rows)
    report["eval_cohort"] = {
        "definition": (
            f"users with >={MIN_TRAIN_RATINGS} ratings in train and "
            f">={MIN_EVAL_RATINGS} ratings in the evaluation split"
        ),
        "min_train_ratings": MIN_TRAIN_RATINGS,
        "min_eval_ratings": MIN_EVAL_RATINGS,
        "rows": rows,
    }
    return cohort


def write_splits(ratings: pd.DataFrame) -> None:
    """Persist train/validation to processed/, and seal test under reserved_test/."""
    cols = ["userId", "movieId", "rating", "timestamp"]

    for name in ("train", "validation"):
        part = ratings.loc[ratings["split"] == name, cols]
        out = PROCESSED_DIR / f"ratings_{name}.parquet"
        part.to_parquet(out, index=False)
        log(f"  wrote {out.name} ({len(part):,} rows)")

    test = ratings.loc[ratings["split"] == "test", cols]
    out = RESERVED_TEST_DIR / "ratings_test.parquet"
    test.to_parquet(out, index=False)

    seal = RESERVED_TEST_DIR / "SEALED.md"
    seal.write_text(
        "# Reserved final test data - do not open until D6\n\n"
        f"- Rows: {len(test):,}\n"
        f"- Covers: {report['split']['val_end_utc']} onward\n"
        f"- Created by `prepare_data.py` on the 70/15/15 chronological split.\n\n"
        "Model selection, hyperparameters, and threshold choices in D3-D5 use\n"
        "`data/processed/ratings_validation.parquet` only. This file is read once,\n"
        "in D6, after every modelling choice is frozen. Any earlier read must be\n"
        "disclosed in the final report.\n",
        encoding="utf-8",
    )
    log(f"  sealed {out.name} ({len(test):,} rows)")


def build_aggregates(ratings: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Small summary tables that back the Streamlit dashboard.

    The dashboard reads these from Supabase, never the 31M-row ratings table.
    """
    tables: dict[str, pd.DataFrame] = {}
    train = ratings[ratings["split"] == "train"]

    # 1. Movie dimension, with train-only popularity so nothing leaks from test.
    stats = train.groupby("movieId")["rating"].agg(["size", "mean"])
    stats.columns = ["n_ratings_train", "mean_rating_train"]
    movies_dim = (
        catalog[["movieId", "clean_title", "year", "genres", "titleType"]]
        .merge(stats, on="movieId", how="left")
        .rename(columns={"clean_title": "title"})
    )
    movies_dim["n_ratings_train"] = movies_dim["n_ratings_train"].fillna(0).astype("int64")
    movies_dim["year"] = movies_dim["year"].astype("Float64")
    tables["movies_dim"] = movies_dim

    # 2. Rating-value distribution per split (the half-star artifact).
    dist = (
        ratings.groupby(["split", "rating"], observed=True)
        .size()
        .rename("n_ratings")
        .reset_index()
    )
    dist["split"] = dist["split"].astype(str)
    tables["rating_value_distribution"] = dist

    # 2b. MovieLens only allowed whole stars until Feb 2003. Training data spans
    # both regimes, so the rating scale is not one consistent measurement.
    first_half = ratings.loc[
        ratings["rating"].isin([0.5, 1.5, 2.5, 3.5, 4.5]), "rated_at"
    ].min()
    era = np.where(train["rated_at"] < first_half, "Whole stars only", "Half stars allowed")
    era_dist = (
        pd.DataFrame({"era": era, "rating": train["rating"].to_numpy()})
        .groupby(["era", "rating"])
        .size()
        .rename("n_ratings")
        .reset_index()
    )
    era_dist["share_within_era"] = era_dist["n_ratings"] / era_dist.groupby("era")[
        "n_ratings"
    ].transform("sum")
    tables["rating_distribution_by_era"] = era_dist
    report["half_star_change"] = {
        "first_half_star_rating_utc": str(first_half),
        "share_half_star_after_change": float(
            train.loc[train["rated_at"] >= first_half, "rating"]
            .isin([0.5, 1.5, 2.5, 3.5, 4.5])
            .mean()
        ),
    }

    # 3. Monthly activity: volume and mean rating drift over 28 years.
    monthly = ratings.set_index("rated_at").groupby(pd.Grouper(freq="MS")).agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        n_users=("userId", "nunique"),
        n_movies=("movieId", "nunique"),
    ).reset_index().rename(columns={"rated_at": "month"})
    monthly["month"] = monthly["month"].dt.tz_localize(None)
    tables["monthly_activity"] = monthly

    # 4. Ratings-per-user distribution (repeated-entity / heavy-user concern).
    per_user = train.groupby("userId").size()
    edges = [0, 20, 50, 100, 250, 500, 1000, 5000, np.inf]
    labels = ["1-19", "20-49", "50-99", "100-249", "250-499", "500-999", "1k-5k", "5k+"]
    ub = pd.cut(per_user, bins=edges, labels=labels, right=False)
    user_bins = (
        pd.DataFrame({"bucket": ub, "n_ratings": per_user})
        .groupby("bucket", observed=False)
        .agg(n_users=("n_ratings", "size"), n_ratings=("n_ratings", "sum"))
        .reset_index()
    )
    user_bins["bucket"] = user_bins["bucket"].astype(str)
    tables["user_activity_bins"] = user_bins

    # 5. Ratings-per-movie distribution (long tail / cold-start concern).
    per_movie = train.groupby("movieId").size()
    m_edges = [0, 1, 5, 20, 100, 1000, 10000, np.inf]
    m_labels = ["0", "1-4", "5-19", "20-99", "100-999", "1k-10k", "10k+"]
    full = per_movie.reindex(catalog["movieId"], fill_value=0)
    mb = pd.cut(full, bins=m_edges, labels=m_labels, right=False)
    movie_bins = (
        pd.DataFrame({"bucket": mb, "n_ratings": full})
        .groupby("bucket", observed=False)
        .agg(n_movies=("n_ratings", "size"), n_ratings=("n_ratings", "sum"))
        .reset_index()
    )
    movie_bins["bucket"] = movie_bins["bucket"].astype(str)
    tables["movie_popularity_bins"] = movie_bins

    # 6. Genre summary (train only).
    exploded = catalog[["movieId", "genres"]].copy()
    exploded["genre"] = exploded["genres"].str.split("|")
    exploded = exploded.explode("genre")[["movieId", "genre"]]
    genre_join = train[["movieId", "rating"]].merge(exploded, on="movieId", how="inner")
    genre_stats = genre_join.groupby("genre").agg(
        n_ratings=("rating", "size"), mean_rating=("rating", "mean")
    ).reset_index()
    genre_counts = exploded.groupby("genre").size().rename("n_movies").reset_index()
    tables["genre_stats"] = genre_stats.merge(genre_counts, on="genre", how="outer").fillna(
        {"n_ratings": 0, "n_movies": 0}
    )

    # 7. Split summary, so the dashboard can state the reserved-test boundary.
    tables["split_summary"] = pd.DataFrame(report["split"]["splits"])
    tables["eval_cohort"] = pd.DataFrame(report["eval_cohort"]["rows"])

    # 8. Lorenz curve of popularity concentration (popularity-bias evidence).
    sorted_counts = np.sort(per_movie.to_numpy())[::-1]
    cum = np.cumsum(sorted_counts) / sorted_counts.sum()
    pct_movies = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    idx = np.unique(np.linspace(0, len(cum) - 1, 400).astype(int))
    tables["popularity_concentration"] = pd.DataFrame(
        {"pct_movies": pct_movies[idx], "pct_ratings": cum[idx]}
    )

    report["aggregates"] = {k: int(len(v)) for k, v in tables.items()}
    return tables


def main() -> None:
    ensure_dirs()
    catalog, ratings = load_raw()

    log("Dropping television titles...")
    catalog, ratings = drop_television(catalog, ratings)

    log("Cleaning ratings...")
    ratings = clean_ratings(ratings, catalog)

    log("Splitting chronologically...")
    ratings = chronological_split(ratings)

    log("Measuring the scorable evaluation cohort...")
    build_eval_cohort(ratings)

    write_splits(ratings)

    log("Building dashboard aggregates...")
    tables = build_aggregates(ratings, catalog)
    for name, df in tables.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        log(f"  {name}: {len(df):,} rows")

    catalog.to_parquet(PROCESSED_DIR / "catalog.parquet", index=False)

    out = DOCS_DIR / "prep_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"\nWrote {out}")


if __name__ == "__main__":
    main()
