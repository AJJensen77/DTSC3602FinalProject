"""Build the model and movie catalog the website's API serves.

The site serves the model that was evaluated: the regressor is fit exactly as in
evaluate_hybrid.py (2014-10 -> 2016-10 targets, same seed and settings), so the
validation results describe it. Only its population state - movie statistics,
popularity, similarities - is then refreshed with train + validation (through
2019-11-02), the same refresh step the evaluation itself used. Refitting the
regressor on a later window was tried and gave noticeably flatter, less
personal predictions that cannot be checked without opening the sealed test set.

Writes (gitignored, rebuild with this script):
    api/artifacts/hybrid_bundle.joblib   model + metadata
    api/artifacts/catalog.parquet        films for search, the starter grid, and display

Usage:
    uv run python "Gradient boost model/build_site_model.py"
"""

import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import evaluate_hybrid as ev
from gbmodel.hybrid import HybridRecommender
from dtsc_3602_final_project.paths import LINKS_CSV, PROCESSED_DIR

ARTIFACTS = Path(__file__).parent / "api" / "artifacts"

# Tuned on validation in evaluate_hybrid.py (see docs/hybrid_results.json).
SETTINGS = dict(regressor_params=ev.HGB_GRID[1], min_votes=100_000,
                popularity_weight=0.7, popularity_source="recent")

STARTER_GENRES = ["Action", "Comedy", "Drama", "Animation", "Sci-Fi", "Romance",
                  "Horror", "Thriller", "Crime", "Adventure", "Fantasy", "Documentary"]
STARTER_PER_GENRE = 6


def starter_order(cat: pd.DataFrame) -> pd.Series:
    """Rank well-known films so the first screen spans many genres, not just blockbusters."""
    known = cat[(cat["n_ratings"] >= 1000) & (cat["year"] >= 1970)]
    picks: list[int] = []
    for i in range(STARTER_PER_GENRE):
        for g in STARTER_GENRES:
            top = known[known["genres"].str.contains(g, regex=False)].nlargest(40, "n_ratings")
            for mid in top["movieId"]:
                if mid not in picks:
                    picks.append(mid)
                    break
    return pd.Series(np.arange(1, len(picks) + 1), index=picks)


def main() -> None:
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    ratings = pd.concat([train, pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")])

    print("Fitting the evaluated regressor (2014-10 -> 2016-10 targets)...", flush=True)
    hist, tgt = ev.training_rows(train, ev.MAX_TRAIN_USERS)
    model = HybridRecommender(mf, **SETTINGS).fit(hist, tgt)
    print(f"Refreshing movie statistics with {len(ratings):,} ratings through 2019-11-02...",
          flush=True)
    model.refresh(ratings)
    print(f"  fit on {model.n_train_rows_:,} rows, {model.regressor_.n_iter_} trees", flush=True)

    p = model.profile_
    links = pd.read_csv(LINKS_CSV, usecols=["movieId", "tmdbId"])
    cat = mf[["movieId", "clean_title", "year", "genres", "averageRating", "numVotes"]].rename(
        columns={"clean_title": "title", "averageRating": "imdb_rating", "numVotes": "imdb_votes"})
    cat = cat.merge(links, on="movieId", how="left")
    cat["n_ratings"] = p.count_.astype(int)
    cat["recent_ratings"] = p.recent_count_.astype(int)
    recommendable = np.zeros(len(cat), dtype=bool)
    recommendable[p.eligible_[model.candidate_mask()]] = True
    cat["recommendable"] = recommendable
    cat["starter_rank"] = cat["movieId"].map(starter_order(cat))
    cat["tmdbId"] = cat["tmdbId"].astype("Int64")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(ARTIFACTS / "catalog.parquet", index=False)
    bundle = {
        "model": model,
        "metadata": {
            "model": "Hybrid taste-profile rating predictor (HistGradientBoostingRegressor)",
            "author": "Yager Range",
            "regressor_trained_on": "ratings 2014-10-01 to 2016-10-01 (features from earlier)",
            "movie_stats_through": "2019-11-02",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_training_rows": int(model.n_train_rows_),
            "n_trees": int(model.regressor_.n_iter_),
            "features": p.feature_names_,
            "settings": {k: v for k, v in SETTINGS.items()},
            "like_threshold": model.min_predicted,
            "n_films_in_catalog": int(len(cat)),
            "n_films_recommendable": int(recommendable.sum()),
            "genres": model.genres_,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
        },
    }
    joblib.dump(bundle, ARTIFACTS / "hybrid_bundle.joblib", compress=3)
    size = (ARTIFACTS / "hybrid_bundle.joblib").stat().st_size / 1e6
    print(f"  wrote hybrid_bundle.joblib ({size:.0f} MB) and catalog.parquet "
          f"({recommendable.sum():,} recommendable films)")

    # Sanity check: user 5778 from the write-up (crime-drama fan), all films then Horror only.
    t = ratings[ratings["userId"] == 5778]
    t = t[t["timestamp"] < pd.Timestamp("2016-10-01", tz="UTC").timestamp()]
    titles = cat.set_index("movieId")
    for genres in (None, ["Horror"]):
        recs = model.recommend_from_ratings(zip(t["movieId"], t["rating"]), genres=genres)
        print(f"\n  user 5778, genres={genres}:")
        for mid, stars in recs:
            print(f"    {stars:.2f}  {titles.at[mid, 'title']} ({titles.at[mid, 'genres']})")


if __name__ == "__main__":
    main()
