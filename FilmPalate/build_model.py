"""Build the fitted pipeline and film catalog the FilmPalate API serves.

* The List taste model is fit with Ishika's tuned settings, except a popularity
  weight of 2 instead of 6 (more variety; see evaluate_popularity_weight.py), (``List taste model/
  results/list_model_results.json``) on train + validation ratings, i.e. every
  rating before the sealed test period (through 2019-11-02). Her evaluation used
  the training split only; refitting on all non-test data is the usual last step
  before serving, and leaves the reserved test set untouched.
* Films the model never saw (released 2017+, 10,000+ IMDb votes, fewer than 20
  MovieLens ratings or none at all) are added from their IMDb description, with the
  settings chosen in ``evaluate_new_films.py``.

Writes (gitignored, rebuild with this script, ~10 min):
    FilmPalate/api/artifacts/watchlist_pipeline.joblib
    FilmPalate/api/artifacts/catalog.parquet
    FilmPalate/api/artifacts/sample_predictions.json   (for local vs deployed checks)

Usage (repo root):
    uv run python "FilmPalate/build_model.py"
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "List taste model", ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

from dtsc_3602_final_project.paths import LINKS_CSV, PROCESSED_DIR, RAW_DIR  # noqa: E402
from watchlist.imdb import (load_directors, load_ratings, load_recent_films,  # noqa: E402
                            load_top_actors)
from watchlist.pipeline import WatchlistRanker, build_pipeline  # noqa: E402

ARTIFACTS = HERE / "api" / "artifacts"
IMDB_DIR = RAW_DIR.parent
NEW_FROM_YEAR = 2017
NEW_MIN_VOTES = 10_000
NEWEST_YEAR = 2026
IMDB_ID_OFFSET = 1_000_000_000   # IMDb-only films: id = offset + tconst number

ISHIKA = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                    .read_text())["selected"]
# Site choice: popularity weight 2 instead of Ishika's tuned 6 -- about twice the
# variety for a small precision cost (report-half check: evaluate_popularity_weight.py).
SITE_POPULARITY_WEIGHT = 2.0
# ...but users with few ratings need popularity to count much more (evaluate_blend.py,
# tuned per history size): (up to this many ratings, weight). Want to Watch counts 0.5.
POPULARITY_SCHEDULE = ((3, 40.0), (10, 25.0), (50, 15.0), (100, 8.0), (200, 4.0))
BACKTEST = HERE / "results" / "new_films_backtest.json"
NEW_SETTINGS = (json.loads(BACKTEST.read_text())["selected"] if BACKTEST.exists()
                else {"content_alpha": 10.0, "new_popularity_weight": 1.0})

STARTER_GENRES = ["Action", "Comedy", "Drama", "Animation", "Sci-Fi", "Romance",
                  "Horror", "Thriller", "Crime", "Adventure", "Fantasy", "Documentary"]


def imdb_only_films(mf: pd.DataFrame) -> pd.DataFrame:
    """2017+ IMDb films MovieLens doesn't have at all, in movie_features format."""
    films = load_recent_films(IMDB_DIR / "title.basics.tsv.gz", NEW_FROM_YEAR,
                              exclude=set(mf["tconst"].dropna()))
    ratings = load_ratings(IMDB_DIR / "title.ratings.tsv.gz", set(films["tconst"]))
    films = films.merge(ratings[ratings["numVotes"] >= NEW_MIN_VOTES], on="tconst")
    keep = set(films["tconst"])
    films = (films.merge(load_top_actors(IMDB_DIR / "title.principals.tsv.gz", keep),
                         on="tconst", how="left")
                  .merge(load_directors(IMDB_DIR / "title.crew.tsv.gz", keep),
                         on="tconst", how="left"))
    for col in ("actors", "directors"):
        films[col] = films[col].apply(lambda v: list(v) if isinstance(v, list) else [])
    films["movieId"] = IMDB_ID_OFFSET + films["tconst"].str[2:].astype("int64")
    films = films.rename(columns={"title": "clean_title"})
    return films[mf.columns].astype({"year": "float64"})


def starter_order(cat: pd.DataFrame) -> pd.Series:
    """Well-known films spread over many genres, for the first screen a new user sees."""
    known = cat[cat["in_model"] & (cat["year"] >= 1970)]
    picks: list[int] = []
    for _ in range(6):
        for g in STARTER_GENRES:
            top = known[known["genres"].str.contains(g, regex=False)].nlargest(40, "n_ratings")
            for mid in top["movieId"]:
                if mid not in picks:
                    picks.append(mid)
                    break
    return pd.Series(np.arange(1, len(picks) + 1), index=picks)


def main() -> None:
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    ratings = pd.concat([pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet"),
                         pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")])
    print(f"{len(ratings):,} ratings through 2019-11-02", flush=True)

    print("Loading 2017+ films from IMDb...", flush=True)
    extra = imdb_only_films(mf)
    films = pd.concat([mf, extra], ignore_index=True)
    new = films[(films["year"] >= NEW_FROM_YEAR) & (films["numVotes"] >= NEW_MIN_VOTES)]
    print(f"  {len(extra):,} IMDb-only films; {len(new):,} 2017+ candidates in total", flush=True)

    print(f"Fitting the List taste model (Ishika's settings, popularity weight "
          f"{SITE_POPULARITY_WEIGHT} instead of {ISHIKA['popularity_weight']}) "
          f"+ new films {NEW_SETTINGS}...", flush=True)
    ranker = WatchlistRanker(
        n_factors=ISHIKA["n_factors"], popularity_damping=ISHIKA["popularity_damping"],
        popularity_weight=SITE_POPULARITY_WEIGHT, popularity_schedule=POPULARITY_SCHEDULE,
        newest_year=NEWEST_YEAR,
        **NEW_SETTINGS).fit(ratings, films, new)
    pipe = build_pipeline(ranker)
    nb = ranker.n_base_
    print(f"  {nb:,} catalog films + {len(ranker.film_ids_) - nb:,} new films", flush=True)

    # Catalog the API serves: every film the pipeline can take as input.
    links = pd.read_csv(LINKS_CSV, usecols=["movieId", "tmdbId"])
    cat = (films.set_index("movieId").reindex(ranker.film_ids_).reset_index()
           [["movieId", "tconst", "clean_title", "year", "genres", "averageRating", "numVotes"]]
           .rename(columns={"clean_title": "title", "tconst": "imdb_id",
                            "averageRating": "imdb_rating", "numVotes": "imdb_votes"})
           .merge(links, on="movieId", how="left"))
    counts = ratings.groupby("movieId").size()
    cat["n_ratings"] = cat["movieId"].map(counts).fillna(0).astype(int)
    cat["in_model"] = np.arange(len(cat)) < nb
    cat["is_new"] = ~cat["in_model"]
    pop = pd.Series(ranker.base_.popularity_z_, index=ranker.base_.item_ids_)
    cat["popularity_z"] = cat["movieId"].map(pop)
    cat["starter_rank"] = cat["movieId"].map(starter_order(cat))
    cat["tmdbId"] = cat["tmdbId"].astype("Int64")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(ARTIFACTS / "catalog.parquet", index=False)
    metadata = {
        "site": "FilmPalate",
        "model": "List taste model (Ishika Bhattarai) in a WatchlistEncoder -> WatchlistRanker pipeline",
        "site_and_api": "Yager Range",
        "steps": [name for name, _ in pipe.steps],
        "lists": {"my_ratings": ">=3.5 stars +1.0, 2.5-3 +0.25, <=2 -1.0",
                  "want_to_watch": "+0.5"},
        "list_model_settings": {**ISHIKA, "popularity_weight": SITE_POPULARITY_WEIGHT,
                                "ishika_tuned_popularity_weight": ISHIKA["popularity_weight"],
                                "popularity_schedule": POPULARITY_SCHEDULE},
        "new_film_settings": NEW_SETTINGS,
        "ratings_through": "2019-11-02",
        "n_ratings": int(len(ratings)),
        "n_catalog_films": int(nb),
        "n_new_films": int(len(ranker.film_ids_) - nb),
        "new_films": f"released {NEW_FROM_YEAR}+ with {NEW_MIN_VOTES:,}+ IMDb votes",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump({"pipeline": pipe, "metadata": metadata},
                ARTIFACTS / "watchlist_pipeline.joblib", compress=3)
    size = (ARTIFACTS / "watchlist_pipeline.joblib").stat().st_size / 1e6
    print(f"  wrote watchlist_pipeline.joblib ({size:.0f} MB) and catalog.parquet", flush=True)

    # Sanity check + reference outputs for the local-vs-deployed comparison.
    title = cat.set_index("movieId")["title"]
    samples = {
        "empty": {"ratings": {}, "want_to_watch": []},
        "sci_fi_fan": {"ratings": {2571: 5.0, 1196: 4.5, 260: 5.0, 1721: 1.5},
                       "want_to_watch": [79132]},
        "only_want_to_watch": {"ratings": {}, "want_to_watch": [318, 858]},
    }
    out = {}
    for name, user in samples.items():
        top = pipe.predict([user])[0]
        H = pipe[:-1].transform([user])
        new_top = pipe[-1].predict_new(H)[0]
        out[name] = {"request": user, "top": top, "new": new_top}
        print(f"\n  {name}: {user}")
        print("    top: " + "; ".join(title.get(m, str(m)) for m in top))
        print("    new: " + "; ".join(title.get(m, str(m)) for m in new_top))
    (ARTIFACTS / "sample_predictions.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
