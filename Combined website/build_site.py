"""Build the combined model the website API serves, including films from 2017 to today.

Both models keep the settings they were evaluated with; only their movie statistics are
refreshed with train + validation (ratings through 2019-11), the same refresh step
Yager's site builds use.

* Gradient boost: the evaluated regressor (artifacts/gb_eval.joblib from
  evaluate_combined.py), movie statistics refreshed with train + validation.
* List taste model: refit with its selected settings on train + validation, popularity
  weight from results/site_settings.json (audit_bias.py: 2).
* Newer films: Yager's ``WatchlistRanker`` (FilmPalate/D4_submission/watchlist) wraps
  that List taste model and gives each film released 2017+ (with 1,000+ IMDb votes) that
  the model doesn't know a borrowed fingerprint from its 10 closest films by IMDb
  description. Films with 10,000+ votes (Yager's tested cut-off) can be recommended in
  the "New releases for you" row. See combined/site.py.
* Posters: films MovieLens doesn't have get their TMDB id from TMDB's /find by IMDb id
  (needs TMDB_API_KEY in the environment or in web/.env.local; cached in artifacts/).

Writes (gitignored):
    api/artifacts/site_bundle.joblib   site model + metadata
    api/artifacts/catalog.parquet      films for search, the starter grid and display

Usage (repo root, after evaluate_combined.py and audit_bias.py; IMDb files in data/raw/):
    uv run python "Combined website/build_site.py"
"""

from __future__ import annotations

import json
import os
import platform
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "FilmPalate" / "D4_submission", ROOT / "List taste model",
          ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

from build_site_model import starter_order  # noqa: E402  (Yager's starter grid)
from combined.blend import CombinedRecommender  # noqa: E402
from combined.site import SiteRecommender  # noqa: E402
from dtsc_3602_final_project.paths import LINKS_CSV, PROCESSED_DIR, RAW_DIR  # noqa: E402
from watchlist.imdb import (load_directors, load_ratings, load_recent_films,  # noqa: E402
                            load_top_actors)
from watchlist.pipeline import WatchlistRanker  # noqa: E402

OUT = HERE / "api" / "artifacts"
IMDB_DIR = RAW_DIR.parent
IMDB_ID_OFFSET = 1_000_000_000   # same ids as FilmPalate: offset + IMDb tconst number
NEW_FROM_YEAR = 2017
NEW_MIN_VOTES = 1_000            # searchable, listable, has a borrowed fingerprint
RECOMMEND_MIN_VOTES = 10_000     # can be recommended (Yager's tested cut-off)
NEWEST_YEAR = 2026
NEW_TASTE_WEIGHT, NEW_RATING_WEIGHT = 2.0, 0.25  # see README: list-following over backtest precision


def imdb_only_films(mf: pd.DataFrame) -> pd.DataFrame:
    """2017+ IMDb feature films MovieLens doesn't have, in movie_features format."""
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


def tmdb_key() -> str | None:
    key = os.environ.get("TMDB_API_KEY")
    env = HERE / "web" / ".env.local"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TMDB_API_KEY="):
                key = line.split("=", 1)[1].strip()
    return key or None


def tmdb_ids_for(tconsts: list[str]) -> dict[str, int]:
    """IMDb id -> TMDB movie id via TMDB /find (cached between builds)."""
    cache_path = HERE / "artifacts" / "tmdb_from_imdb.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [t for t in tconsts if t not in cache]
    key = tmdb_key()
    if todo and key:
        print(f"  looking up {len(todo):,} TMDB ids...", flush=True)

        def find(t):
            req = urllib.request.Request(
                f"https://api.themoviedb.org/3/find/{t}?external_source=imdb_id",
                headers={"Authorization": f"Bearer {key}"})
            for _ in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=20) as r:
                        hits = json.load(r).get("movie_results") or []
                    return t, (hits[0]["id"] if hits else None)
                except Exception:  # noqa: BLE001 - retry, then give up on this film
                    continue
            return t, "error"

        with ThreadPoolExecutor(8) as pool:
            for t, tid in pool.map(find, todo):
                if tid != "error":
                    cache[t] = tid
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    elif todo:
        print("  no TMDB_API_KEY: newer IMDb-only films will have no posters", flush=True)
    return {t: int(v) for t, v in cache.items() if v}


def main() -> None:
    settings = json.loads((HERE / "results" / "site_settings.json").read_text())
    list_sel = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                          .read_text())["selected"]
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    ratings = pd.concat([pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet"),
                         pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")],
                        ignore_index=True)
    print(f"{len(ratings):,} ratings through 2019-11", flush=True)

    print(f"loading {NEW_FROM_YEAR}+ films from IMDb...", flush=True)
    extra = imdb_only_films(mf)
    films = pd.concat([mf, extra], ignore_index=True)
    new = films[(films["year"] >= NEW_FROM_YEAR) & (films["numVotes"] >= NEW_MIN_VOTES)]
    print(f"  {len(extra):,} IMDb-only films; {len(new):,} {NEW_FROM_YEAR}+ films", flush=True)

    print("fitting the List taste model + newer films (WatchlistRanker)...", flush=True)
    ranker = WatchlistRanker(
        n_factors=list_sel["n_factors"], popularity_damping=list_sel["popularity_damping"],
        popularity_weight=settings["list_popularity_weight"], popularity_schedule=None,
        new_taste_weight=NEW_TASTE_WEIGHT, new_rating_weight=NEW_RATING_WEIGHT,
        newest_year=NEWEST_YEAR).fit(ratings, films, new)
    lm = ranker.base_
    nb = ranker.n_base_
    new_ids = ranker.film_ids_[nb:]
    votes = films.set_index("movieId")["numVotes"]
    new_recommendable = votes.reindex(new_ids).fillna(0).to_numpy() >= RECOMMEND_MIN_VOTES
    print(f"  {nb:,} catalog films + {len(new_ids):,} newer films "
          f"({new_recommendable.sum():,} recommendable)", flush=True)

    print("refreshing the gradient boost model's movie statistics...", flush=True)
    gb = joblib.load(HERE / "artifacts" / "gb_eval.joblib")
    gb.refresh(ratings)
    gb.max_year = None  # the evaluation's "no films after 2016" rule doesn't apply live
    comb = CombinedRecommender(lm, gb, gb_weight=settings["gb_weight"],
                               demote_below_like=settings["demote_below_like"],
                               min_ratings_for_gb=settings["min_ratings_for_gb"])
    site = SiteRecommender(comb, ranker, new_recommendable)

    # Catalog: every film the site can show (MovieLens films + IMDb-only newer films).
    p = gb.profile_
    links = pd.read_csv(LINKS_CSV, usecols=["movieId", "tmdbId"])
    cat = films[["movieId", "tconst", "clean_title", "year", "genres", "averageRating",
                 "numVotes"]].rename(columns={"clean_title": "title", "tconst": "imdb_id",
                                              "averageRating": "imdb_rating",
                                              "numVotes": "imdb_votes"})
    cat = cat.merge(links, on="movieId", how="left")
    found = tmdb_ids_for(cat.loc[cat["tmdbId"].isna() & cat["movieId"].ge(IMDB_ID_OFFSET),
                                 "imdb_id"].tolist())
    cat["tmdbId"] = cat["tmdbId"].fillna(cat["imdb_id"].map(found))
    counts = pd.Series(p.count_, index=p.movie_ids_)
    recent = pd.Series(p.recent_count_, index=p.movie_ids_)
    cat["n_ratings"] = cat["movieId"].map(counts).fillna(0).astype(int)
    cat["recent_ratings"] = cat["movieId"].map(recent).fillna(0).astype(int)
    cat["in_list_model"] = cat["movieId"].isin(ranker.film_ids_)  # has "More like this"
    cat["is_new"] = cat["movieId"].isin(new_ids)
    cat["starter_rank"] = cat["movieId"].map(starter_order(cat))
    cat["tmdbId"] = cat["tmdbId"].astype("Int64")

    OUT.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(OUT / "catalog.parquet", index=False)
    bundle = {
        "site": site,
        "metadata": {
            "model": ("Combined: list taste model candidates + gradient boost predicted stars; "
                      f"newer films ({NEW_FROM_YEAR}-{NEWEST_YEAR}) via FilmPalate's WatchlistRanker"),
            "authors": "Ishika Bhattarai (list taste model), Yager Range (gradient boost, "
                       "WatchlistRanker)",
            "blend": {"n_candidates": comb.n_candidates, "gb_weight": comb.gb_weight,
                      "demote_below_like": comb.demote_below_like,
                      "min_ratings_for_gb": comb.min_ratings_for_gb},
            "list_popularity_weight": settings["list_popularity_weight"],
            "new_films": {"from_year": NEW_FROM_YEAR, "min_votes": NEW_MIN_VOTES,
                          "recommend_min_votes": RECOMMEND_MIN_VOTES,
                          "taste_weight": NEW_TASTE_WEIGHT, "rating_weight": NEW_RATING_WEIGHT,
                          "n_new": int(len(new_ids)),
                          "n_recommendable": int(new_recommendable.sum())},
            "movie_stats_through": "2019-11-02",
            "imdb_snapshot": "2026-11-23",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_films_in_catalog": int(len(cat)),
            "n_films_list_model": int(nb),
            "genres": gb.genres_,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
        },
    }
    joblib.dump(bundle, OUT / "site_bundle.joblib", compress=3)
    size = (OUT / "site_bundle.joblib").stat().st_size / 1e6
    print(f"wrote site_bundle.joblib ({size:.0f} MB) and catalog.parquet ({len(cat):,} films, "
          f"{cat['tmdbId'].notna().mean():.0%} with posters)")

    # Sanity check: a sci-fi fan.
    titles = cat.set_index("movieId")["title"]
    demo = {2571: 5.0, 260: 4.5, 1196: 4.5, 1270: 4.0, 541: 4.5, 1240: 4.0}
    picks, newer = site.recommend(demo, plan_to_watch=[79132], k=10, k_new=10)
    for label, rows in (("picks", picks), ("new releases", newer)):
        print(f"  {label}:")
        for r in rows:
            print(f"    {titles.get(r['movieId'])}  pred={r['predicted_rating']}  "
                  f"because={titles.get(r['because'])}")


if __name__ == "__main__":
    main()
