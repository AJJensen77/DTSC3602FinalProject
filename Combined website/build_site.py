"""Build the combined model the website API serves.

Both models keep the settings they were evaluated with; only their movie statistics are
refreshed with train + validation (ratings through 2019-11), so newer films can be found.
This is the same refresh step Yager's site build uses.

* Gradient boost: the evaluated regressor (artifacts/gb_eval.joblib from
  evaluate_combined.py), movie statistics refreshed with train + validation.
* List taste model: refit with its selected settings on train + validation. It is a
  matrix factorization with no learned weights besides the factors, so "refreshing"
  it means refitting the factors.
* Blend settings: results/site_settings.json from audit_bias.py (tuned on the tune half,
  then the least popularity-driven option within 0.005 precision of the most accurate).

Writes (gitignored):
    api/artifacts/site_bundle.joblib   combined model + metadata
    api/artifacts/catalog.parquet      films for search, the starter grid and display

Usage (repo root, after evaluate_combined.py and audit_bias.py):
    uv run python "Combined website/build_site.py"
"""

from __future__ import annotations

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

from build_site_model import starter_order  # noqa: E402  (Yager's starter grid)
from combined.blend import CombinedRecommender  # noqa: E402
from dtsc_3602_final_project.paths import LINKS_CSV, PROCESSED_DIR  # noqa: E402
from listmodel.recommender import ListTasteRecommender  # noqa: E402

OUT = HERE / "api" / "artifacts"


def main() -> None:
    # Settings chosen by audit_bias.py (accuracy + popularity-bias check).
    blend = json.loads((HERE / "results" / "site_settings.json").read_text())
    list_sel = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                          .read_text())["selected"]
    list_sel = {k: list_sel[k] for k in ("popularity_damping", "n_factors", "min_mean",
                                         "popularity_weight")}
    list_sel["popularity_weight"] = blend["list_popularity_weight"]
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    ratings = pd.concat([pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet"),
                         pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")],
                        ignore_index=True)
    print(f"{len(ratings):,} ratings through 2019-11", flush=True)

    print(f"refitting the list taste model {list_sel}...", flush=True)
    lm = ListTasteRecommender(window_days=365, **list_sel).fit(ratings)
    print("refreshing the gradient boost model's movie statistics...", flush=True)
    gb = joblib.load(HERE / "artifacts" / "gb_eval.joblib")
    gb.refresh(ratings)
    gb.max_year = None  # the evaluation's "no films after 2016" rule doesn't apply live
    comb = CombinedRecommender(lm, gb, gb_weight=blend["gb_weight"],
                               demote_below_like=blend["demote_below_like"],
                               min_ratings_for_gb=blend["min_ratings_for_gb"])

    # Catalog: every film either model knows, with display fields (same columns as Yager's).
    p = gb.profile_
    links = pd.read_csv(LINKS_CSV, usecols=["movieId", "tmdbId"])
    cat = mf[["movieId", "clean_title", "year", "genres", "averageRating", "numVotes"]].rename(
        columns={"clean_title": "title", "averageRating": "imdb_rating", "numVotes": "imdb_votes"})
    cat = cat.merge(links, on="movieId", how="left")
    cat["n_ratings"] = p.count_.astype(int)
    cat["recent_ratings"] = p.recent_count_.astype(int)
    cat["in_list_model"] = cat["movieId"].isin(lm.item_ids_)
    cat["starter_rank"] = cat["movieId"].map(starter_order(cat))
    cat["tmdbId"] = cat["tmdbId"].astype("Int64")

    OUT.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(OUT / "catalog.parquet", index=False)
    bundle = {
        "combined": comb,
        "metadata": {
            "model": "Combined: list taste model candidates + gradient boost predicted stars",
            "authors": "Ishika Bhattarai (list taste model), Yager Range (gradient boost)",
            "blend": {"n_candidates": comb.n_candidates, "gb_weight": comb.gb_weight,
                      "demote_below_like": comb.demote_below_like,
                      "min_ratings_for_gb": comb.min_ratings_for_gb},
            "list_model_settings": list_sel,
            "movie_stats_through": "2019-11-02",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_films_in_catalog": int(len(cat)),
            "n_films_list_model": int(len(lm.item_ids_)),
            "genres": gb.genres_,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
        },
    }
    joblib.dump(bundle, OUT / "site_bundle.joblib", compress=3)
    size = (OUT / "site_bundle.joblib").stat().st_size / 1e6
    print(f"wrote site_bundle.joblib ({size:.0f} MB) and catalog.parquet ({len(cat):,} films)")

    # Sanity check: a sci-fi fan.
    titles = cat.set_index("movieId")["title"]
    demo = {2571: 5.0, 260: 4.5, 1196: 4.5, 1270: 4.0, 541: 4.5, 1240: 4.0}  # Matrix, Star Wars...
    for r in comb.recommend_one(demo, plan_to_watch=[79132], k=10):
        print(f"  {titles.get(r['movieId'])}  pred={r['predicted_rating']}  "
              f"because={titles.get(r['because'])}")


if __name__ == "__main__":
    main()
