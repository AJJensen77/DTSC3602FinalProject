"""When should FilmPalate switch from popular picks to personal picks?

The site switches as soon as a user's lists contain anything positive. This checks
whether one rating is already enough: each report-half user (group protocol, train
split only) is given only their N most recent training ratings, and the List taste
model (site settings, popularity weight 2) is compared with the popularity baseline
on the validation period (hit = rated >= 3.5), for N = 1, 2, 3, 5, 10, 20.

As on the site, a user whose N ratings are all <= 2 stars (nothing positive) gets
the popular picks. Want to Watch cannot be tested: MovieLens has no such data.

Usage (repo root, ~5 min):
    uv run python "FilmPalate/extras/evaluate_min_ratings.py"
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]          # repo root
for p in (HERE.parent / "D4_submission", ROOT / "List taste model", ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402
from listmodel.recommender import ListTasteRecommender  # noqa: E402

ISHIKA = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                    .read_text())["selected"]
SITE_WEIGHT = 2.0
HISTORY_SIZES = [1, 2, 3, 5, 10, 20]
K, WINDOW = 10, 365


def main():
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    _, users = ev.cohort(train, val)
    print(f"report half: {len(users):,} users (all have >= 20 training ratings)", flush=True)

    model = ListTasteRecommender(n_factors=ISHIKA["n_factors"],
                                 popularity_damping=ISHIKA["popularity_damping"],
                                 popularity_weight=SITE_WEIGHT, window_days=WINDOW).fit(train)
    baseline = PopularityRecommender(window_days=WINDOW).fit(train)
    mine = train[train["userId"].isin(users)].sort_values(["userId", "timestamp"])

    # Both methods skip every film the user rated in training (they've seen it), so
    # the comparison is about which *unseen* films to suggest from N known ratings.
    full = model.history_matrix(mine, users)
    seen_rows, seen_cols = full.nonzero()
    recs_b = baseline.recommend(baseline.history_matrix(mine, users), K)
    gb = ev.per_user_metrics(recs_b, users, val)["precision"]

    out = {"n_users": int(len(users)), "popular_p10": float(gb.mean()), "by_history_size": []}
    for n in HISTORY_SIZES:
        hist = mine.groupby("userId").tail(n)
        # history_matrix already turns stars into list weights (+1 / +0.25 / -1).
        Hm = model.history_matrix(hist, users)
        positive = np.asarray(Hm.maximum(0).sum(axis=1)).ravel() > 0
        s = model.score(Hm).astype(np.float32)
        s[seen_rows, seen_cols] = -np.inf
        top = np.argpartition(-s, K, axis=1)[:, :K]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)
        recs_m = model.item_ids_[top]
        recs_m[~positive] = recs_b[~positive]          # site falls back to popular
        gm = ev.per_user_metrics(recs_m, users, val)["precision"]
        diff = ev.bootstrap_diff(gm, gb)
        row = {"history_size": n, "model_p10": float(gm.mean()), "popular_p10": float(gb.mean()),
               "share_with_positive_history": float(positive.mean()), "difference": diff}
        out["by_history_size"].append(row)
        print(f"N={n:>2}: model {gm.mean():.4f} vs popular {gb.mean():.4f}  "
              f"diff {diff['diff']:+.4f} [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}]  "
              f"({positive.mean():.0%} had a positive rating)", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "min_ratings_check.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
