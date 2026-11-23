"""How much should popularity count for users with only a few ratings?

``evaluate_min_ratings.py`` showed that with popularity weight 2 (the site setting)
the List taste model loses to plain popular picks until a user has about 20
ratings. Instead of a hard switch, FilmPalate can lean on popularity for users with
few ratings and shift toward taste as ratings grow, using the model's own
``popularity_weight`` setting -- no new model parts.

For each history size N (each user's N most recent training ratings), this picks
the best weight on the tune half and reports it on the report half against
popular picks (group protocol; both skip films the user rated in training).

Usage (repo root, ~10 min):
    uv run python "FilmPalate/extras/evaluate_blend.py"
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
HISTORY_SIZES = [1, 2, 3, 5, 10, 20, 50]
WEIGHTS = [2.0, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0]
K, WINDOW = 10, 365


def main():
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    tune_users, report_users = ev.cohort(train, val)
    model = ListTasteRecommender(n_factors=ISHIKA["n_factors"],
                                 popularity_damping=ISHIKA["popularity_damping"],
                                 window_days=WINDOW).fit(train)
    baseline = PopularityRecommender(window_days=WINDOW).fit(train)

    def setup(users):
        mine = train[train["userId"].isin(users)].sort_values(["userId", "timestamp"])
        seen = model.history_matrix(mine, users).nonzero()
        recs_b = baseline.recommend(baseline.history_matrix(mine, users), K)
        return mine, seen, recs_b, ev.per_user_metrics(recs_b, users, val)["precision"]

    halves = {"tune": (tune_users, *setup(tune_users)), "report": (report_users, *setup(report_users))}

    def precision(half, n, w):
        users, mine, seen, recs_b, _ = halves[half]
        H = model.history_matrix(mine.groupby("userId").tail(n), users)  # list weights
        positive = np.asarray(H.maximum(0).sum(axis=1)).ravel() > 0
        model.popularity_weight = w
        s = model.score(H).astype(np.float32)
        s[seen] = -np.inf
        top = np.argpartition(-s, K, axis=1)[:, :K]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)
        recs = model.item_ids_[top]
        recs[~positive] = recs_b[~positive]
        return ev.per_user_metrics(recs, users, val)["precision"]

    out = {"weights_tried": WEIGHTS, "by_history_size": []}
    for n in HISTORY_SIZES:
        tune = {w: float(precision("tune", n, w).mean()) for w in WEIGHTS}
        best = max(tune, key=tune.get)
        rep, pop = precision("report", n, best), halves["report"][4]
        site = precision("report", n, 2.0)
        diff = ev.bootstrap_diff(rep, pop)
        out["by_history_size"].append({
            "history_size": n, "tune_p10_by_weight": tune, "best_weight": best,
            "report_p10_best": float(rep.mean()), "report_p10_weight2": float(site.mean()),
            "report_p10_popular": float(pop.mean()), "best_minus_popular": diff})
        print(f"N={n:>2}: best weight {best:>4} -> {rep.mean():.4f} vs popular {pop.mean():.4f} "
              f"diff {diff['diff']:+.4f} [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}]  "
              f"(weight 2: {site.mean():.4f})", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "blend_check.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
