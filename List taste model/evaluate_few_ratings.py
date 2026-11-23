"""How many movies does a user need before the list taste model beats popularity?

Takes the same 2,953 report-half users as the main evaluation, keeps only each user's
N most recent training ratings (N = 1, 2, 3, 5, 10, 15, 20, all), and scores both the
saved list taste model and the popularity baseline from that cut-down history. Both
models see exactly the same history, and both skip films in it. Validation ratings
(Oct 2016 - Nov 2019) are the answer key, same as always. The reserved test set is not read.

Caveat: the saved model's movie factors were learned from all training users, including
these users' full histories. With ~200k training users, one user's ratings barely move
the factors, but this is a "short history at prediction time" test, not a true cold start.

Writes results/few_ratings.csv and figures/fig14_few_ratings.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Gradient boost model"))

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402

K = ev.K
SIZES = [1, 2, 3, 5, 10, 15, 20, None]  # None = full training history
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    _, users = ev.cohort(train, val)
    log(f"report users: {len(users):,}")

    model = joblib.load(HERE / "artifacts" / "list_model.joblib")["model"]
    baseline = PopularityRecommender(window_days=365).fit(train)
    log("baseline fitted")

    # Report users' training ratings, newest first (ties broken by movieId for stability).
    hist = train[train["userId"].isin(users)]
    del train
    hist = hist.sort_values(["userId", "timestamp", "movieId"], ascending=[True, False, True])
    hist["recency_rank"] = hist.groupby("userId").cumcount()

    rows = []
    for n in SIZES:
        h = hist if n is None else hist[hist["recency_rank"] < n]
        l_recs = model.recommend(model.history_matrix(h, users), K)
        b_recs = baseline.recommend(baseline.history_matrix(h, users), K)
        gl = ev.per_user_metrics(l_recs, users, val)
        gb = ev.per_user_metrics(b_recs, users, val)
        ci = ev.bootstrap_diff(gl["precision"], gb["precision"])
        row = {
            "movies_known": "all" if n is None else n,
            "list_precision_at_10": gl["precision"].mean(),
            "baseline_precision_at_10": gb["precision"].mean(),
            "diff": ci["diff"], "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"],
            "list_hit_rate": gl["hit_rate"].mean(),
            "baseline_hit_rate": gb["hit_rate"].mean(),
            "users_list_better": float((gl["precision"] > gb["precision"]).mean()),
            "users_baseline_better": float((gl["precision"] < gb["precision"]).mean()),
        }
        rows.append(row)
        log(f"N={row['movies_known']}: list {row['list_precision_at_10']:.3f} vs "
            f"popularity {row['baseline_precision_at_10']:.3f} "
            f"(diff {ci['diff']:+.3f}, CI {ci['ci95_low']:+.3f} to {ci['ci95_high']:+.3f})")

    out = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "few_ratings.csv", index=False)

    # Figure: precision@10 vs number of movies known.
    labels = [str(x) for x in out["movies_known"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, out["baseline_precision_at_10"], marker="o", color="#9a9a9a", label="Popularity baseline")
    ax.plot(x, out["list_precision_at_10"], marker="o", color="#2b6cb0", label="List taste model")
    lo = out["baseline_precision_at_10"] + out["ci95_low"]
    hi = out["baseline_precision_at_10"] + out["ci95_high"]
    ax.fill_between(x, lo, hi, color="#2b6cb0", alpha=0.15, label="95% CI of list model's lead")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Movies the model knows about (user's most recent ratings)")
    ax.set_ylabel("Precision@10")
    ax.set_title("How many movies before the list model beats popularity?")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig14_few_ratings.png", dpi=150)
    log("wrote results/few_ratings.csv and figures/fig14_few_ratings.png")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
