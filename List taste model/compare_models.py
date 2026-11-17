"""Three-way comparison: popularity baseline vs gradient boost model vs list taste model.

All three were scored on the same 2,953 report-half users with the same metric code.
The list model and the baseline are re-scored here per user (paired bootstrap CIs).
The gradient boost model's per-user results were not saved, so it enters through its
published aggregates (``Gradient boost model/results/``); the list-vs-gradient-boost
interval is therefore an *unpaired* normal approximation using the larger of the two
per-user spreads we can measure. Ignoring the positive pairing makes it wider (more
conservative) than a paired interval would be.

Usage (from the repo root, after evaluate_list_model.py):
    uv run python "List taste model/compare_models.py"
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GB_DIR = HERE.parent / "Gradient boost model"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GB_DIR))

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from dtsc_3602_final_project import viz  # noqa: E402
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402

RESULTS, FIGURES = HERE / "results", HERE / "figures"
METRICS = [("precision", "precision_at_10", "Precision@10"), ("recall", "recall_at_10", "Recall@10"),
           ("ndcg", "ndcg_at_10", "NDCG@10"), ("hit_rate", "hit_rate_at_10", "Hit rate@10")]
EXTRA = [("liked_share_of_rated_recs", "Liked share of picks the user watched"),
         ("distinct_films_recommended", "Distinct films recommended"),
         ("share_recs_top1pct_films", "Share of picks from top-1% films"),
         ("median_train_ratings_of_recs", "Median training ratings of a pick")]
NAMES = {"baseline": "Popularity baseline", "gradient_boost": "Gradient boost model",
         "list_model": "List taste model"}
COLORS = {"baseline": viz.SERIES_2, "gradient_boost": viz.SERIES_1, "list_model": viz.SERIES_3}


def main() -> None:
    model = joblib.load(HERE / "artifacts" / "list_model.joblib")["model"]
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    popularity = train.groupby("movieId").size()
    _, users = ev.cohort(train, val)

    l_recs = model.recommend(model.history_matrix(train, users), ev.K)
    base = PopularityRecommender(window_days=365).fit(train)
    b_recs = base.recommend(base.history_matrix(train, users), ev.K)
    gl, gb = ev.per_user_metrics(l_recs, users, val), ev.per_user_metrics(b_recs, users, val)

    gb_res = json.loads((GB_DIR / "results" / "hybrid_results.json").read_text())["report"]
    agg = {"baseline": ev.summarize(gb, b_recs, popularity),
           "gradient_boost": gb_res["hybrid"],
           "list_model": ev.summarize(gl, l_recs, popularity)}
    # Same users + same code => the baseline must reproduce the published number exactly.
    assert abs(agg["baseline"]["precision_at_10"] - gb_res["baseline"]["precision_at_10"]) < 1e-9

    rows = []
    for short, key, label in METRICS:
        lm, hm, bm = (agg[m][key] for m in ("list_model", "gradient_boost", "baseline"))
        paired = ev.bootstrap_diff(gl[short], gb[short])
        sd = max(gl[short].std(), gb[short].std())
        n = gl[short].notna().sum()
        se = sd * np.sqrt(2 / n)
        rows.append({"metric": label, "baseline": bm, "gradient_boost": hm, "list_model": lm,
                     "list_minus_baseline": paired["diff"],
                     "list_minus_baseline_ci95": f"{paired['ci95_low']:+.3f} to {paired['ci95_high']:+.3f} (paired bootstrap)",
                     "list_minus_gradient_boost": lm - hm,
                     "list_minus_gradient_boost_ci95": f"{lm - hm - 1.96 * se:+.3f} to {lm - hm + 1.96 * se:+.3f} (unpaired approx.)",
                     "gradient_boost_minus_baseline_ci95": "{:+.3f} to {:+.3f} (paired, published)".format(
                         gb_res["difference_hybrid_minus_baseline"][short]["ci95_low"],
                         gb_res["difference_hybrid_minus_baseline"][short]["ci95_high"])})
    for key, label in EXTRA:
        rows.append({"metric": label, **{m: agg[m][key] for m in NAMES}})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "three_way_comparison.csv", index=False)

    # By user group: same bucket definitions as evaluate_hybrid.py.
    mine = pd.read_csv(RESULTS / "list_model_error_by_user_group.csv")
    theirs = pd.read_csv(GB_DIR / "results" / "hybrid_error_by_user_group.csv")
    groups = mine.merge(theirs[["group", "bucket", "n_users", "hybrid_precision"]],
                        on=["group", "bucket", "n_users"], how="inner")
    groups = groups[["group", "bucket", "n_users", "baseline_precision", "hybrid_precision",
                     "list_precision"]].rename(columns={"hybrid_precision": "gradient_boost_precision"})
    groups["best"] = groups[["baseline_precision", "gradient_boost_precision", "list_precision"]] \
        .idxmax(axis=1).str.replace("_precision", "")
    groups.to_csv(RESULTS / "three_way_by_user_group.csv", index=False)

    # Figures ------------------------------------------------------------------------
    viz.use_project_style()
    order = ["baseline", "gradient_boost", "list_model"]
    x, w = np.arange(len(METRICS)), 0.27
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(order):
        bars = ax.bar(x + (i - 1) * (w + 0.01), [agg[m][k] for _, k, _ in METRICS], w,
                      color=COLORS[m], label=NAMES[m], zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, color=viz.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in METRICS])
    ax.set_ylabel("Mean over report-half users (n=2,953)")
    ax.set_title("Baseline vs. gradient boost vs. list taste model")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    fig.savefig(FIGURES / "fig12_three_way_comparison.png", bbox_inches="tight")
    plt.close(fig)

    t = groups[groups["group"] == "history_size"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (m, col) in enumerate([("baseline", "baseline_precision"),
                                  ("gradient_boost", "gradient_boost_precision"),
                                  ("list_model", "list_precision")]):
        bars = ax.bar(x + (i - 1) * (w + 0.01), t[col], w, color=COLORS[m], label=NAMES[m], zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, color=viz.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(t["bucket"], t["n_users"])])
    ax.set_ylabel("Precision@10")
    ax.set_title("Precision@10 by how many films the user had rated")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    fig.savefig(FIGURES / "fig13_three_way_by_history_size.png", bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
