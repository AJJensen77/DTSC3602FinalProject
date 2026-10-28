"""Step 3 of the D2 pipeline: render the five figures used in the data note.

Each figure answers one question the data note then interprets. Reads only the
small aggregate tables written by prepare_data.py.

Usage:
    uv run python make_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from dtsc_3602_final_project import viz
from dtsc_3602_final_project.paths import FIGURES_DIR, PROCESSED_DIR, ensure_dirs


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def save(fig, name: str) -> None:
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig1_rating_scale_change() -> None:
    """Rating distribution before vs after half-stars existed."""
    era = load("rating_distribution_by_era")
    before = era[era["era"] == "Whole stars only"].set_index("rating")["share_within_era"]
    after = era[era["era"] == "Half stars allowed"].set_index("rating")["share_within_era"]

    values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    b = [before.get(v, 0.0) * 100 for v in values]
    a = [after.get(v, 0.0) * 100 for v in values]

    x = np.arange(len(values))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    # 2px surface gap between adjacent bars comes from the width/offset pair.
    ax.bar(x - w / 2 - 0.01, b, w, label="Before Feb 2003 (whole stars only)",
           color=viz.SERIES_1, zorder=3)
    ax.bar(x + w / 2 + 0.01, a, w, label="After Feb 2003 (half stars allowed)",
           color=viz.SERIES_2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Share of ratings in era (%)")
    ax.set_title("The rating scale changed partway through the training period")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    save(fig, "fig1_rating_scale_change")


def fig2_activity_over_time() -> None:
    """Monthly rating volume, with the chronological split boundaries marked."""
    monthly = load("monthly_activity")
    splits = load("split_summary").set_index("split")
    train_end = pd.Timestamp(splits.loc["train", "end"]).tz_localize(None)
    val_end = pd.Timestamp(splits.loc["validation", "end"]).tz_localize(None)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["month"], monthly["n_ratings"], color=viz.SERIES_1, linewidth=2, zorder=3)
    top = monthly["n_ratings"].max() * 1.12
    ax.set_ylim(0, top)

    # Both boundaries sit near the right edge, so the labels hang to the LEFT of
    # their rules at staggered heights - otherwise they collide and overflow.
    for boundary, label, frac in (
        (train_end, "train ends", 0.97),
        (val_end, "validation ends", 0.84),
    ):
        ax.axvline(boundary, color=viz.MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.annotate(
            label,
            xy=(boundary, top * frac),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=viz.INK_SECONDARY,
            fontsize=9,
        )

    ax.set_ylabel("Ratings submitted per month")
    ax.set_xlabel("Month")
    ax.set_title("Rating volume is uneven across the 28-year collection window")
    ax.yaxis.set_major_formatter(FuncFormatter(viz.thousands))
    viz.style_axes(ax)
    save(fig, "fig2_activity_over_time")


def fig3_popularity_concentration() -> None:
    """Lorenz curve: what share of ratings the most-rated movies absorb."""
    pc = load("popularity_concentration")
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot([0, 1], [0, 1], color=viz.GRID, linewidth=1.5, linestyle="--", zorder=2)
    ax.annotate("if every movie were\nrated equally often", xy=(0.62, 0.62),
                xytext=(0, -34), textcoords="offset points",
                color=viz.MUTED, fontsize=8.5, ha="center")
    ax.plot(pc["pct_movies"], pc["pct_ratings"], color=viz.SERIES_1, linewidth=2, zorder=3)

    for q in (0.01, 0.10):
        i = (pc["pct_movies"] - q).abs().idxmin()
        yv = pc.loc[i, "pct_ratings"]
        ax.scatter([q], [yv], s=40, color=viz.BLUE_600, zorder=4,
                   edgecolors=viz.SURFACE, linewidths=2)
        ax.annotate(f"top {q:.0%} of movies\nhold {yv:.0%} of ratings",
                    xy=(q, yv), xytext=(12, -6), textcoords="offset points",
                    color=viz.INK, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    ax.set_xlabel("Movies, most-rated first")
    ax.set_ylabel("Cumulative share of training ratings")
    ax.set_title("A small set of movies absorbs almost all ratings")
    viz.style_axes(ax)
    save(fig, "fig3_popularity_concentration")


def fig4_long_tail() -> None:
    """How many movies sit at each popularity level, and what they contribute."""
    mb = load("movie_popularity_bins")
    order = ["0", "1-4", "5-19", "20-99", "100-999", "1k-10k", "10k+"]
    mb = mb.set_index("bucket").reindex(order).reset_index()
    y = np.arange(len(mb))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)

    ax = axes[0]
    ax.barh(y, mb["n_movies"], color=viz.SERIES_1, height=0.62, zorder=3)
    for yi, v in zip(y, mb["n_movies"]):
        ax.text(v + mb["n_movies"].max() * 0.02, yi, f"{v:,}", va="center",
                color=viz.INK, fontsize=9)
    ax.set_xlim(0, mb["n_movies"].max() * 1.28)
    ax.set_title("Movies in the catalog", fontsize=11)
    ax.set_xlabel("Number of movies")

    ax2 = axes[1]
    share = mb["n_ratings"] / mb["n_ratings"].sum() * 100
    ax2.barh(y, share, color=viz.BLUE_600, height=0.62, zorder=3)
    for yi, v in zip(y, share):
        ax2.text(v + 1.5, yi, f"{v:.0f}%", va="center", color=viz.INK, fontsize=9)
    ax2.set_xlim(0, max(share) * 1.28)
    ax2.set_title("Share of training ratings", fontsize=11)
    ax2.set_xlabel("Percent of all training ratings")

    for a in axes:
        a.set_yticks(y)
        a.set_yticklabels(mb["bucket"])
        a.invert_yaxis()
        viz.style_axes(a, xgrid=True, ygrid=False)
    axes[0].set_ylabel("Training ratings per movie")

    fig.suptitle("Most of the catalog is too sparse to learn from",
                 fontsize=12, fontweight="semibold", color=viz.INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_long_tail")


def fig5_eval_cohort() -> None:
    """How much of each evaluation split a personalized model can actually score."""
    cohort = load("eval_cohort")
    labels = cohort["split"].str.capitalize().tolist()
    scored = cohort["n_ratings_scored"].to_numpy()
    unscored = (cohort["n_ratings_total"] - cohort["n_ratings_scored"]).to_numpy()

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 3.2))

    ax.barh(y, scored, 0.5, color=viz.SERIES_1, zorder=3,
            label="Scorable: user has >=20 training ratings")
    ax.barh(y, unscored, 0.5, left=scored + max(scored) * 0.004, color=viz.SERIES_2,
            zorder=3, label="Not scorable: little or no training history")

    for yi, s, u in zip(y, scored, unscored):
        total = s + u
        ax.text(s / 2, yi, f"{s / total:.0%}", va="center", ha="center",
                color="white", fontsize=10, fontweight="semibold")
        ax.text(s + u / 2, yi, f"{u / total:.0%}", va="center", ha="center",
                color="white", fontsize=10, fontweight="semibold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ratings in split")
    ax.xaxis.set_major_formatter(FuncFormatter(viz.thousands))
    ax.set_title("Most later ratings come from users with no history to personalize from")
    viz.style_axes(ax, xgrid=True, ygrid=False)
    # Bars span the full width, so the legend goes below the axes rather than
    # on top of the percentage labels.
    viz.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, "fig5_eval_cohort")


def main() -> None:
    ensure_dirs()
    viz.use_project_style()
    fig1_rating_scale_change()
    fig2_activity_over_time()
    fig3_popularity_concentration()
    fig4_long_tail()
    fig5_eval_cohort()
    print("\nAll figures written to figures/")


if __name__ == "__main__":
    main()
