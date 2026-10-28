"""D2 dashboard: data preparation and EDA for the movie recommender.

Reads the aggregate tables from Supabase Postgres (never the raw ratings) and
presents the audit, the five EDA findings, the split design, and the dictionary.

Local:  uv run streamlit run app.py
Deploy: uv run modal deploy modal_app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter

from common import get_engine, table
from dtsc_3602_final_project import viz

load_dotenv()
viz.use_project_style()

st.set_page_config(page_title="Movie Recommender - Data & EDA", layout="wide")

# Offline development only. The graded path is the Supabase one (the default);
# this flag exists so the layout can be worked on without a database.
USE_LOCAL = os.environ.get("USE_LOCAL_PARQUET") == "1"


@st.cache_data(ttl=600)
def load(name: str) -> pd.DataFrame:
    if USE_LOCAL:
        from dtsc_3602_final_project.paths import PROCESSED_DIR

        return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return pd.read_sql_table(table(name), get_engine())


def pct(x: float) -> str:
    return f"{x:.1f}%"


try:
    splits = load("split_summary")
    cohort = load("eval_cohort")
    movies = load("movies_dim")
except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
    st.title("Movie Recommender - Data Preparation & EDA")
    st.error(
        "Couldn't reach the Supabase database. Check that DATABASE_URL is set "
        "(see .env.example) and that `upload_to_supabase.py` has been run once."
    )
    st.exception(exc)
    st.stop()

st.title("Movie Recommender - Data Preparation & EDA")
st.caption(
    "DTSC 3602 - D2 - MovieLens 32M joined to IMDb title types, served from "
    "Supabase Postgres via a Modal-deployed Streamlit app"
)

st.markdown(
    "**Research question.** Can a personalized model built from a user's earlier "
    "ratings recommend movies they go on to rate highly, better than recommending "
    "the most popular movies?"
)

total_ratings = int(splits["n_ratings"].sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ratings retained", f"{total_ratings / 1e6:.1f}M")
k2.metric("Films in catalog", f"{len(movies):,}")
k3.metric("Users in training", f"{int(splits.loc[splits['split'] == 'train', 'n_users'].iloc[0]):,}")
k4.metric("TV titles removed", "2,505")

st.divider()

tab_overview, tab_quality, tab_split, tab_catalog, tab_dict = st.tabs(
    ["Preparation", "Data quality", "Split & evaluation", "Catalog", "Data dictionary"]
)


# --------------------------------------------------------------------------
with tab_overview:
    st.subheader("What the preparation step did")
    st.markdown(
        """
MovieLens ships no title-type field, so every catalog entry was joined to IMDb's
official `title.basics.tsv` through `links.csv`, and **episodic television was
dropped** on IMDb's `titleType`: `tvSeries`, `tvMiniSeries`, `tvEpisode`,
`tvSpecial`, `tvShort`, and two `videoGame` rows. Made-for-TV films (`tvMovie`),
direct-to-video films (`video`) and short films (`short`) were kept - these are
standalone films, not shows. 228 titles had no IMDb match; inspection showed
ordinary films with a stale id, so they were kept and flagged `unknown`.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
| Step | Rows affected |
|---|---|
| Raw MovieLens ratings | 32,000,204 |
| Dropped with TV titles | -106,959 (0.33%) |
| Duplicate (user, movie) pairs | 0 |
| Ratings outside 0.5-5.0 | 0 |
| Ratings with no catalog entry | 0 |
| **Retained** | **31,893,245** |
            """
        )
    with c2:
        st.markdown(
            """
| Step | Titles affected |
|---|---|
| Raw MovieLens catalog | 87,585 |
| Episodic TV dropped | -2,505 |
| **Retained films** | **85,080** |
| ...of which never rated in train | 49,419 |
            """
        )
    st.info(
        "The raw files were clean on the dimensions that usually break a join: no "
        "duplicate user-movie pairs, no out-of-range stars, no orphan movie ids. "
        "The real problems are structural, not dirty values - see **Data quality**."
    )


# --------------------------------------------------------------------------
with tab_quality:
    st.subheader("Finding 1 - the rating scale changed partway through")
    era = load("rating_distribution_by_era")
    values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    before = era[era["era"] == "Whole stars only"].set_index("rating")["share_within_era"]
    after = era[era["era"] == "Half stars allowed"].set_index("rating")["share_within_era"]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(values))
    w = 0.4
    ax.bar(x - w / 2 - 0.01, [before.get(v, 0) * 100 for v in values], w,
           label="Before Feb 2003 (whole stars only)", color=viz.SERIES_1, zorder=3)
    ax.bar(x + w / 2 + 0.01, [after.get(v, 0) * 100 for v in values], w,
           label="After Feb 2003 (half stars allowed)", color=viz.SERIES_2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Share of era (%)")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    st.pyplot(fig)
    st.markdown(
        "Half-stars did not exist before **18 Feb 2003**. After that date they are "
        "**42%** of all ratings. The training window spans both regimes, so star "
        "values are not one consistent measurement. Anything that treats the scale "
        "as stable across 1995-2016 is mixing two different instruments."
    )

    st.divider()
    st.subheader("Finding 2 - rating volume is uneven across 28 years")
    monthly = load("monthly_activity")
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(monthly["month"], monthly["n_ratings"], color=viz.SERIES_1, linewidth=2, zorder=3)
    ax.set_ylabel("Ratings per month")
    ax.set_xlabel("Month")
    ax.yaxis.set_major_formatter(FuncFormatter(viz.thousands))
    viz.style_axes(ax)
    st.pyplot(fig)
    st.markdown(
        "Volume swings by an order of magnitude between bursts and quiet stretches. "
        "Any metric averaged over calendar time is dominated by the busiest months, "
        "which is why the splits below are cut on **rating count**, not on dates."
    )

    st.divider()
    st.subheader("Finding 3 - a few movies absorb almost all ratings")
    c1, c2 = st.columns([3, 2])
    with c1:
        pc = load("popularity_concentration")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], color=viz.GRID, linewidth=1.5, linestyle="--", zorder=2)
        ax.plot(pc["pct_movies"], pc["pct_ratings"], color=viz.SERIES_1, linewidth=2, zorder=3)
        for q in (0.01, 0.10):
            i = (pc["pct_movies"] - q).abs().idxmin()
            yv = pc.loc[i, "pct_ratings"]
            ax.scatter([q], [yv], s=40, color=viz.BLUE_600, zorder=4,
                       edgecolors=viz.SURFACE, linewidths=2)
            ax.annotate(f"top {q:.0%} hold {yv:.0%}", xy=(q, yv), xytext=(12, -4),
                        textcoords="offset points", color=viz.INK, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
        ax.set_xlabel("Movies, most-rated first")
        ax.set_ylabel("Cumulative share of training ratings")
        viz.style_axes(ax)
        st.pyplot(fig)
    with c2:
        st.markdown(
            "The top **1%** of movies hold **39%** of training ratings; the top "
            "**10%** hold **90%**.\n\n"
            "This is the popularity bias named in the proposal, measured. A "
            "most-popular baseline is therefore a genuinely strong opponent, not a "
            "straw man - and a personalized model that merely re-ranks blockbusters "
            "will look good on accuracy while adding nothing."
        )

    st.divider()
    st.subheader("Finding 4 - most of the catalog is too sparse to learn from")
    mb = load("movie_popularity_bins")
    order = ["0", "1-4", "5-19", "20-99", "100-999", "1k-10k", "10k+"]
    mb = mb.set_index("bucket").reindex(order).reset_index()
    y = np.arange(len(mb))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    axes[0].barh(y, mb["n_movies"], color=viz.SERIES_1, height=0.62, zorder=3)
    axes[0].set_xlim(0, mb["n_movies"].max() * 1.28)
    axes[0].set_title("Movies in catalog", fontsize=11)
    for yi, v in zip(y, mb["n_movies"]):
        axes[0].text(v + mb["n_movies"].max() * 0.02, yi, f"{v:,}", va="center", fontsize=9)
    share = mb["n_ratings"] / mb["n_ratings"].sum() * 100
    axes[1].barh(y, share, color=viz.BLUE_600, height=0.62, zorder=3)
    axes[1].set_xlim(0, share.max() * 1.28)
    axes[1].set_title("Share of training ratings", fontsize=11)
    for yi, v in zip(y, share):
        axes[1].text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9)
    for a in axes:
        a.set_yticks(y)
        a.set_yticklabels(mb["bucket"])
        a.invert_yaxis()
        viz.style_axes(a, xgrid=True, ygrid=False)
    axes[0].set_ylabel("Training ratings per movie")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "**49,419 of 85,080 films have zero training ratings**, and 516 films "
        "(0.6% of the catalog) carry 47% of them. Recommendations can only be "
        "made over the warm part of the catalog; the website's browsable catalog "
        "and its recommendable catalog are not the same set."
    )


# --------------------------------------------------------------------------
with tab_split:
    st.subheader("Chronological split")
    st.markdown(
        "The question is explicitly about predicting **later** ratings from "
        "**earlier** ones, so the data is cut by time at the 70 / 15 / 15 "
        "quantiles of the rating timestamp. A random split would let a model see "
        "a user's future while predicting their past."
    )
    show = splits.copy()
    show["start"] = show["start"].astype(str).str[:10]
    show["end"] = show["end"].astype(str).str[:10]
    st.dataframe(
        show[["split", "n_ratings", "n_users", "n_movies", "start", "end",
              "mean_rating", "pct_ratings_from_cold_users"]].rename(
            columns={
                "split": "Split", "n_ratings": "Ratings", "n_users": "Users",
                "n_movies": "Movies", "start": "From", "end": "To",
                "mean_rating": "Mean stars",
                "pct_ratings_from_cold_users": "% from unseen users",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.success(
        "**Test data is reserved.** Everything after 2019-11-02 is written to "
        "`data/reserved_test/` and is not read again until D6, after the model "
        "and its settings are frozen."
    )

    st.divider()
    st.subheader("Finding 5 - most later ratings cannot be personalized at all")
    labels = cohort["split"].str.capitalize().tolist()
    scored = cohort["n_ratings_scored"].to_numpy()
    unscored = (cohort["n_ratings_total"] - cohort["n_ratings_scored"]).to_numpy()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.barh(y, scored, 0.5, color=viz.SERIES_1, zorder=3,
            label="Scorable: user has >=20 training ratings")
    ax.barh(y, unscored, 0.5, left=scored + max(scored) * 0.004, color=viz.SERIES_2,
            zorder=3, label="Not scorable: little or no training history")
    for yi, s, u in zip(y, scored, unscored):
        t = s + u
        ax.text(s / 2, yi, f"{s / t:.0%}", va="center", ha="center", color="white",
                fontsize=10, fontweight="semibold")
        ax.text(s + u / 2, yi, f"{u / t:.0%}", va="center", ha="center", color="white",
                fontsize=10, fontweight="semibold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ratings in split")
    ax.xaxis.set_major_formatter(FuncFormatter(viz.thousands))
    viz.style_axes(ax, xgrid=True, ygrid=False)
    viz.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
    st.pyplot(fig)

    st.warning(
        "**This is the most consequential finding in D2.** MovieLens users join "
        "over time, so most users who appear after the training cutoff are brand "
        "new: 85% of validation ratings and 91% of test ratings come from users "
        "with no training history. No model built on a user's past can score them."
    )
    st.markdown(
        "D3 therefore reports results on an explicit **scorable cohort** - users "
        "with at least 20 training ratings and at least 5 ratings in the "
        "evaluation split - and reports the excluded remainder separately rather "
        "than averaging it in:"
    )
    st.dataframe(
        cohort[["split", "n_users_scored", "n_ratings_scored", "pct_ratings_scored",
                "mean_rating_scored", "mean_rating_unscored"]].rename(
            columns={
                "split": "Split", "n_users_scored": "Scorable users",
                "n_ratings_scored": "Scorable ratings",
                "pct_ratings_scored": "% of split",
                "mean_rating_scored": "Mean stars (scorable)",
                "mean_rating_unscored": "Mean stars (excluded)",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "The cohort is **not** a representative sample: scorable users rate about "
        "0.25 stars lower than the users they exclude (3.35 vs 3.60 on validation). "
        "Heavy raters are harsher raters. Results on this cohort therefore describe "
        "experienced MovieLens users, not newcomers, and the final report must say "
        "so rather than generalising to all viewers."
    )


# --------------------------------------------------------------------------
with tab_catalog:
    st.subheader("Catalog explorer")
    st.caption(
        "Popularity and mean rating are computed on training data only, so "
        "browsing here cannot leak anything from the reserved test split."
    )

    genres = sorted({g for gs in movies["genres"].dropna().str.split("|") for g in gs})
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        pick = st.selectbox("Genre", ["All genres"] + genres)
    with c2:
        min_ratings = st.slider("Minimum training ratings", 0, 5000, 1000, step=100)
    with c3:
        kinds = sorted(movies["titleType"].dropna().unique())
        kind = st.multiselect("Title type", kinds, default=kinds)

    view = movies[movies["n_ratings_train"] >= min_ratings]
    view = view[view["titleType"].isin(kind)]
    if pick != "All genres":
        view = view[view["genres"].fillna("").str.contains(pick, regex=False)]

    st.caption(
        f"{len(view):,} of {len(movies):,} films match "
        f"({len(view) / len(movies) * 100:.1f}% of the catalog)."
    )
    st.dataframe(
        view.sort_values("n_ratings_train", ascending=False)
        .head(300)[["title", "year", "genres", "titleType",
                    "n_ratings_train", "mean_rating_train"]]
        .rename(columns={
            "title": "Title", "year": "Year", "genres": "Genres",
            "titleType": "Type", "n_ratings_train": "Train ratings",
            "mean_rating_train": "Mean stars",
        }),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    st.subheader("Genres")
    gs = load("genre_stats").sort_values("n_ratings", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(gs))
    ax.barh(y, gs["n_ratings"], color=viz.SERIES_1, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(gs["genre"])
    ax.invert_yaxis()
    ax.set_xlabel("Training ratings")
    ax.xaxis.set_major_formatter(FuncFormatter(viz.thousands))
    for yi, v, m in zip(y, gs["n_ratings"], gs["mean_rating"]):
        ax.text(v + gs["n_ratings"].max() * 0.01, yi, f"{m:.2f} stars",
                va="center", fontsize=8.5, color=viz.INK_SECONDARY)
    ax.set_xlim(0, gs["n_ratings"].max() * 1.18)
    viz.style_axes(ax, xgrid=True, ygrid=False)
    st.pyplot(fig)
    st.caption(
        "Bar length is rating volume; the label is that genre's mean star rating. "
        "Drama and Comedy dominate volume, but mean ratings sit in a narrow band - "
        "genre alone separates taste weakly."
    )


# --------------------------------------------------------------------------
with tab_dict:
    st.subheader("Data dictionary")
    dd = load("data_dictionary")
    pick = st.selectbox("Table", sorted(dd["table_name"].unique()))
    sub = dd[dd["table_name"] == pick]
    st.caption(sub["table_purpose"].iloc[0])
    st.dataframe(
        sub[["column_name", "dtype", "description"]].rename(
            columns={"column_name": "Column", "dtype": "Type",
                     "description": "Description"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Generated from the parquet files the pipeline wrote, so it cannot drift "
        "from the real schema. Source: MovieLens 32M (GroupLens), joined to IMDb "
        "`title.basics.tsv`. MovieLens is free for non-commercial research use "
        "with attribution and no implied endorsement."
    )
