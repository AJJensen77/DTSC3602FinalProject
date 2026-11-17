"""Live demo of Ishika's list taste model (Streamlit).

Run locally (from the repo root, after evaluate_list_model.py has saved the artifact):
    uv run streamlit run "Ishika's model/demo_app/app.py"
Deploy:
    uv run modal deploy "Ishika's model/demo_app/modal_app.py"
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
# Locally: the model folder and Yager's shared code. On Modal both are copied next to app.py.
for p in (HERE, HERE.parent, HERE.parent.parent / "Gradient boost model"):
    sys.path.insert(0, str(p))

ARTIFACT = Path(os.environ.get("LIST_MODEL_ARTIFACT",
                               HERE.parent / "artifacts" / "list_model.joblib"))

st.set_page_config(page_title="List Taste Model Demo", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def load():
    bundle = joblib.load(ARTIFACT)
    model = bundle["model"]
    movies = bundle["movies"].copy()
    movies["n_ratings"] = model.item_counts_
    movies["label"] = movies.apply(
        lambda m: f"{m['clean_title']} ({m['year']:.0f})" if pd.notna(m["year"]) else m["clean_title"],
        axis=1)
    movies = movies.set_index("movieId")
    # Most-rated first, so the pickers show familiar films at the top.
    options = movies.sort_values("n_ratings", ascending=False)
    return model, movies, options


try:
    model, movies, options = load()
except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a traceback
    st.error(f"The model could not be loaded ({exc}). Try again in a minute.")
    st.stop()

label_to_id = dict(zip(options["label"], options.index))
all_labels = list(options["label"])


def movie_table(ids, extra=None):
    rows = []
    for rank, mid in enumerate(ids, 1):
        m = movies.loc[mid]
        q = quote_plus(f"{m['clean_title']} {'' if pd.isna(m['year']) else int(m['year'])} trailer")
        row = {
            "#": rank,
            "Movie": m["label"],
            "Genres": str(m["genres"]).replace("|", ", "),
            "Trailer": f"https://www.youtube.com/results?search_query={q}",
            "TMDB": (f"https://www.themoviedb.org/movie/{int(m['tmdbId'])}"
                     if pd.notna(m["tmdbId"]) else None),
        }
        if extra is not None:
            row["Similarity"] = round(float(extra[rank - 1]), 2)
        rows.append(row)
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Trailer": st.column_config.LinkColumn("Trailer", display_text="▶ Watch"),
            "TMDB": st.column_config.LinkColumn("TMDB", display_text="Details"),
            "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1,
                                                          format="%.2f"),
        },
    )


st.title("🎬 List Taste Model — live demo")
st.caption(
    "DTSC 3602 · Ishika Bhattarai · candidate model #2. Learned from 22M MovieLens ratings "
    "(1995–2016). Add movies to your lists and it suggests films people with similar lists "
    "went on to enjoy. Offline test: 2.1 good picks out of 10 vs 1.6 for 'most popular'."
)

tab_lists, tab_similar, tab_about = st.tabs(["Because of your lists", "More like this", "How it works"])

with tab_lists:
    st.write("Pick a few movies for each list. **Dropped** movies push similar films *down*.")
    defaults = {
        "completed": ["Spirited Away (Sen to Chihiro no kamikakushi) (2001)",
                      "Princess Bride, The (1987)"],
        "watching": [],
        "plan_to_watch": [],
        "dropped": [],
    }
    names = {"completed": "✅ Completed", "watching": "▶️ Currently Watching",
             "plan_to_watch": "🕒 Plan to Watch", "dropped": "✖️ Dropped"}
    cols = st.columns(4)
    picked = {}
    for col, key in zip(cols, names):
        with col:
            chosen = st.multiselect(names[key], all_labels,
                                    default=[d for d in defaults[key] if d in label_to_id],
                                    key=key, placeholder="Search movies...")
            picked[key] = [int(label_to_id[c]) for c in chosen]
    k = st.slider("How many recommendations", 5, 30, 10)

    if not any(picked.values()):
        st.info("Your lists are empty, so these are just popular, well-liked films. "
                "Add something to see personalized picks.")
    with st.spinner("Finding movies for you..."):
        recs = model.recommend_from_lists(picked, k)
    st.subheader("Recommended for you")
    movie_table(recs)

with tab_similar:
    default_idx = all_labels.index("Toy Story (1995)") if "Toy Story (1995)" in all_labels else 0
    choice = st.selectbox("Pick a movie", all_labels, index=default_idx)
    n = st.slider("How many similar movies", 5, 30, 10, key="n_similar")
    sim = model.similar_movies(int(label_to_id[choice]), n)
    st.subheader(f"More like {choice}")
    movie_table(sim["movieId"].tolist(), extra=sim["similarity"].tolist())

with tab_about:
    st.markdown(f"""
**What the model does.** Every film gets a short "taste fingerprint" ({model.n_factors} numbers)
learned with a truncated SVD from who rated what in MovieLens. Films watched by the same kinds of
people get similar fingerprints.

**Your lists → one taste profile.** Completed counts +1, Currently Watching +0.8, Plan to Watch
+0.5, Dropped −1. The weighted fingerprints are added up and every film is scored against the
result, then blended with how popular the film was in the last year of training data (weight
{model.popularity_weight}). Films already on your lists are never recommended.

**More like this** = the films whose fingerprints point the same direction (cosine similarity).

**How well it works (validation period, 2,953 users not used for tuning).**

| | List taste model | Most-popular baseline |
|---|---|---|
| Good picks out of 10 (rated ≥3.5★ later) | **2.08** | 1.65 |
| Users with ≥1 good pick | **71%** | 59% |
| Different films recommended | **569** | 223 |

**Limits.** Only films with ≥20 MovieLens ratings before Oct 2016 can be recommended
({len(movies):,} films), so newer releases are missing. MovieLens has ratings rather than lists,
so the list weights for Watching and Plan to Watch are design choices, not tested.
Trailer links open a YouTube search.

Code and full results: GitHub `AJJensen77/DTSC3602FinalProject`, branch `D3`, folder `Ishika's model/`.
""")
