"""Try the fitted list model by hand (run evaluate_list_model.py first to create the artifact).

Usage (from the repo root):
    uv run python "List taste model/demo.py"
    uv run python "List taste model/demo.py" --similar "Spirited Away"
"""

import argparse
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Gradient boost model"))

bundle = joblib.load(HERE / "artifacts" / "list_model.joblib")
model, movies = bundle["model"], bundle["movies"].set_index("movieId")


def find(title: str) -> int:
    hits = movies[movies["clean_title"].str.contains(title, case=False, regex=False)]
    if hits.empty:
        raise SystemExit(f"no film matching {title!r}")
    return int(hits.index[0])


def show(ids, header):
    print(f"\n{header}")
    for i in ids:
        m = movies.loc[i]
        print(f"  {m['clean_title']} ({m['year']:.0f})  tmdb={m['tmdbId']:.0f}")


ap = argparse.ArgumentParser()
ap.add_argument("--similar", help="film title for 'More like this'")
args = ap.parse_args()

if args.similar:
    mid = find(args.similar)
    show(model.similar_movies(mid, 10)["movieId"], f"More like {movies.loc[mid, 'clean_title']}:")
else:
    lists = {
        "completed": [find("Spirited Away"), find("Princess Bride"), find("Amelie")],
        "watching": [find("Grand Budapest Hotel")],
        "plan_to_watch": [find("Your Name")] if movies["clean_title"].str.contains("Your Name").any() else [],
        "dropped": [find("Transformers")],
    }
    for name, ids in lists.items():
        print(f"{name}: {[movies.loc[i, 'clean_title'] for i in ids]}")
    show(model.recommend_from_lists(lists, 10), "Because of your lists:")
