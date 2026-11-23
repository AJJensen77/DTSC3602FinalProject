"""Do the site's recommendations actually follow what's on a person's list?

Runs the exact model the API serves (api/artifacts/site_bundle.joblib) on made-up
website users with very different lists, and checks that:

1. Each persona's picks match their taste (share of picks in the persona's genre,
   vs the same share among the 10 picks a brand-new user with an empty list gets).
2. Different lists give different picks (overlap between personas' top-10s).
3. Lists alone work (Finished watching, no stars) and Plan to watch counts.
4. Scores matter: the same films rated 1 star instead of 5 change the picks.
5. Nothing already on a list is recommended back.

Writes results/persona_check.json and prints a readable report.
Usage (repo root, after build_site.py):
    uv run python "Combined website/check_personas.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "FilmPalate" / "D4_submission", ROOT / "List taste model",
          ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

ART = HERE / "api" / "artifacts"
N_SEED = 12  # films on each persona's Finished watching list

# Persona -> (genre that defines them, genres their seed films must NOT have)
PERSONAS = {
    "horror fan": ("Horror", {"Comedy", "Children"}),
    "animation / family": ("Animation", {"Horror"}),
    "documentary watcher": ("Documentary", set()),
    "romance fan": ("Romance", {"Horror", "Animation"}),
    "western fan": ("Western", {"Comedy"}),
    "sci-fi fan": ("Sci-Fi", {"Comedy", "Animation", "Children"}),
    "war films": ("War", {"Comedy", "Animation"}),
    "musicals": ("Musical", {"Horror"}),
}


def main() -> None:
    bundle = joblib.load(ART / "site_bundle.joblib")
    site = bundle["site"]
    comb = site.combined
    cat = pd.read_parquet(ART / "catalog.parquet").set_index("movieId")
    genres = cat["genres"].fillna("").str.split("|")
    lm_ids = set(comb.list_model.item_ids_.tolist())
    rng = np.random.default_rng(3602)

    def has(mid, g):
        return g in genres.get(mid, [])

    def seed_films(genre, avoid):
        pool = cat[cat.index.isin(lm_ids) & (cat["n_ratings"] >= 2000)]
        pool = pool[pool["genres"].str.contains(genre, regex=False)]
        pool = pool[~pool["genres"].apply(lambda s: any(a in s for a in avoid))]
        # A mix of famous and less famous films from the genre, not just the top 12.
        top = pool.nlargest(60, "n_ratings").index.to_numpy()
        return [int(x) for x in rng.choice(top, N_SEED, replace=False)]

    def titles(ids):
        return [cat.at[m, "title"] for m in ids]

    def picks(**kw):
        return [r["movieId"] for r in site.recommend(k=10, k_new=0, **kw)[0]]

    def new_picks(**kw):
        return [r["movieId"] for r in site.recommend(k=0, k_new=10, **kw)[1]]

    empty = picks(ratings={})
    out, report = {}, []
    report.append(f"Brand-new user (empty list) gets: {titles(empty)}")
    rec_sets = {}
    for name, (genre, avoid) in PERSONAS.items():
        seeds = seed_films(genre, avoid)
        ratings = {m: 4.5 for m in seeds}
        r_rated = picks(ratings=ratings)
        r_lists = picks(ratings={}, finished=seeds)  # list only, no stars
        r_low = picks(ratings={m: 1.0 for m in seeds})  # same films, hated them
        match = np.mean([has(m, genre) for m in r_rated])
        base = np.mean([has(m, genre) for m in empty])
        catalog_share = np.mean([has(m, genre) for m in lm_ids])
        leaked = sorted(set(r_rated) & set(seeds))
        r_new = new_picks(ratings=ratings)
        rec_sets[name] = set(r_rated)
        out[name] = {
            "genre": genre, "seed_films": titles(seeds), "picks_rated_4_5": titles(r_rated),
            "picks_finished_list_only": titles(r_lists), "picks_rated_1_0": titles(r_low),
            "genre_share_of_picks": float(match), "genre_share_empty_user": float(base),
            "genre_share_of_catalog": float(catalog_share),
            "list_only_genre_share": float(np.mean([has(m, genre) for m in r_lists])),
            "hated_genre_share": float(np.mean([has(m, genre) for m in r_low])),
            "seed_films_recommended_back": titles(leaked),
            "new_release_picks": titles(r_new),
            "new_release_genre_share": float(np.mean([has(m, genre) for m in r_new])) if r_new else None,
        }
        report.append(
            f"\n{name.upper()} ({genre}): {match:.0%} of picks are {genre} "
            f"(empty-list user: {base:.0%}, catalog: {catalog_share:.0%}); list-only: "
            f"{out[name]['list_only_genre_share']:.0%}; if they hated those films: "
            f"{out[name]['hated_genre_share']:.0%}\n  seeds: {titles(seeds)[:5]}...\n"
            f"  picks: {titles(r_rated)}\n"
            f"  new releases ({out[name]['new_release_genre_share']:.0%} {genre}): {titles(r_new)[:6]}")

    names = list(rec_sets)
    overlaps = [len(rec_sets[a] & rec_sets[b]) / 10 for i, a in enumerate(names) for b in names[i + 1:]]
    plan = PERSONAS["horror fan"]
    seeds = out["horror fan"]["seed_films"]
    plan_ids = [m for m in cat.index if cat.at[m, "title"] in set(seeds)][:N_SEED]
    r_plan = picks(ratings={}, plan_to_watch=plan_ids)
    empty_new = new_picks(ratings={})
    report.append(f"\nBrand-new user's new releases: {titles(empty_new)}")
    summary = {
        "mean_overlap_between_personas_top10": float(np.mean(overlaps)),
        "max_overlap_between_personas_top10": float(np.max(overlaps)),
        "personas_where_genre_share_beats_empty_user": int(sum(
            v["genre_share_of_picks"] > v["genre_share_empty_user"] for v in out.values())),
        "personas_with_seed_films_recommended_back": int(sum(
            bool(v["seed_films_recommended_back"]) for v in out.values())),
        "plan_to_watch_only_horror_share": float(np.mean([has(m, plan[0]) for m in r_plan])),
        "mean_new_release_genre_share": float(np.mean([v["new_release_genre_share"] or 0 for v in out.values()])),
        "mean_new_release_genre_share_empty_user": float(np.mean([
            np.mean([has(m, g) for m in empty_new]) for g, _ in PERSONAS.values()])),
    }
    report.append(f"\nSUMMARY {json.dumps(summary, indent=2)}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "persona_check.json").write_text(
        json.dumps({"summary": summary, "personas": out, "empty_list_picks": titles(empty)},
                   indent=2))
    print("\n".join(report))


if __name__ == "__main__":
    main()
