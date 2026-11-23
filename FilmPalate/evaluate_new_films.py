"""Backtest: how should the site rank films the List taste model has never seen?

On the website, 2017-2026 films have no MovieLens ratings. This script checks the
options on films whose outcome we *can* see:

1. Pretend films released 2015-2016 (10,000+ IMDb votes) are brand new: drop every
   training rating of them and fit the List taste model (Ishika's tuned settings)
   on what is left.
2. Give each one the averaged fingerprint of its closest catalog films by IMDb
   description, as ``WatchlistRanker`` does on the site.
3. For the group's evaluation cohort, rank only these films from each user's
   training history and score against the validation period (hit = rated >= 3.5),
   the group's precision@10. Users who rated none of the films are skipped.

Ranking = IMDb buzz (votes per year) + rating_weight x IMDb rating
          + taste_weight x the user's taste score.
Baseline = buzz alone ("most voted"). Settings are picked on the tune half and
reported on the report half, the group's protocol. Also reported: within-user AUC,
i.e. how often a liked film outscores a rated-but-not-liked one (0.5 = chance).

Usage (repo root, ~2 min):
    uv run python "FilmPalate/evaluate_new_films.py"
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "List taste model", ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

import evaluate_hybrid as ev  # noqa: E402  (shared cohort + bootstrap code)
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402
from watchlist.pipeline import LIKED_AT, WatchlistRanker  # noqa: E402

ISHIKA = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                    .read_text())["selected"]
NEW_YEARS = (2015, 2016)
MIN_VOTES = 10_000            # same rule the site uses for new films
TASTE_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
RATING_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
K = 10
OUT = HERE / "results"

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main():
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")

    held = mf[mf["year"].between(*NEW_YEARS) & (mf["numVotes"] >= MIN_VOTES)]
    held_ids = set(held["movieId"])
    train_old = train[~train["movieId"].isin(held_ids)]
    log(f"{len(held):,} held-out {NEW_YEARS} films; dropped "
        f"{len(train) - len(train_old):,} of their training ratings")

    rank = WatchlistRanker(n_factors=ISHIKA["n_factors"],
                           popularity_damping=ISHIKA["popularity_damping"],
                           popularity_weight=ISHIKA["popularity_weight"],
                           newest_year=NEW_YEARS[1]).fit(train_old, mf, held)
    nb, base = rank.n_base_, rank.base_
    new_ids = rank.film_ids_[nb:]
    log(f"fitted on {len(train_old):,} ratings; catalog {nb:,} films, {len(new_ids):,} new")

    v = val[val["movieId"].isin(held_ids)]
    liked = v[v["rating"] >= LIKED_AT].groupby("userId")["movieId"].agg(set).to_dict()
    new_index = pd.Index(new_ids)

    def half(users):
        users = np.array([u for u in users if u in set(v["userId"])])
        H = base.history_matrix(train_old, users)   # already list weights (+1/+0.25/-1)
        H = sp.hstack([H, sp.csr_matrix((len(users), len(new_ids)))], format="csr")
        return users, rank.taste_z(H)[:, nb:]

    def precision(users, score) -> pd.Series:
        top = new_ids[np.argsort(-score, axis=1)[:, :K]]
        return pd.Series([len(set(t.tolist()) & liked.get(u, set())) / K
                          for u, t in zip(users, top)], index=users)

    def auc(users, score) -> float:
        pos, out = {u: i for i, u in enumerate(users)}, []
        for u, g in v[v["userId"].isin(pos)].groupby("userId"):
            j = new_index.get_indexer(g["movieId"])
            y, s = g["rating"].to_numpy() >= LIKED_AT, score[pos[u], j]
            if y.any() and (~y).any():
                out.append((s[y][:, None] > s[~y][None, :]).mean())
        return float(np.mean(out))

    tune_users, report_users = ev.cohort(train, val)
    tune, report = half(tune_users), half(report_users)
    buzz, rz = rank.new_buzz_z_, rank.new_rating_z_

    def score(z, tw, rw):
        return tw * z + buzz + rw * rz

    grid, best = [], None
    for tw in TASTE_WEIGHTS:
        for rw in RATING_WEIGHTS:
            p = float(precision(tune[0], score(tune[1], tw, rw)).mean())
            grid.append({"taste_weight": tw, "rating_weight": rw, "tune_p10": p})
            log(f"taste {tw:<4} rating {rw:<4} tune precision@10 {p:.4f}")
            if best is None or p > best[2]:
                best = (tw, rw, p)
    tw, rw, _ = best

    users, z = report
    tiled = np.tile(buzz, (len(users), 1))
    chosen_p, buzz_p = precision(users, score(z, tw, rw)), precision(users, tiled)
    diff = ev.bootstrap_diff(chosen_p, buzz_p)
    signals = {"taste_only": auc(users, z), "imdb_rating_only": auc(users, np.tile(rz, (len(users), 1))),
               "buzz_only": auc(users, tiled), "chosen": auc(users, score(z, tw, rw))}
    results = {
        "held_out_years": NEW_YEARS, "n_held_out_films": int(len(new_ids)),
        "grid": grid,
        "selected": {"new_taste_weight": tw, "new_rating_weight": rw},
        "report": {"n_users": int(len(users)), "chosen_p10": float(chosen_p.mean()),
                   "most_voted_p10": float(buzz_p.mean()), "difference": diff,
                   "within_user_auc": signals},
    }
    log(f"selected taste_weight={tw}, rating_weight={rw}")
    log(f"report half ({len(users):,} users): chosen {chosen_p.mean():.4f} vs most-voted "
        f"{buzz_p.mean():.4f}; diff {diff}")
    log(f"within-user AUC: {signals}")
    OUT.mkdir(exist_ok=True)
    (OUT / "new_films_backtest.json").write_text(json.dumps(results, indent=2, default=float))
    log(f"wrote {OUT / 'new_films_backtest.json'}")


if __name__ == "__main__":
    main()
