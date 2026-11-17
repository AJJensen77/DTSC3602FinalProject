"""Evaluate the list-based taste model against the popularity baseline.

Same protocol as every group model (see ``Gradient boost model/baseline/README.md``):

* D2's chronological split. The model learns only from the training split
  (1995 -> Oct 2016); hits are judged on the validation split (Oct 2016 -> Nov 2019).
  The reserved test split is never opened (checked by its access time).
* D2's scorable cohort (>= 20 training ratings, >= 5 validation ratings), shuffled with
  seed 3602 and split 50/50. Every setting is chosen on the *tune* half; every reported
  number comes from the *report* half.
* A hit is a recommended film the user went on to rate >= 3.5 stars.
* Shared metric code (``per_user_metrics``, ``summarize``, ``bootstrap_diff``) imported
  from ``evaluate_hybrid.py``, so the numbers are directly comparable.

Usage (from the repo root):
    uv run python "Ishika's model/evaluate_list_model.py"          # full run, ~10-15 min
    uv run python "Ishika's model/evaluate_list_model.py" --quick  # 15k-user sample, checks the code

Writes ``Ishika's model/results/`` (JSON + CSVs), ``Ishika's model/figures/``, and the fitted
model to ``Ishika's model/artifacts/list_model.joblib``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Gradient boost model"))  # shared evaluation code

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from dtsc_3602_final_project import viz  # noqa: E402
from dtsc_3602_final_project.paths import LINKS_CSV, PROCESSED_DIR, RESERVED_TEST_DIR  # noqa: E402
from listmodel.recommender import ListTasteRecommender  # noqa: E402

RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
ARTIFACT_DIR = HERE / "artifacts"
K, SEED = ev.K, ev.SEED

# Tuning grid. Each damping value is one SVD fit at the largest size; smaller factor
# counts are truncations of it. Popularity weight / quality floor are scoring-only.
DAMPING_GRID = [0.3, 0.6]
FACTOR_GRID = [64, 128, 192]
POP_WEIGHT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
MIN_MEAN_GRID = [None, 3.5]
BASELINE_WINDOW = 365  # the window the baseline selected on the tune half

# Films for the qualitative "more like this" check.
EXAMPLE_FILMS = {1: "Toy Story", 2571: "The Matrix", 318: "Shawshank Redemption",
                 4973: "Amelie", 5618: "Spirited Away", 2959: "Fight Club",
                 1197: "The Princess Bride", 6539: "Pirates of the Caribbean"}

t0 = time.time()
results: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load(quick: bool):
    catalog = pd.read_parquet(PROCESSED_DIR / "catalog.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    if quick:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(train["userId"].unique(), 15_000, replace=False)
        train, val = train[train["userId"].isin(keep)], val[val["userId"].isin(keep)]
    return catalog, train, val


def evaluate(model, train, users, val, popularity):
    recs = model.recommend(model.history_matrix(train, users), K)
    g = ev.per_user_metrics(recs, users, val)
    return recs, g, ev.summarize(g, recs, popularity)


# -- figures --------------------------------------------------------------------


def fig_comparison(rep: dict) -> None:
    metrics = [("precision_at_10", "Precision@10"), ("recall_at_10", "Recall@10"),
               ("ndcg_at_10", "NDCG@10"), ("hit_rate_at_10", "Hit rate@10")]
    x, w = np.arange(len(metrics)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, key, label, color in [(-w / 2 - 0.01, "baseline", "Popularity baseline", viz.SERIES_2),
                                   (w / 2 + 0.01, "list_model", "List taste model", viz.SERIES_3)]:
        bars = ax.bar(x + off, [rep[key][m] for m, _ in metrics], w, color=color,
                      label=label, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=viz.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Mean over report-half users")
    ax.set_title("List taste model vs. popularity baseline on validation ratings")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    fig.savefig(FIGURES_DIR / "fig9_list_model_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)


def fig_by_group(table: pd.DataFrame, group: str, title: str, fname: str) -> None:
    t = table[table["group"] == group]
    x, w = np.arange(len(t)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, col, label, color in [(-w / 2 - 0.01, "baseline_precision", "Popularity baseline", viz.SERIES_2),
                                   (w / 2 + 0.01, "list_precision", "List taste model", viz.SERIES_3)]:
        bars = ax.bar(x + off, t[col], w, color=color, label=label, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=viz.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(t["bucket"], t["n_users"])])
    ax.set_ylabel("Precision@10")
    ax.set_title(title)
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    plt.close(fig)


# -- main -------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for d in (RESULTS_DIR, FIGURES_DIR, ARTIFACT_DIR):
        d.mkdir(exist_ok=True)
    sealed = RESERVED_TEST_DIR / "ratings_test.parquet"
    sealed_before = sealed.stat().st_atime

    catalog, train, val = load(args.quick)
    titles = catalog.set_index("movieId")["clean_title"]
    popularity = train.groupby("movieId").size()
    tune_users, report_users = ev.cohort(train, val)
    log(f"cohort: {len(tune_users):,} tune / {len(report_users):,} report users")
    results["cohort"] = {"tune_users": int(len(tune_users)), "report_users": int(len(report_users))}

    # Tune ---------------------------------------------------------------------------
    tuning, best = [], None
    for damping in DAMPING_GRID:
        log(f"fitting SVD: damping={damping}, {max(FACTOR_GRID)} factors...")
        big = ListTasteRecommender(n_factors=max(FACTOR_GRID), popularity_damping=damping,
                                   window_days=BASELINE_WINDOW).fit(train)
        for nf in FACTOR_GRID:
            m = big.with_factors(nf)
            for mm in MIN_MEAN_GRID:
                for pw in POP_WEIGHT_GRID:
                    m.min_mean, m.popularity_weight = mm, pw
                    _, _, s = evaluate(m, train, tune_users, val, popularity)
                    row = {"popularity_damping": damping, "n_factors": nf, "min_mean": mm,
                           "popularity_weight": pw, **s}
                    tuning.append(row)
                    if best is None or s["precision_at_10"] > best["precision_at_10"]:
                        best = row
            log(f"  factors={nf}: best so far precision@10 {best['precision_at_10']:.4f}")
        del big
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(RESULTS_DIR / "list_model_tuning.csv", index=False)
    chosen = {k: best[k] for k in ("popularity_damping", "n_factors", "min_mean", "popularity_weight")}
    chosen = {k: (None if pd.isna(v) else v) for k, v in chosen.items()}
    chosen["n_factors"] = int(chosen["n_factors"])
    results["selected"] = chosen | {"tune_precision_at_10": float(best["precision_at_10"])}
    log(f"selected {chosen}")

    model = ListTasteRecommender(window_days=BASELINE_WINDOW, **chosen).fit(train)
    baseline = PopularityRecommender(window_days=BASELINE_WINDOW).fit(train)

    # Report half ----------------------------------------------------------------------
    log("report half: scoring list model and baseline...")
    l_recs, gl, sl = evaluate(model, train, report_users, val, popularity)
    b_recs = baseline.recommend(baseline.history_matrix(train, report_users), K)
    gb = ev.per_user_metrics(b_recs, report_users, val)
    rep = {"list_model": sl, "baseline": ev.summarize(gb, b_recs, popularity)}
    rep["difference_list_minus_baseline"] = {
        m: ev.bootstrap_diff(gl[m], gb[m]) for m in ("precision", "recall", "ndcg", "hit_rate")}
    results["report"] = rep

    # Variants on the report half: what each design choice contributes ---------------------
    log("variants...")
    variants = {"Selected list model": sl}

    def variant(label, **params):
        v = model.with_factors(model.n_factors)
        v.set_params(**params)
        variants[label] = evaluate(v, train, report_users, val, popularity)[2]

    variant("Taste only (no popularity blend)", popularity_weight=0.0, min_mean=None)
    variant("Dropped ignored (weight 0)", list_weights={"dropped": 1e-3})
    variant("All lists equal (no signs)", list_weights={"dropped": 1.0}, meh_weight=1.0)
    variants["Popularity baseline"] = rep["baseline"]
    results["variants_report_half"] = variants

    # Error analysis --------------------------------------------------------------------
    log("error analysis...")
    tr = train[train["userId"].isin(report_users)]
    pop_share = popularity.rank(pct=True)
    tr_u = tr.groupby("userId").agg(n_train=("rating", "size"), mean_train=("rating", "mean"),
                                    last=("timestamp", "max"))
    tr_u["taste_mainstream"] = (tr.assign(p=pop_share.reindex(tr["movieId"]).to_numpy())
                                .groupby("userId")["p"].mean())
    tr_u["share_dropped"] = (tr["rating"] <= model.dropped_at_or_below).groupby(tr["userId"]).mean()
    first_val = val[val["userId"].isin(report_users)].groupby("userId")["timestamp"].min()
    tr_u["gap_days"] = (first_val - tr_u["last"]) / 86400
    groups = {
        "history_size": pd.cut(tr_u["n_train"], [19, 49, 99, 249, 999, np.inf],
                               labels=["20-49", "50-99", "100-249", "250-999", "1000+"]),
        "user_generosity": pd.qcut(tr_u["mean_train"], 3,
                                   labels=["harsh (low mean)", "middle", "generous (high mean)"]),
        "taste_mainstream": pd.qcut(tr_u["taste_mainstream"], 3,
                                    labels=["niche taste", "middle", "mainstream taste"]),
        "share_dropped": pd.qcut(tr_u["share_dropped"].rank(method="first"), 3,
                                 labels=["few dropped", "some dropped", "many dropped"]),
        "gap_since_last_train_rating": pd.cut(tr_u["gap_days"], [-1, 30, 365, 1825, np.inf],
                                              labels=["<1 month", "1-12 months", "1-5 years", "5+ years"]),
    }
    rows = []
    for gname, buckets in groups.items():
        for bucket in buckets.cat.categories:
            ids = buckets.index[buckets == bucket]
            if len(ids) == 0:
                continue
            ci = ev.bootstrap_diff(gl.loc[ids, "precision"], gb.loc[ids, "precision"], n=1000)
            rows.append({"group": gname, "bucket": bucket, "n_users": int(len(ids)),
                         "list_precision": gl.loc[ids, "precision"].mean(),
                         "baseline_precision": gb.loc[ids, "precision"].mean(),
                         "diff_ci95_low": ci["ci95_low"], "diff_ci95_high": ci["ci95_high"],
                         "list_recall": gl.loc[ids, "recall"].mean(),
                         "baseline_recall": gb.loc[ids, "recall"].mean()})
    by_group = pd.DataFrame(rows)
    by_group.to_csv(RESULTS_DIR / "list_model_error_by_user_group.csv", index=False)

    # Which films get recommended, and how often they turn out to be hits, by popularity.
    def by_film_popularity(recs, name):
        long = pd.DataFrame({"userId": np.repeat(report_users, K), "movieId": recs.ravel()})
        long = long[long["movieId"] >= 0].merge(val[["userId", "movieId", "rating"]],
                                                on=["userId", "movieId"], how="left")
        long["hit"] = long["rating"] >= 3.5
        long["bucket"] = pd.cut(popularity.reindex(long["movieId"]).fillna(0).to_numpy(),
                                [19, 99, 999, 9999, np.inf],
                                labels=["20-99", "100-999", "1k-10k", "10k+"])
        out = long.groupby("bucket", observed=False).agg(n_recs=("hit", "size"), hits=("hit", "sum"))
        out["share_of_recs"] = out["n_recs"] / out["n_recs"].sum()
        out["hit_rate"] = out["hits"] / out["n_recs"].clip(lower=1)
        return out.add_prefix(f"{name}_")

    by_pop = pd.concat([by_film_popularity(l_recs, "list"), by_film_popularity(b_recs, "baseline")],
                       axis=1).reset_index().rename(columns={"bucket": "film_training_ratings"})
    by_pop.to_csv(RESULTS_DIR / "list_model_recs_by_film_popularity.csv", index=False)

    # Hits by genre of the recommended film.
    long = pd.DataFrame({"userId": np.repeat(report_users, K), "movieId": l_recs.ravel()})
    long = long[long["movieId"] >= 0].merge(val[["userId", "movieId", "rating"]],
                                            on=["userId", "movieId"], how="left")
    long["hit"] = long["rating"] >= 3.5
    long = long.merge(catalog[["movieId", "genres"]], on="movieId")
    long = long.assign(genre=long["genres"].str.split("|")).explode("genre")
    by_genre = (long.groupby("genre").agg(n_recs=("hit", "size"), hit_rate=("hit", "mean"))
                .query("n_recs >= 200").sort_values("hit_rate").reset_index())
    by_genre.to_csv(RESULTS_DIR / "list_model_hits_by_genre.csv", index=False)
    results["error_analysis"] = {"by_user_group": by_group.to_dict("records"),
                                 "by_film_popularity": by_pop.to_dict("records"),
                                 "by_genre": by_genre.to_dict("records")}

    # "More like this" and list examples (qualitative) ----------------------------------
    similar = {}
    for mid, name in EXAMPLE_FILMS.items():
        try:
            sim = model.similar_movies(mid, 8)
        except KeyError:
            continue
        similar[name] = [f"{titles.get(i, i)} ({s:.2f})" for i, s in zip(sim["movieId"], sim["similarity"])]
    results["more_like_this_examples"] = similar
    demo_lists = {"completed": [2571, 1196, 260], "plan_to_watch": [79132],
                  "dropped": [1721]}  # Matrix, Empire, Star Wars / Inception / Titanic
    results["lists_example"] = {
        "lists": {k: [titles.get(i, i) for i in v] for k, v in demo_lists.items()},
        "recommendations": [titles.get(i, i) for i in model.recommend_from_lists(demo_lists, 10)],
    }

    # Outputs -------------------------------------------------------------------------
    viz.use_project_style()
    fig_comparison(rep)
    fig_by_group(by_group, "history_size", "Precision@10 by how many films the user had rated",
                 "fig10_list_model_by_history_size.png")
    fig_by_group(by_group, "taste_mainstream", "Precision@10 by how mainstream the user's taste is",
                 "fig11_list_model_by_taste.png")

    if not args.quick:
        links = pd.read_csv(LINKS_CSV, dtype={"imdbId": str})
        meta = (catalog[["movieId", "clean_title", "year", "genres"]]
                .merge(links, on="movieId", how="left")
                .set_index("movieId").reindex(model.item_ids_).reset_index())
        joblib.dump({"model": model, "movies": meta}, ARTIFACT_DIR / "list_model.joblib",
                    compress=3)
        log(f"saved {ARTIFACT_DIR / 'list_model.joblib'}")

    results["reserved_test_untouched"] = bool(sealed.stat().st_atime == sealed_before)
    results["quick_mode"] = args.quick
    out = RESULTS_DIR / ("list_model_results_quick.json" if args.quick else "list_model_results.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out}")
    print(json.dumps({k: results[k] for k in ("selected", "report")}, indent=2, default=str))


if __name__ == "__main__":
    main()
