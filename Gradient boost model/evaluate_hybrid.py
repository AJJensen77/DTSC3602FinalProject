"""Hybrid-model step 2: train, tune, and evaluate against the popularity baseline.

Protocol (no leakage, reserved test data never opened):

1. Training rows. Inside the training split, ratings before 2014-10-01 are the
   *history*; ratings from 2014-10-01 to the end of training are the *targets* the
   regressor learns to predict, each described using only the history.
2. Evaluation. Population state is rebuilt from the full training split and the
   model predicts the validation split (2016-10-01 -> 2019-11-02) for the D2
   scorable cohort. That cohort is split 50/50 at random: settings are chosen on
   the *tune* half and every reported number comes from the *report* half.
3. A recommendation is a hit when the user went on to rate that film >= 3.5.

Usage:
    uv run python "Gradient boost model/evaluate_hybrid.py"           # full run, ~30-40 min
    uv run python "Gradient boost model/evaluate_hybrid.py" --quick   # small sample, for checking the code
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtsc_3602_final_project import viz
from dtsc_3602_final_project.paths import PROCESSED_DIR, RESERVED_TEST_DIR
from baseline.popularity import PopularityRecommender
from gbmodel.hybrid import HybridRecommender
from gbmodel.recommenders import LIKE_THRESHOLD

HERE = Path(__file__).parent
DOCS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

K = 10
SEED = 3602
TARGET_START = pd.Timestamp("2014-10-01", tz="UTC").timestamp()
TRAIN_END_YEAR = 2016
MIN_TRAIN_RATINGS, MIN_EVAL_RATINGS = 20, 5  # D2's scorable-cohort definition
MAX_TRAIN_USERS, MAX_TARGETS_PER_USER = 20_000, 100

HGB_GRID = [
    {"learning_rate": 0.1, "max_leaf_nodes": 63},
    {"learning_rate": 0.05, "max_leaf_nodes": 127, "max_iter": 500},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 1.0},
]
MIN_VOTES_GRID = [None, 100_000, 25_000]
POP_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
POP_SOURCES = ["all", "recent"]
BASELINE_WINDOWS = [None, 1825, 730, 365]

results: dict = {}
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# -- metrics -------------------------------------------------------------------


def per_user_metrics(recs: np.ndarray, users: np.ndarray, val: pd.DataFrame) -> pd.DataFrame:
    """One row per user: hits, precision/recall/NDCG@K, and hit quality among rated."""
    long = pd.DataFrame({
        "userId": np.repeat(users, K),
        "rank": np.tile(np.arange(1, K + 1), len(users)),
        "movieId": recs.ravel(),
    })
    long = long[long["movieId"] >= 0].merge(val[["userId", "movieId", "rating"]],
                                            on=["userId", "movieId"], how="left")
    long["hit"] = long["rating"] >= LIKE_THRESHOLD
    long["rated"] = long["rating"].notna()
    long["gain"] = long["hit"] / np.log2(long["rank"] + 1)
    g = long.groupby("userId").agg(n_recs=("movieId", "size"), hits=("hit", "sum"),
                                   rated=("rated", "sum"), dcg=("gain", "sum"))
    g = g.reindex(users).fillna(0)

    n_rel = val[val["rating"] >= LIKE_THRESHOLD].groupby("userId").size()
    g["n_relevant"] = n_rel.reindex(users).fillna(0).to_numpy()
    discounts = np.cumsum(1 / np.log2(np.arange(2, K + 2)))
    ideal = np.where(g["n_relevant"] > 0,
                     discounts[np.minimum(g["n_relevant"], K).astype(int) - 1], np.nan)
    g["precision"] = g["hits"] / K
    g["recall"] = np.where(g["n_relevant"] > 0, g["hits"] / g["n_relevant"].clip(lower=1), np.nan)
    g["ndcg"] = g["dcg"] / ideal
    g["hit_rate"] = (g["hits"] > 0).astype(float)
    g["liked_share_of_rated"] = np.where(g["rated"] > 0, g["hits"] / g["rated"].clip(lower=1), np.nan)
    return g


def summarize(g: pd.DataFrame, recs: np.ndarray, popularity: pd.Series) -> dict:
    ids = recs[recs >= 0]
    head = set(popularity.nlargest(max(1, int(len(popularity) * 0.01))).index)
    return {
        "precision_at_10": float(g["precision"].mean()),
        "recall_at_10": float(g["recall"].mean()),
        "ndcg_at_10": float(g["ndcg"].mean()),
        "hit_rate_at_10": float(g["hit_rate"].mean()),
        "liked_share_of_rated_recs": float(g["liked_share_of_rated"].mean()),
        "mean_recs_per_user": float(g["n_recs"].mean()),
        "distinct_films_recommended": int(len(np.unique(ids))),
        "share_recs_top1pct_films": float(np.isin(ids, list(head)).mean()),
        "median_train_ratings_of_recs": float(np.median(popularity.reindex(ids).fillna(0))),
        "n_users": int(len(g)),
    }


def bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 2000) -> dict:
    """Paired bootstrap over users of mean(a - b)."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n_users": int(len(d))}


# -- data ----------------------------------------------------------------------


def load(quick: bool):
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    if quick:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(train["userId"].unique(), 15_000, replace=False)
        train, val = train[train["userId"].isin(keep)], val[val["userId"].isin(keep)]
    return mf, train, val


def training_rows(train: pd.DataFrame, max_users: int, target_start: float = TARGET_START):
    hist = train[train["timestamp"] < target_start]
    tgt = train[train["timestamp"] >= target_start]
    counts = hist["userId"].value_counts()
    eligible = np.intersect1d(counts[counts >= MIN_TRAIN_RATINGS].index, tgt["userId"].unique())
    rng = np.random.default_rng(SEED)
    users = rng.choice(eligible, min(max_users, len(eligible)), replace=False)
    tgt = (tgt[tgt["userId"].isin(users)].sample(frac=1, random_state=SEED)
           .groupby("userId").head(MAX_TARGETS_PER_USER))
    # Leakage guard: every feature comes strictly from before every target.
    assert hist["timestamp"].max() < target_start <= tgt["timestamp"].min()
    results["training_rows"] = {
        "history_ratings": int(len(hist)), "target_ratings": int(len(tgt)),
        "users": int(len(users)),
        "target_window_start": str(pd.Timestamp(target_start, unit="s").date()),
    }
    return hist, tgt


def cohort(train: pd.DataFrame, val: pd.DataFrame):
    tc, vc = train["userId"].value_counts(), val["userId"].value_counts()
    users = np.intersect1d(tc[tc >= MIN_TRAIN_RATINGS].index, vc[vc >= MIN_EVAL_RATINGS].index)
    rng = np.random.default_rng(SEED)
    users = rng.permutation(users)
    half = len(users) // 2
    return np.sort(users[:half]), np.sort(users[half:])


# -- scoring -------------------------------------------------------------------


def predict_users(model: HybridRecommender, train, users, only=None, batch=128):
    """Yield (users, history matrix, prediction block) in batches.

    ``only`` limits prediction to candidate films (the rest come back NaN and are
    never ranked), which is what keeps a full run inside an hour.
    """
    train = train[train["userId"].isin(users)]
    for start in range(0, len(users), batch):
        ub = users[start : start + batch]
        H = model.profile_.encode(train, ub)
        yield ub, H, model.predict_block(H, only)


def rank_grid(model, train, users, grid):
    """Recommendations for every (min_votes, weight, source) setting in one pass."""
    out = {key: [] for key in grid}
    masks = {mv: model.candidate_mask(min_votes=mv, max_year=TRAIN_END_YEAR)
             for mv in {key[0] for key in grid}}
    union = np.logical_or.reduce(list(masks.values()))
    for _, H, pred in predict_users(model, train, users, only=union):
        for mv, w, src in grid:
            out[(mv, w, src)].append(model.rank(pred, H, K, mask=masks[mv],
                                                popularity_weight=w, popularity_source=src))
    return {key: np.concatenate(v) for key, v in out.items()}


def rmse_report(model, train, val, users) -> dict:
    v = val[val["userId"].isin(users)]
    X, pos = model.profile_.pair_features(train, v)
    y = v["rating"].to_numpy(np.float32)[pos]
    pred = np.clip(model.regressor_.predict(X), 0.5, 5.0)
    names = model.profile_.feature_names_
    bias = np.clip(X[:, names.index("baseline_pred")], 0.5, 5.0)
    mu = model.profile_.mu_

    def rmse(p):
        return float(np.sqrt(np.mean((p - y) ** 2)))

    return {"n_ratings": int(len(y)), "pct_validation_ratings_scorable": float(len(y) / len(v) * 100),
            "hybrid_rmse": rmse(pred), "hybrid_mae": float(np.mean(np.abs(pred - y))),
            "bias_only_rmse": rmse(bias), "global_mean_rmse": rmse(np.full_like(y, mu)),
            "_pred": pred, "_y": y, "_pos": pos}


# -- figures -------------------------------------------------------------------


def fig_comparison(rep: dict) -> None:
    metrics = [("precision_at_10", "Precision@10"), ("recall_at_10", "Recall@10"),
               ("ndcg_at_10", "NDCG@10"), ("hit_rate_at_10", "Hit rate@10")]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, key, label, color in [(-w / 2 - 0.01, "baseline", "Popularity baseline", viz.SERIES_2),
                                   (w / 2 + 0.01, "hybrid", "Hybrid model", viz.SERIES_1)]:
        vals = [rep[key][m] for m, _ in metrics]
        bars = ax.bar(x + off, vals, w, color=color, label=label, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=viz.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Mean over report-half users")
    ax.set_title("Hybrid model vs. popularity baseline on validation ratings")
    viz.style_axes(ax)
    viz.legend(ax, loc="upper left")
    fig.savefig(FIGURES_DIR / "fig6_hybrid_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)


def fig_by_group(table: pd.DataFrame, group: str, title: str, fname: str) -> None:
    t = table[table["group"] == group]
    x = np.arange(len(t))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, col, label, color in [(-w / 2 - 0.01, "baseline_precision", "Popularity baseline", viz.SERIES_2),
                                   (w / 2 + 0.01, "hybrid_precision", "Hybrid model", viz.SERIES_1)]:
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


def fig_ablation(abl: dict) -> None:
    names = list(abl)
    vals = [abl[n]["precision_at_10"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(np.arange(len(names)), vals, 0.6, color=viz.SERIES_1, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=viz.INK_SECONDARY)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Precision@10 (report half)")
    ax.set_title("What each feature group contributes")
    viz.style_axes(ax, xgrid=True, ygrid=False)
    fig.savefig(FIGURES_DIR / "fig8_hybrid_ablation.png", bbox_inches="tight")
    plt.close(fig)


# -- main ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sealed_before = (RESERVED_TEST_DIR / "ratings_test.parquet").stat().st_atime

    mf, train, val = load(args.quick)
    popularity = train.groupby("movieId").size()
    tune_users, report_users = cohort(train, val)
    log(f"cohort: {len(tune_users):,} tune / {len(report_users):,} report users")
    results["cohort"] = {"tune_users": int(len(tune_users)), "report_users": int(len(report_users))}

    # Baseline -----------------------------------------------------------------
    log("baseline: tuning the popularity window...")
    base_tune = {}
    for wd in BASELINE_WINDOWS:
        b = PopularityRecommender(window_days=wd).fit(train)
        recs = b.recommend(b.history_matrix(train, tune_users), K)
        base_tune[str(wd)] = summarize(per_user_metrics(recs, tune_users, val), recs, popularity)
    best_wd = max(BASELINE_WINDOWS, key=lambda wd: base_tune[str(wd)]["precision_at_10"])
    results["baseline_tuning"] = base_tune
    results["baseline_selected"] = {"window_days": best_wd}
    baseline = PopularityRecommender(window_days=best_wd).fit(train)
    log(f"  selected window_days={best_wd}")

    # Hybrid: training matrix, then tune ---------------------------------------
    hist, tgt = training_rows(train, 3000 if args.quick else MAX_TRAIN_USERS)
    log(f"hybrid: building {len(tgt):,} training rows from {len(hist):,} history ratings...")
    proto = HybridRecommender(mf, max_year=TRAIN_END_YEAR)
    X, y = proto.fit_features(hist, tgt)
    hist_profile = proto.profile_
    log("  rebuilding population state from the full training split...")
    full_profile = hist_profile.__class__(**hist_profile.get_params()).fit(train)

    grid = [(mv, w, src) for mv in MIN_VOTES_GRID for w in POP_WEIGHT_GRID for src in POP_SOURCES]
    tuning, best = [], None
    for i, params in enumerate(HGB_GRID):
        m = HybridRecommender(mf, regressor_params=params, max_year=TRAIN_END_YEAR)
        m.profile_ = hist_profile
        m.fit_regressor(X, y)
        m.profile_ = full_profile
        log(f"  config {i}: {params} ({m.regressor_.n_iter_} trees); ranking tune users...")
        all_recs = rank_grid(m, train, tune_users, grid)
        for (mv, w, src), recs in all_recs.items():
            s = summarize(per_user_metrics(recs, tune_users, val), recs, popularity)
            row = {"config": i, **params, "min_votes": mv, "popularity_weight": w,
                   "popularity_source": src, **s}
            tuning.append(row)
            if best is None or s["precision_at_10"] > best[0]["precision_at_10"]:
                best = (row, m, mv, w, src)
    pd.DataFrame(tuning).to_csv(DOCS_DIR / "hybrid_tuning.csv", index=False)
    row, model, best_mv, best_w, best_src = best
    model.min_votes, model.popularity_weight, model.popularity_source = best_mv, best_w, best_src
    results["hybrid_selected"] = {"regressor_params": HGB_GRID[row["config"]],
                                  "min_votes": best_mv, "popularity_weight": best_w,
                                  "popularity_source": best_src,
                                  "tune_precision_at_10": row["precision_at_10"],
                                  "n_training_rows": int(model.n_train_rows_),
                                  "n_trees": int(model.regressor_.n_iter_),
                                  "imdb_calibration": full_profile.imdb_calibration_}
    log(f"  selected config {row['config']}, min_votes={best_mv}, weight={best_w} ({best_src})")

    # Report half ----------------------------------------------------------------
    log("report half: scoring hybrid and baseline...")
    mask = model.candidate_mask(max_year=TRAIN_END_YEAR)
    h_recs = np.concatenate([model.rank(pred, H, K, mask=mask)
                             for _, H, pred in predict_users(model, train, report_users, mask)])
    b_recs = baseline.recommend(baseline.history_matrix(train, report_users), K)
    gh = per_user_metrics(h_recs, report_users, val)
    gb = per_user_metrics(b_recs, report_users, val)
    rep = {"hybrid": summarize(gh, h_recs, popularity), "baseline": summarize(gb, b_recs, popularity)}
    rep["difference_hybrid_minus_baseline"] = {
        m: bootstrap_diff(gh[m], gb[m]) for m in ("precision", "recall", "ndcg", "hit_rate")}
    rep["recommendable_films"] = {"hybrid": int(mask.sum()),
                                  "baseline": int(np.isfinite(baseline.scores_).sum())}
    results["report"] = rep

    log("report half: rating accuracy...")
    acc = rmse_report(model, train, val, report_users)
    results["rating_accuracy"] = {k: v for k, v in acc.items() if not k.startswith("_")}

    # Error analysis ------------------------------------------------------------
    log("error analysis...")
    tr_u = train[train["userId"].isin(report_users)].groupby("userId").agg(
        n_train=("rating", "size"), mean_train=("rating", "mean"), last=("timestamp", "max"))
    first_val = val[val["userId"].isin(report_users)].groupby("userId")["timestamp"].min()
    tr_u["gap_days"] = (first_val - tr_u["last"]) / 86400
    groups = {
        "history_size": pd.cut(tr_u["n_train"], [19, 49, 99, 249, 999, np.inf],
                               labels=["20-49", "50-99", "100-249", "250-999", "1000+"]),
        "user_generosity": pd.qcut(tr_u["mean_train"], 3,
                                   labels=["harsh (low mean)", "middle", "generous (high mean)"]),
        "gap_since_last_train_rating": pd.cut(tr_u["gap_days"], [-1, 30, 365, 1825, np.inf],
                                              labels=["<1 month", "1-12 months", "1-5 years", "5+ years"]),
    }
    rows = []
    for gname, buckets in groups.items():
        for bucket in buckets.cat.categories:
            ids = buckets.index[buckets == bucket]
            if len(ids) == 0:
                continue
            rows.append({"group": gname, "bucket": bucket, "n_users": int(len(ids)),
                         "hybrid_precision": gh.loc[ids, "precision"].mean(),
                         "baseline_precision": gb.loc[ids, "precision"].mean(),
                         "hybrid_recall": gh.loc[ids, "recall"].mean(),
                         "baseline_recall": gb.loc[ids, "recall"].mean()})
    by_group = pd.DataFrame(rows)
    by_group.to_csv(DOCS_DIR / "hybrid_error_by_user_group.csv", index=False)

    # Rating error by film popularity and by genre.
    v = val[val["userId"].isin(report_users)].iloc[acc["_pos"]].copy()
    v["pred"], v["err"] = acc["_pred"], acc["_pred"] - acc["_y"]
    v["film_train_ratings"] = popularity.reindex(v["movieId"]).fillna(0).to_numpy()
    v["popularity_bucket"] = pd.cut(v["film_train_ratings"], [-1, 0, 19, 99, 999, np.inf],
                                    labels=["0 (IMDb only)", "1-19", "20-99", "100-999", "1000+"])
    err_pop = v.groupby("popularity_bucket", observed=True).agg(
        n=("err", "size"), rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))),
        mean_error=("err", "mean")).reset_index()
    err_pop.to_csv(DOCS_DIR / "hybrid_error_by_film_popularity.csv", index=False)
    g = v.merge(mf[["movieId", "genres"]], on="movieId")
    g = g.assign(genre=g["genres"].str.split("|")).explode("genre")
    err_genre = g.groupby("genre").agg(
        n=("err", "size"), rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))),
        mean_error=("err", "mean")).reset_index().sort_values("rmse", ascending=False)
    err_genre.to_csv(DOCS_DIR / "hybrid_error_by_genre.csv", index=False)
    results["error_analysis"] = {"by_user_group": by_group.to_dict("records"),
                                 "by_film_popularity": err_pop.to_dict("records"),
                                 "by_genre": err_genre.to_dict("records")}

    # Ablations -----------------------------------------------------------------
    log("ablations...")
    ablations = {"Full model": rep["hybrid"]}
    for label, groups_kept in [("Without IMDb features", ("people", "knn")),
                               ("Without actors & director", ("imdb", "knn")),
                               ("Without collaborative (kNN)", ("imdb", "people")),
                               ("Genre + user/movie stats only", ())]:
        log(f"  {label}")
        a = HybridRecommender(mf, profile_params={"feature_groups": groups_kept},
                              regressor_params=HGB_GRID[row["config"]], min_votes=best_mv,
                              popularity_weight=best_w, popularity_source=best_src,
                              max_year=TRAIN_END_YEAR)
        a.fit(hist, tgt)
        a.profile_ = a.profile_.__class__(**a.profile_.get_params()).fit(train)
        am = a.candidate_mask(max_year=TRAIN_END_YEAR)
        recs = np.concatenate([a.rank(p, H, K, mask=am)
                               for _, H, p in predict_users(a, train, report_users, am)])
        ablations[label] = summarize(per_user_metrics(recs, report_users, val), recs, popularity)
        ablations[label]["rmse"] = rmse_report(a, train, val, report_users)["hybrid_rmse"]
    ablations["Full model"]["rmse"] = results["rating_accuracy"]["hybrid_rmse"]
    ablations["Popularity baseline"] = rep["baseline"]
    results["ablations"] = ablations

    # Outputs -------------------------------------------------------------------
    viz.use_project_style()
    fig_comparison(rep)
    fig_by_group(by_group, "history_size", "Precision@10 by how many films the user had rated",
                 "fig7_hybrid_by_history_size.png")
    fig_ablation({k: v for k, v in ablations.items()})

    sealed_after = (RESERVED_TEST_DIR / "ratings_test.parquet").stat().st_atime
    results["reserved_test_untouched"] = bool(sealed_after == sealed_before)
    results["quick_mode"] = args.quick
    out = DOCS_DIR / ("hybrid_results_quick.json" if args.quick else "hybrid_results.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out}")
    print(json.dumps({k: results[k] for k in ("hybrid_selected", "report", "rating_accuracy")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
