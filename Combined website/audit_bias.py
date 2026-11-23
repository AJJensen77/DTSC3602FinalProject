"""Bias audit for the combined model (the D1/D2 risks), and the setting the site uses.

D1 flagged popularity bias; D2 found the top 1% of films hold 39% of training ratings
and promised to "check catalog coverage and performance on less-popular movies
alongside accuracy". This script does that for every candidate the site could serve:

* Popularity baseline, and the list taste model at popularity weight 6 (its
  accuracy-only tuned value) and 2 (the "fair share" value: its top-1% share matches
  the top 1%'s 39% share of all ratings, for -0.004 precision; also FilmPalate's choice).
* The combined model built on each, with gb_weight re-tuned on the tune half.

Measured on the report half (2,953 users): precision@10, hit rate, distinct films,
share of picks from the top 1% / top 10% most-rated films, "long-tail precision"
(hits on films outside the top 1%), personalization (mean overlap between two random
users' top-10 lists), and precision by user group. Then the same short-history check as
evaluate_combined.py for the chosen setting.

Writes results/bias_audit.json, results/bias_audit_by_group.csv,
results/bias_audit_few_ratings.csv and results/site_settings.json (read by build_site.py).
Reserved test split never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "List taste model", ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from combined.blend import CombinedRecommender  # noqa: E402
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402
from evaluate_combined import FEW_SIZES, WEIGHT_GRID, score_users  # noqa: E402

K = ev.K
RESULTS = HERE / "results"
LIST_POP_WEIGHTS = [6.0, 2.0]
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def bias_stats(recs, g, popularity, rng) -> dict:
    ids = recs[recs >= 0]
    ranked = popularity.sort_values(ascending=False)
    top1 = set(ranked.index[: max(1, len(ranked) // 100)])
    top10 = set(ranked.index[: max(1, len(ranked) // 10)])
    # Personalization: mean Jaccard overlap of two random users' top-10 lists.
    a, b = rng.integers(0, len(recs), (2, 4000))
    sets = [set(r[r >= 0]) for r in recs]
    overlap = np.mean([len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
                       for i, j in zip(a, b) if i != j])
    return {
        "precision_at_10": float(g["precision"].mean()),
        "hit_rate_at_10": float(g["hit_rate"].mean()),
        "distinct_films": int(len(np.unique(ids))),
        "share_top1pct": float(np.isin(ids, list(top1)).mean()),
        "share_top10pct": float(np.isin(ids, list(top10)).mean()),
        "mean_overlap_between_users": float(overlap),
    }


def long_tail_precision(recs, users, val, popularity) -> pd.Series:
    """Per-user hits on films outside the top 1% most-rated, divided by K."""
    ranked = popularity.sort_values(ascending=False)
    top1 = set(ranked.index[: max(1, len(ranked) // 100)])
    long = pd.DataFrame({"userId": np.repeat(users, K), "movieId": recs.ravel()})
    long = long[(long["movieId"] >= 0) & ~long["movieId"].isin(top1)]
    long = long.merge(val[["userId", "movieId", "rating"]], on=["userId", "movieId"])
    hits = long[long["rating"] >= ev.LIKE_THRESHOLD].groupby("userId").size()
    return hits.reindex(users, fill_value=0) / K


def user_groups(train, users) -> pd.DataFrame:
    t = train[train["userId"].isin(users)].groupby("userId")["rating"].agg(["size", "mean"])
    t = t.reindex(users)
    out = pd.DataFrame(index=users)
    out["history_size"] = pd.cut(t["size"], [0, 49, 99, 249, 999, np.inf],
                                 labels=["20-49", "50-99", "100-249", "250-999", "1000+"])
    out["user_generosity"] = pd.qcut(t["mean"], 3, labels=["harsh", "middle", "generous"])
    # Mainstream vs niche taste: how popular the films they rated were (median).
    pop = train.groupby("movieId").size()
    med = (train[train["userId"].isin(users)].assign(p=lambda d: d["movieId"].map(pop))
           .groupby("userId")["p"].median().reindex(users))
    out["taste"] = pd.qcut(med, 3, labels=["niche", "in-between", "mainstream"])
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(ev.SEED)
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    popularity = train.groupby("movieId").size()
    tune_users, report_users = ev.cohort(train, val)

    gb = joblib.load(HERE / "artifacts" / "gb_eval.joblib")
    lm = joblib.load(ROOT / "List taste model" / "artifacts" / "list_model.joblib")["model"]
    baseline = PopularityRecommender(window_days=365).fit(train)
    groups = user_groups(train, report_users)

    recs = {"baseline": baseline.recommend(baseline.history_matrix(train, report_users), K)}
    chosen = {}
    for w in LIST_POP_WEIGHTS:
        lm.popularity_weight = w
        comb = CombinedRecommender(lm, gb)
        log(f"list popularity weight {w}: tuning gb_weight on the tune half...")
        ids, s, pred, _ = score_users(comb, train, tune_users)
        best = max(WEIGHT_GRID, key=lambda gw: ev.per_user_metrics(
            comb.top_k(ids, comb.blend(s, pred, gw, False), K), tune_users, val)["precision"].mean())
        chosen[w] = best
        log(f"  gb_weight={best}; scoring the report half...")
        ids, s, pred, _ = score_users(comb, train, report_users)
        recs[f"list_model_pop{w:g}"] = comb.top_k(ids, s, K)
        recs[f"combined_pop{w:g}"] = comb.top_k(ids, comb.blend(s, pred, best, False), K)

    g = {k: ev.per_user_metrics(r, report_users, val) for k, r in recs.items()}
    table = {k: bias_stats(recs[k], g[k], popularity, rng) for k in recs}
    for k in recs:
        table[k]["long_tail_precision_at_10"] = float(
            long_tail_precision(recs[k], report_users, val, popularity).mean())
    # Yager's model alone, from his own results file (same report users).
    yr = json.loads((ROOT / "Gradient boost model" / "results" / "hybrid_results.json")
                    .read_text())["report"]["hybrid"]
    table["gradient_boost (Yager's reported)"] = {
        "precision_at_10": yr["precision_at_10"], "hit_rate_at_10": yr["hit_rate_at_10"],
        "distinct_films": yr["distinct_films_recommended"],
        "share_top1pct": yr["share_recs_top1pct_films"]}
    print(pd.DataFrame(table).T.round(3).to_string())

    # Pick the site setting: the least popularity-driven combined model that is still more
    # accurate than every single model (the list taste model at its tuned weight 6,
    # Yager's gradient boost model, and the popularity baseline).
    combos = {w: table[f"combined_pop{w:g}"] for w in LIST_POP_WEIGHTS}
    best_single = max(table[k]["precision_at_10"] for k in
                      ("baseline", "list_model_pop6", "gradient_boost (Yager's reported)"))
    ok = [w for w, c in combos.items() if c["precision_at_10"] > best_single]
    site_w = min(ok, key=lambda w: combos[w]["share_top1pct"])
    diff = {f"combined_pop{site_w:g}_minus_{k}": ev.bootstrap_diff(
        g[f"combined_pop{site_w:g}"]["precision"], g[k]["precision"])
        for k in recs if k != f"combined_pop{site_w:g}"}
    log(f"site setting: list popularity weight {site_w}, gb_weight {chosen[site_w]}")

    # Precision by user group (is any group left behind?).
    rows = []
    for col in groups.columns:
        for bucket, ids_ in groups.groupby(col, observed=True).groups.items():
            rows.append({"group": col, "bucket": bucket, "n_users": len(ids_),
                         **{k: g[k].loc[ids_, "precision"].mean() for k in recs}})
    by_group = pd.DataFrame(rows)
    by_group.to_csv(RESULTS / "bias_audit_by_group.csv", index=False)
    print(by_group.round(3).to_string(index=False))

    # Short histories for the chosen setting (decides when the GB step switches on).
    lm.popularity_weight = site_w
    comb = CombinedRecommender(lm, gb, gb_weight=chosen[site_w])
    hist = train[train["userId"].isin(report_users)].sort_values(
        ["userId", "timestamp", "movieId"], ascending=[True, False, True])
    hist["recency_rank"] = hist.groupby("userId").cumcount()
    # Two versions of "a user with N ratings":
    #   same_info:  both models only know the N most recent films (and skip only those);
    #   skip_seen:  both skip every film the user rated in training, but taste comes from
    #               N films (FilmPalate's evaluate_blend.py protocol).
    few = []
    full = train[train["userId"].isin(report_users)]
    for protocol in ("same_info", "skip_seen"):
        for n in FEW_SIZES:
            h = hist[hist["recency_rank"] < n]
            ex = full if protocol == "skip_seen" else None
            ids, s, pred, _ = score_users(comb, h, report_users, exclude_history=ex)
            b_hist = full if protocol == "skip_seen" else h
            rr = {"combined": comb.top_k(ids, comb.blend(s, pred), K),
                  "list_model": comb.top_k(ids, s, K),
                  "baseline": baseline.recommend(baseline.history_matrix(b_hist, report_users), K)}
            gg = {k: ev.per_user_metrics(x, report_users, val) for k, x in rr.items()}
            ci = ev.bootstrap_diff(gg["combined"]["precision"], gg["list_model"]["precision"])
            cb = ev.bootstrap_diff(gg["combined"]["precision"], gg["baseline"]["precision"])
            few.append({"protocol": protocol, "ratings_known": n,
                        **{f"{k}_precision_at_10": gg[k]["precision"].mean() for k in rr},
                        "combined_minus_list": ci["diff"], "ci95_low": ci["ci95_low"],
                        "ci95_high": ci["ci95_high"], "combined_minus_baseline": cb["diff"],
                        "cb_ci95_low": cb["ci95_low"], "cb_ci95_high": cb["ci95_high"]})
            log(f"  {protocol} N={n}: " + ", ".join(f"{k} {gg[k]['precision'].mean():.3f}" for k in rr))
    few = pd.DataFrame(few)
    few.to_csv(RESULTS / "bias_audit_few_ratings.csv", index=False)
    same = few[few["protocol"] == "same_info"]
    ok_n = same[same["combined_minus_list"] >= 0]["ratings_known"]
    min_gb = int(ok_n.min()) if len(ok_n) else 10**9

    settings = {"list_popularity_weight": site_w, "gb_weight": chosen[site_w],
                "demote_below_like": False, "min_ratings_for_gb": min_gb,
                "rule": ("least popularity-driven combined model that is still more accurate "
                         "(precision@10) than every single model")}
    (RESULTS / "site_settings.json").write_text(json.dumps(settings, indent=2))
    (RESULTS / "bias_audit.json").write_text(json.dumps(
        {"table": table, "gb_weight_by_list_pop_weight": {str(k): v for k, v in chosen.items()},
         "site_settings": settings, "differences": diff,
         "few_ratings": few.to_dict(orient="records")}, indent=2, default=float))
    log(f"wrote results; site settings {settings}")


if __name__ == "__main__":
    main()
