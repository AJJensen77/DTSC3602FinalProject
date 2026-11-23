"""Tune and test the combined model (list taste model + gradient boost) for the website.

Same protocol as both model folders: D2's chronological split, the scorable cohort
(>= 20 training / >= 5 validation ratings) split 50/50 into tune and report halves with
seed 3602, a hit = a recommended film the user went on to rate >= 3.5 in validation.
The reserved test split is never read.

* Yager's model is fit exactly as in his evaluation (HGB config 1, 2014-10 -> 2016-10
  targets, min_votes=100k, popularity 0.7 recent, max_year 2016), then its movie
  statistics are refreshed with the full training split - the version he reported.
* Ishika's list model is the saved, already-evaluated artifact (training split only).
* The blend's two knobs (gb_weight, demote_below_like) are picked on the tune half;
  everything reported comes from the report half.
* Short-history check: the report half again, with each user cut to their N most
  recent training ratings, to set how many ratings the gradient boost step needs.

Usage (repo root, after Gradient boost model/prepare_imdb.py):
    uv run python "Combined website/evaluate_combined.py"            # ~30-40 min
Writes results/combined_*.{json,csv} and artifacts/gb_eval.joblib (reused by the build).
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
from gbmodel.hybrid import HybridRecommender  # noqa: E402

K = ev.K
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"
GB_SETTINGS = dict(regressor_params=ev.HGB_GRID[1], min_votes=100_000,
                   popularity_weight=0.7, popularity_source="recent")
WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
DEMOTE_GRID = [False, True]
FEW_SIZES = [1, 3, 5, 10, 20]
BATCH = 128
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def fit_gb(mf, train):
    path = ARTIFACTS / "gb_eval.joblib"
    if path.exists():
        log("loading the evaluated gradient boost model from artifacts/gb_eval.joblib")
        return joblib.load(path)
    log("fitting the gradient boost model exactly as in Yager's evaluation...")
    hist, tgt = ev.training_rows(train, ev.MAX_TRAIN_USERS)
    gb = HybridRecommender(mf, max_year=ev.TRAIN_END_YEAR, **GB_SETTINGS).fit(hist, tgt)
    del hist, tgt
    log(f"  {gb.n_train_rows_:,} rows, {gb.regressor_.n_iter_} trees; refreshing stats with train")
    gb.refresh(train)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(gb, path, compress=3)
    return gb


def score_users(comb, history, users, gb_alone=False, exclude_history=None):
    """Candidates, list scores and GB predictions per user (+ Yager's own top-10).

    ``exclude_history`` (optional) is a fuller history whose films are also skipped.
    """
    lm, gb = comb.list_model, comb.gb_model
    history = history[history["userId"].isin(users)]
    if exclude_history is not None:
        exclude_history = exclude_history[exclude_history["userId"].isin(users)]
    gb_mask = gb.candidate_mask(max_year=ev.TRAIN_END_YEAR) if gb_alone else None
    ids, scores, preds, gb_recs = [], [], [], []
    for start in range(0, len(users), BATCH):
        ub = users[start : start + BATCH]
        ex = None if exclude_history is None else lm.history_matrix(exclude_history, ub)
        c_ids, c_s = comb.candidates(lm.history_matrix(history, ub), exclude=ex)
        H_gb = gb.profile_.encode(history, ub)
        pred, block = comb.gb_predictions(H_gb, c_ids, extra_mask=gb_mask)
        ids.append(c_ids), scores.append(c_s), preds.append(pred)
        if gb_alone:
            gb_recs.append(gb.rank(block, H_gb, K, mask=gb_mask))
    out = [np.concatenate(ids), np.concatenate(scores), np.concatenate(preds)]
    return (*out, np.concatenate(gb_recs) if gb_alone else None)


def metrics(recs, users, val):
    return ev.per_user_metrics(recs, users, val)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    mf = pd.read_parquet(PROCESSED_DIR / "movie_features.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    popularity = train.groupby("movieId").size()
    tune_users, report_users = ev.cohort(train, val)
    log(f"cohort: {len(tune_users):,} tune / {len(report_users):,} report users")

    gb = fit_gb(mf, train)
    lm = joblib.load(ROOT / "List taste model" / "artifacts" / "list_model.joblib")["model"]
    baseline = PopularityRecommender(window_days=365).fit(train)
    comb = CombinedRecommender(lm, gb)
    results: dict = {"n_candidates": comb.n_candidates}

    # Tune -------------------------------------------------------------------------
    log("tune half: candidates + gradient boost predictions...")
    ids, s, pred, _ = score_users(comb, train, tune_users)
    tuning, best = [], None
    for d in DEMOTE_GRID:
        for w in WEIGHT_GRID:
            recs = comb.top_k(ids, comb.blend(s, pred, w, d), K)
            sm = ev.summarize(metrics(recs, tune_users, val), recs, popularity)
            tuning.append({"gb_weight": w, "demote_below_like": d, **sm})
            if best is None or sm["precision_at_10"] > best["precision_at_10"]:
                best = tuning[-1]
    pd.DataFrame(tuning).to_csv(RESULTS / "combined_tuning.csv", index=False)
    comb.gb_weight, comb.demote_below_like = best["gb_weight"], bool(best["demote_below_like"])
    results["selected"] = {"gb_weight": comb.gb_weight, "demote_below_like": comb.demote_below_like,
                           "tune_precision_at_10": best["precision_at_10"]}
    log(f"selected gb_weight={comb.gb_weight}, demote={comb.demote_below_like} "
        f"(tune precision@10 {best['precision_at_10']:.4f})")

    # Report -----------------------------------------------------------------------
    log("report half: combined, list alone, gradient boost alone, popularity...")
    ids, s, pred, gb_recs = score_users(comb, train, report_users, gb_alone=True)
    recs = {
        "combined": comb.top_k(ids, comb.blend(s, pred), K),
        "list_model": comb.top_k(ids, s, K),
        "gradient_boost": gb_recs,
        "baseline": baseline.recommend(baseline.history_matrix(train, report_users), K),
    }
    g = {name: metrics(r, report_users, val) for name, r in recs.items()}
    rep = {name: ev.summarize(g[name], recs[name], popularity) for name in recs}
    for other in ("list_model", "gradient_boost", "baseline"):
        rep[f"combined_minus_{other}"] = {
            m: ev.bootstrap_diff(g["combined"][m], g[other][m])
            for m in ("precision", "recall", "ndcg", "hit_rate")}
    results["report"] = rep
    for name in recs:
        log(f"  {name:15s} precision@10 {rep[name]['precision_at_10']:.4f}")

    # Short histories ----------------------------------------------------------------
    log("short-history check on the report half...")
    hist = train[train["userId"].isin(report_users)].sort_values(
        ["userId", "timestamp", "movieId"], ascending=[True, False, True])
    hist["recency_rank"] = hist.groupby("userId").cumcount()
    few = []
    for n in FEW_SIZES:
        h = hist[hist["recency_rank"] < n]
        ids, s, pred, gb_recs = score_users(comb, h, report_users, gb_alone=True)
        r = {"combined": comb.top_k(ids, comb.blend(s, pred), K),
             "list_model": comb.top_k(ids, s, K),
             "gradient_boost": gb_recs,
             "baseline": baseline.recommend(baseline.history_matrix(h, report_users), K)}
        gg = {name: metrics(x, report_users, val) for name, x in r.items()}
        row = {"ratings_known": n, **{f"{name}_precision_at_10": gg[name]["precision"].mean()
                                      for name in r}}
        ci = ev.bootstrap_diff(gg["combined"]["precision"], gg["list_model"]["precision"])
        row |= {"combined_minus_list": ci["diff"], "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"]}
        few.append(row)
        log(f"  N={n}: " + ", ".join(f"{k.split('_precision')[0]} {v:.3f}"
                                     for k, v in row.items() if k.endswith("precision_at_10")))
    few = pd.DataFrame(few)
    few.to_csv(RESULTS / "combined_few_ratings.csv", index=False)
    # Use the gradient boost step from the first size where it no longer hurts.
    ok = few[few["combined_minus_list"] >= 0]["ratings_known"]
    comb.min_ratings_for_gb = int(ok.min()) if len(ok) else 10**9
    results["selected"]["min_ratings_for_gb"] = comb.min_ratings_for_gb
    results["few_ratings"] = few.to_dict(orient="records")
    log(f"min_ratings_for_gb = {comb.min_ratings_for_gb}")

    (RESULTS / "combined_results.json").write_text(json.dumps(results, indent=2, default=float))
    log("wrote results/combined_results.json")
    print(json.dumps({k: results[k] for k in ("selected",)}, indent=2))


if __name__ == "__main__":
    main()
