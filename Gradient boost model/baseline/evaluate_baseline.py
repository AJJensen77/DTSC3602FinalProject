"""Evaluate the popularity baseline on its own, with the exact protocol every model uses.

Same scorable cohort, same 50/50 tune/report user split (seed 3602), same metrics as
evaluate_hybrid.py, so any group model scored on the report half can be compared
directly against these numbers. The recency window is tuned on the tune half.

Usage (from the repo root):
    uv run python "Gradient boost model/baseline/evaluate_baseline.py"

Writes baseline/results/baseline_results.json and baseline_tuning.csv.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # shared evaluation code and packages

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402

RESULTS = HERE / "results"


def main() -> None:
    mf, train, val = ev.load(quick=False)
    popularity = train.groupby("movieId").size()
    tune_users, report_users = ev.cohort(train, val)
    print(f"cohort: {len(tune_users):,} tune / {len(report_users):,} report users", flush=True)

    rows = []
    for wd in ev.BASELINE_WINDOWS:
        b = PopularityRecommender(window_days=wd).fit(train)
        recs = b.recommend(b.history_matrix(train, tune_users), ev.K)
        s = ev.summarize(ev.per_user_metrics(recs, tune_users, val), recs, popularity)
        rows.append({"window_days": wd, **s})
        print(f"  window_days={wd}: tune precision@10 {s['precision_at_10']:.4f}", flush=True)
    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning["precision_at_10"].idxmax(), "window_days"]
    best = None if pd.isna(best) else int(best)

    b = PopularityRecommender(window_days=best).fit(train)
    recs = b.recommend(b.history_matrix(train, report_users), ev.K)
    report = ev.summarize(ev.per_user_metrics(recs, report_users, val), recs, popularity)
    titles = mf.set_index("movieId")["clean_title"]
    top10 = [titles.get(m, str(m)) for m in recs[0] if m >= 0]

    RESULTS.mkdir(exist_ok=True)
    tuning.to_csv(RESULTS / "baseline_tuning.csv", index=False)
    out = {
        "definition": ("Most-rated films in the last `window_days` of training among films with "
                       ">=20 training ratings and a damped mean (50 pseudo-ratings) >= 3.5; "
                       "the user's already-rated films are skipped."),
        "selected": {"window_days": best},
        "report_half": report,
        "example_top10_for_a_new_user": top10,
    }
    (RESULTS / "baseline_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["selected"] | {k: round(v, 4) for k, v in report.items()
                                        if isinstance(v, float)}, indent=2))


if __name__ == "__main__":
    main()
