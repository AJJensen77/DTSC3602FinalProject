"""Report-half check for the site's popularity setting (weight 2 instead of 6).

Ishika tuned the List taste model's popularity weight to 6.0 on the tune half. The
FilmPalate site uses 2.0, trading a little precision for much more variety. This
script scores both on the *report* half with the group's protocol and metric code
(train split only, D2 cohort, hit = validation rating >= 3.5), with paired
bootstrap CIs against the popularity baseline and against weight 6.

Usage (repo root, ~5 min):
    uv run python "List model app/evaluate_popularity_weight.py"
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "List taste model", ROOT / "Gradient boost model"):
    sys.path.insert(0, str(p))

import evaluate_hybrid as ev  # noqa: E402
from baseline.popularity import PopularityRecommender  # noqa: E402
from dtsc_3602_final_project.paths import PROCESSED_DIR  # noqa: E402
from listmodel.recommender import ListTasteRecommender  # noqa: E402

ISHIKA = json.loads((ROOT / "List taste model" / "results" / "list_model_results.json")
                    .read_text())["selected"]
SITE_WEIGHT = 2.0
K, WINDOW = 10, 365


def main():
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "ratings_validation.parquet")
    popularity = train.groupby("movieId").size()
    _, users = ev.cohort(train, val)
    print(f"report half: {len(users):,} users", flush=True)

    model = ListTasteRecommender(n_factors=ISHIKA["n_factors"],
                                 popularity_damping=ISHIKA["popularity_damping"],
                                 window_days=WINDOW).fit(train)
    baseline = PopularityRecommender(window_days=WINDOW).fit(train)

    def run(m):
        recs = m.recommend(m.history_matrix(train, users), K)
        g = ev.per_user_metrics(recs, users, val)
        return g, ev.summarize(g, recs, popularity)

    out, per_user = {}, {}
    for name, w in (("site_weight_2", SITE_WEIGHT), ("ishika_weight_6", ISHIKA["popularity_weight"])):
        model.popularity_weight = w
        per_user[name], out[name] = run(model)
    per_user["baseline"], out["baseline"] = run(baseline)

    p = {k: g.set_index("userId")["precision"] if "userId" in g else g["precision"]
         for k, g in per_user.items()}
    out["site_minus_baseline"] = ev.bootstrap_diff(p["site_weight_2"], p["baseline"])
    out["site_minus_ishika"] = ev.bootstrap_diff(p["site_weight_2"], p["ishika_weight_6"])
    for k in ("site_weight_2", "ishika_weight_6", "baseline"):
        s = out[k]
        print(f"{k:16s} precision@10 {s['precision_at_10']:.4f}  ndcg {s['ndcg_at_10']:.4f}  "
              f"distinct films {s['distinct_films_recommended']:,}  "
              f"top-1% share {s['share_recs_top1pct_films']:.2f}")
    print("site - baseline:", out["site_minus_baseline"])
    print("site - weight 6:", out["site_minus_ishika"])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "popularity_weight_check.json").write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
