# List Taste Model — Summary for the Group

**Author:** Ishika Bhattarai · **Live demo:** https://bhattaraiishika--ishika-list-model-demo-run.us-east.modal.run

**Question (D1):** Can a personalized model built from a user's earlier ratings recommend
movies they go on to rate highly (3.5★+) better than recommending the most popular movies?

**Short answer:** Yes. On the same 2,953 report-half users used for the gradient boost model,
the list taste model got **2.1 good picks per 10 vs 1.6 for the popularity baseline**
(precision@10 **0.208 vs 0.165**, 95% CI on the difference +0.038 to +0.049) and **vs 1.75 for
the gradient boost model** (0.175; approx. 95% CI on the difference +0.022 to +0.044). It is the
best of the three on every ranking metric and in 11 of 12 user groups, while relying less on
blockbusters than either.

## 1. What the model does

It **ranks** films directly from a user's watch lists; it does not predict star ratings.

1. **Taste fingerprints.** A truncated SVD (scikit-learn `TruncatedSVD`, 192 factors) of the
   user-by-film training matrix gives every film a vector of latent taste factors. Films
   watched by the same kinds of people get similar vectors. Columns are divided by
   `n_ratings^0.3` first so blockbusters don't dominate the factors.
2. **Lists → one taste profile.** Each listed film adds its fingerprint with a signed weight:

   | List | Weight |
   |---|---|
   | Completed | +1.0 (+0.25 if rated only 2.5–3★) |
   | Currently Watching | +0.8 |
   | Plan to Watch | +0.5 |
   | Dropped | −1.0 (pushes similar films down) |

3. **Score and rank.** Every film is scored against the profile, put on a per-user standard
   scale, and blended with how popular the film was in the last 365 days of training
   (weight 6.0). Listed films are never recommended.
4. **More like this** = the films with the most similar fingerprints (cosine similarity).

New users are folded in from their lists at request time — **no retraining** — which is what
the website needs (site users are not MovieLens users).

## 2. How it was tested (no leakage, same protocol as every group model)

- D2's chronological split. Fitted on the training split only (1995 → Oct 2016); hits judged
  on the validation split (Oct 2016 → Nov 2019). **Reserved test data never opened**
  (checked automatically by the file's access time).
- D2's scorable cohort (≥20 training ratings, ≥5 validation ratings), shuffled with seed
  3602 and split 50/50: **settings tuned on the tune half (2,953 users); every number below
  is from the report half (2,953 users)** — the same users as the gradient boost model.
- Metrics come from the shared code in `Gradient boost model/evaluate_hybrid.py`
  (`per_user_metrics`, `summarize`, `bootstrap_diff`). **Hit** = a recommended film the user
  later rated ≥3.5. The re-scored baseline reproduces the published 0.1647 exactly, which
  confirms the users and code are identical.
- **MovieLens has ratings, not lists**, so ratings stand in for lists: ≥3.5★ → Completed
  (liked), 2.5–3★ → Completed (so-so), ≤2★ → Dropped.
- Tuned on the tune half (108 settings): popularity damping {0.3, 0.6}, factors {64, 128, 192},
  popularity blend {0 … 10}, optional 3.5★ quality floor. Selected: 0.3 / 192 / 6.0 / off.

## 3. Results — three-way comparison (report half, 2,953 users)

| Metric | Popularity baseline | Gradient boost | **List taste model** |
|---|---|---|---|
| Precision@10 | 0.165 | 0.175 | **0.208** |
| Recall@10 | 0.039 | 0.043 | **0.052** |
| NDCG@10 | 0.180 | 0.190 | **0.225** |
| Hit rate (≥1 good pick of 10) | 59.4% | 62.6% | **71.0%** |
| Liked share of picks the user watched | 78% | **86%** | 84% |
| Distinct films recommended | 223 | **763** | 569 |
| Share of picks from top-1% films | 63% | 70% | **51%** |
| Median training ratings of a pick | 25,961 | 29,890 | **13,734** |
| Rating RMSE (secondary) | — | **0.81** | n/a (ranks, no star predictions) |

**Differences on precision@10 (95% CI):**

| Comparison | Difference | 95% CI | How computed |
|---|---|---|---|
| List model − baseline | **+0.043** | +0.038 to +0.049 | paired bootstrap over users |
| List model − gradient boost | **+0.033** | +0.022 to +0.044 | unpaired normal approximation* |
| Gradient boost − baseline | +0.010 | +0.007 to +0.014 | paired bootstrap (Yager's run) |

\*The gradient boost model's per-user results weren't saved, so this interval uses published
aggregates and the larger measured per-user spread. Ignoring the pairing makes it wider than a
paired test would be; it still excludes zero. Same pattern for recall, NDCG, and hit rate
(`results/three_way_comparison.csv`).

**What each part of the list model contributes:**

| Version | Precision@10 | Distinct films |
|---|---|---|
| Full list model | **0.208** | 569 |
| Taste only (no popularity blend) | 0.183 | 2,696 |
| All lists equal (no negative weights) | 0.194 | 547 |
| Dropped ignored | 0.208 | 564 |
| Popularity baseline | 0.165 | 223 |

Taste alone already beats both the baseline and the gradient boost model (0.183 vs 0.165 /
0.175). The popularity blend adds +0.025.

## 4. Error analysis

**Precision@10 by user group** (same buckets as the gradient boost model;
`results/three_way_by_user_group.csv`, fig13):

| Group | Bucket (n) | Baseline | Gradient boost | List model |
|---|---|---|---|---|
| History size | 20–49 (251) | 0.186 | 0.199 | **0.227** |
| | 50–99 (347) | 0.165 | 0.171 | **0.219** |
| | 100–249 (731) | 0.147 | 0.163 | **0.203** |
| | 250–999 (1,275) | 0.164 | 0.172 | **0.203** |
| | 1000+ (349) | 0.190 | 0.202 | **0.214** |
| Generosity | harsh (985) | 0.140 | 0.152 | **0.176** |
| | middle (984) | 0.182 | 0.193 | **0.220** |
| | generous (984) | 0.173 | 0.180 | **0.229** |
| Gap before next rating | <1 month (563) | 0.181 | 0.196 | **0.244** |
| | 1–12 months (1,250) | 0.142 | 0.154 | **0.201** |
| | 1–5 years (993) | 0.159 | 0.165 | **0.186** |
| | **5+ years (147)** | **0.337** | **0.337** | 0.282 |

List-model-only groups (vs baseline): niche taste 0.188 vs 0.139, mainstream taste 0.231 vs
0.196; users with many low ratings 0.173 vs 0.131.

- **Best in 11 of 12 groups.** The list model's edge is largest for users with modest
  histories (50–249 ratings) and for niche-taste users — the people a "most popular" list
  serves worst.
- **Users returning after 5+ years** (147) are the one loss: their lists describe old taste,
  and what they rate on return is mostly what's popular *now*. Both personalized models tie
  or lose to the baseline there; a site could weight recent list activity more.
- **Less popularity-driven than the gradient boost model.** 51% of its picks come from the
  top 1% of films (gradient boost 70%, baseline 63%), and a typical pick has half as many
  training ratings. It still almost never picks films with <1,000 training ratings.
- **By genre of recommended film:** weakest hit rates for War (0.16), Romance (0.19),
  Animation (0.20); strongest for Sci-Fi and action/adventure.

## 5. Comparison with the gradient boost model — honest caveats

**Where the list model is better:** higher precision, recall, NDCG and hit rate; wins in
almost every user group; less concentrated on blockbusters; much simpler (no IMDb data,
cast/director tables, or second training stage); no IMDb-snapshot timing caveat; natively
handles Plan to Watch / Dropped lists the site will collect.

**Where the gradient boost model is better:** it predicts star ratings (RMSE 0.81) and so can
explain *how much* a user will like a film; slightly higher share of liked picks among those
watched (86% vs 84%); more distinct films (763 vs 569); and it can score films with **no**
MovieLens ratings via IMDb fill-in, while the list model only covers films with ≥20 training
ratings. The catalogs end up similar in size (14,417 recommendable films vs 14,367), but only
the gradient boost model can reach films with little or no MovieLens history, such as new
releases.

**Caveats for the list model:**
- **Lists are simulated** from ratings. *Watching* and *Plan to Watch* never occur in
  MovieLens; their weights are untested design choices. The Dropped weight made no
  measurable difference offline.
- **Heavy popularity blend (6.0).** Best described as "recently popular films re-ranked by your
  taste." The taste-only variant (0.183, 2,696 distinct films) is the option if the group
  values discovery over raw precision.
- **The offline test favors popular films** — a great niche pick the user never happened to
  rate counts as a miss.
- **Experienced users only** (D2 cohort); an empty list falls back to popular films.
- Validation numbers only; the reserved test set is for D6.

## 6. Where things are

Everything lives in `Ishika's model/` on the `D3` branch.

| File | What it is |
|---|---|
| `listmodel/recommender.py` | `ListTasteRecommender`: fit, list encoding, `recommend_from_lists`, `similar_movies` |
| `evaluate_list_model.py` | Tuning, report-half results vs baseline, variants, error analysis, saves the model |
| `compare_models.py` | Three-way comparison table and figures (fig12, fig13) |
| `RESULTS.md` | Detailed results for the list model alone |
| `demo_app/` | Live Streamlit demo and its Modal deployment |
| `results/`, `figures/` | All outputs (fig9–fig13) |

```bash
uv run python "Ishika's model/evaluate_list_model.py"   # ~13 min, ~3 GB RAM
uv run python "Ishika's model/compare_models.py"        # ~1 min
```

*AI assistance: built with Claude Code (model code, evaluation, comparison, and drafting this
summary), per the course AI policy.*
