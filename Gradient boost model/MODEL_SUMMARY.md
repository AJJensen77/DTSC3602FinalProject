# Gradient Boosted Hybrid Model — Summary for the Group

**Author:** Yager Range · **Live demo:** https://reelpicks.vercel.app ·
**API docs:** https://yagerr05--movie-recommender-api-fastapi-app.us-east.modal.run/docs

**Question (D1):** Can a personalized model built from a user's earlier ratings recommend
movies they go on to rate highly (3.5★+) better than recommending the most popular movies?

**Short answer:** Yes, by a small but statistically reliable margin. On 2,953 users the model
never saw during tuning, it got more good picks per 10 than a tuned popularity baseline
(precision@10 **0.175 vs 0.165**, 95% CI on the difference +0.007 to +0.014). It also
recommended 3.4× more distinct movies.

## 1. What the model does

For each (user, movie) pair it **predicts the star rating** the user would give, then
recommends the 10 movies with the highest score among those predicted at **3.5★ or higher**.

It learns from 20 features in four groups:

| Group | Features |
|---|---|
| The user | number of ratings, average rating, spread, share of ratings ≥3.5 |
| The movie | MovieLens rating count and mean, ratings in the last year ("trending"), **blended prior** (MovieLens mean pulled toward the movie's IMDb rating when MovieLens ratings are few), IMDb rating and votes, year, a simple baseline guess |
| User × movie taste | how much above/below their own average the user rates this movie's **genres**, **top-3 billed actors**, and **director** (plus how much evidence each is based on) |
| Similar movies | item-kNN score: how the user rated movies that tend to be rated by the same people |

A **gradient-boosted regressor** (scikit-learn `HistGradientBoostingRegressor`, 500 trees,
learning rate 0.05) learns how to combine them. Each new tree is fit to the errors left by
the trees before it, so the model chips away at mistakes rather than averaging independent
trees like a random forest.

Final ranking score = predicted rating + 0.7 × log(ratings in the last year). The predicted
rating decides *whether* a movie is liked enough; recent popularity helps decide *which*
liked movies the user is likely to actually watch (see §4).

## 2. How it was tested (no leakage)

- Uses D2's chronological split only. **Reserved test data was never opened.**
- **Training rows:** ratings from 2014-10 to 2016-10 (209,300 rows, 3,904 users), each
  described using only that user's and the movies' data from **before** 2014-10.
- **Evaluation:** predict the validation period (2016-10 → 2019-11) for D2's scorable cohort
  (≥20 training ratings, ≥5 validation ratings). The cohort was split 50/50 at random:
  settings were tuned on one half (2,953 users) and **every number below comes from the other
  half** (2,953 users).
- **Hit** = a recommended movie the user later rated ≥3.5. **Precision@10** = hits out of 10.
- **Baseline:** the most-rated movies of the last 12 months among those with a damped mean
  ≥3.5 (the recency window was tuned too, so the baseline is as strong as we could make it).

## 3. Results (report half, 2,953 users)

| Metric | Gradient boost model | Popularity baseline | Difference (95% CI) |
|---|---|---|---|
| Precision@10 | **0.175** | 0.165 | +0.010 (+0.007, +0.014) |
| Recall@10 | **0.043** | 0.039 | +0.004 (+0.002, +0.005) |
| NDCG@10 | **0.190** | 0.180 | +0.010 (+0.006, +0.015) |
| Hit rate (≥1 good pick) | **62.6%** | 59.4% | +3.2 pts (+1.8, +4.6) |
| Liked share of picks the user watched | **86%** | 78% | |
| Distinct movies recommended | **763** | 223 | |

**Rating accuracy (secondary):** RMSE **0.81** vs 0.88 for a user+movie average and 1.02 for
the global mean.

**What each part contributes** (remove one group, retrain, same users):

| Version | Precision@10 | RMSE |
|---|---|---|
| Full model | **0.175** | **0.809** |
| Without IMDb features | 0.174 | 0.825 |
| Without similar-movies (kNN) | 0.172 | 0.820 |
| Without actors & director | 0.167 | 0.813 |
| Genre + user/movie stats only | 0.163 | 0.848 |
| Popularity baseline | 0.165 | — |

Every group helps; actors & director help ranking most, IMDb helps rating accuracy most.
Movies with **no** MovieLens training ratings (IMDb fill-in only) were predicted as accurately
(RMSE 0.79) as heavily rated ones (0.82).

## 4. Error analysis and honest caveats

- **Wins in every user group** (precision@10, model vs baseline): history size 20–49
  (0.199 vs 0.186) through 1000+ (0.202 vs 0.190); harsh raters (0.152 vs 0.140) and generous
  raters (0.180 vs 0.173). Largest gap for users with 100–249 ratings (0.163 vs 0.147).
- **Much of the edge depends on blending with popularity.** Ranked by predicted rating
  alone, precision@10 is only 0.040, even though those picks are the most *liked* when
  watched (92%). The model knows taste well but cannot learn from ratings alone whether
  someone will *watch* a film; popularity answers that. Honest framing: **"popular movies
  filtered and reordered by your taste beat popularity alone."**
- **The test favors popular movies.** A great niche pick the user never happened to watch
  counts as a miss, and MovieLens was already recommending popular films to these users.
- **Generous raters:** with an absolute 3.5 cutoff, someone averaging 4+ passes almost
  everything, so popularity drives their list more. We kept 3.5 on purpose ("they just like a
  lot of movies").
- **Applies to experienced users** (D2 cohort), not brand-new MovieLens users.
- **IMDb ratings are a 2026 snapshot**, so they include votes cast after 2016 — a mild
  predictor-timing caveat. Removing IMDb changes precision@10 by only 0.001.
- Not tested yet: D2's proposed post-2003 training window.

## 5. Live check and website

The deployed site (https://reelpicks.vercel.app) uses the **same regressor**, with movie
statistics refreshed through 2019-11. Sign up → rate 10+ movies → get 10 picks → filter by
genre. In my own test, every recommendation was a movie I had already watched and enjoyed;
after rating those and refreshing, it moves on to new picks.

The site only depends on one API call — ratings in, 10 movies out — so **if the group picks a
different model, it can be served behind the same API and website** without UI changes.

## 6. Where things are

Everything lives in `Gradient boost model/` (delete the folder to remove it). See
`Gradient boost model/README.md` for how to rerun the evaluation and redeploy.
The popularity baseline has its own subfolder, `baseline/`, with a standalone evaluation
(`baseline/evaluate_baseline.py`) on the same users and split, so **other group models can
be scored against the same baseline numbers**.
Detailed numbers: `results/hybrid_results.json`; figures: `figures/fig6–fig8`.

*AI assistance: built with Claude Code (code, evaluation, and drafting this summary), per the
course AI policy; modeling choices and testing reviewed by me.*
