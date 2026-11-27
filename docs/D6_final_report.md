# D6 — Final Report: Personalized Movie Recommendations vs. Popularity

**DTSC 3602 — Deliverable 6** · Ishika Bhattarai, A.J. Jensen, Srijith G Menon,
Yashvi Patel, Yager Range, Max Valentine

| | |
|---|---|
| Repository | https://github.com/AJJensen77/DTSC3602FinalProject — branch `D6`, commit `[FILL IN]` |
| Website (Vercel) | https://filmpalate.vercel.app |
| Model API (Modal) | https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run · [/docs](https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run/docs) |
| D2 dashboard (Streamlit on Modal) | https://ajjensen77--dtsc-3602-final-project-d2-run.us-east.modal.run |
| Postman | `FilmPalate/extras/postman/FilmPalate.postman_collection.json` (13 requests, 32 assertions) |

> **Short answer.** Yes. For experienced users, a personalized model built from their
> earlier ratings recommends films they go on to rate 3.5★ or higher more often than a
> tuned most-popular list. On validation, the frozen FilmPalate model found **2.0 good
> films per 10 recommendations vs 1.6** for popularity (precision@10 0.204 vs 0.165,
> 95% CI on the difference +0.032 to +0.045) while recommending **6× as many different
> films**. On the sealed test period (Nov 2019 – Oct 2023) it scored **[TEST P@10]** vs
> **[TEST BASELINE P@10]** (difference **[DIFF]**, 95% CI **[LOW, HIGH]**). The gain is
> real but modest, about half of it comes from a better choice of *which popular films*
> to show, and it applies only to users with an established history.

---

## 1. Problem and domain rationale

Streaming services and review sites show each visitor a short row of films. The person
using the row wants to find something they'll like without scrolling through thousands of
titles. The operator wants them to keep watching. The usual default is a **most-popular
list**: cheap, easy to explain and hard to beat, because popular films are popular for a
reason. It is also the same for everyone, so it serves people with niche taste worst and
keeps sending attention to films that already have plenty of it.

The decision our analysis supports is **which ten films to show a returning user**. A
personalized recommender costs more to build, run and explain. It is only worth it if it
measurably beats the popular list for the same people. That framed the approved question:

> **Can a personalized model built from a user's earlier ratings recommend movies they go
> on to rate highly (≥3.5★), better than recommending the most popular movies?**

- **Observation:** one user at a point in time, with every rating they made before it.
- **Outcome:** of the 10 films recommended, how many the user later rates ≥3.5★.
- **Population and period:** MovieLens users with ≥20 earlier ratings. Models learn from
  1995 – Oct 2016, are tuned on Oct 2016 – Nov 2019 and are tested on Nov 2019 – Oct 2023.
- **Available at prediction time:** the user's own earlier ratings, and other users'
  ratings from before the cutoff. Nothing from the future.

We chose movies because recommendation is one of the most widely used kinds of prediction
in consumer products. It also has a large, honest public dataset with real timestamps, so
the "earlier → later" question can be tested the way it would be used. We wanted to learn
whether personalization beats a strong simple rule once the evaluation is fair, and not
just in a tutorial setup that lets the model see the future.

## 2. Data

| Source | Role | Size |
|---|---|---|
| [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) (GroupLens) | Ratings: the signal and the target | 32,000,204 ratings, 200,948 users |
| [IMDb non-commercial datasets](https://datasets.imdbws.com/) | Title type (removes TV), plus genres, cast, directors, votes for newer films | `title.basics`, `.ratings`, `.principals`, `.crew` |
| TMDB API | Posters and trailers on the website only. Never used by the model. | — |

**Preparation (D2, `prepare_data.py`).** MovieLens is billed as a movie dataset, but its
catalog contains TV and nothing in it marks which titles are TV. We joined every title to
IMDb through `links.csv` and dropped episodic television on IMDb's `titleType`: 2,505
titles and 106,959 ratings (0.33%). The raw files had no duplicates, no out-of-range stars
and no orphan IDs. **31,893,245 ratings** were kept. The data dictionary (72 columns, 14
tables) is generated from the files the pipeline wrote, so it cannot drift from them
(`docs/data_dictionary.md`).

**Split.** We cut by time at the 70/15/15 quantiles of the rating timestamp: train to
2016-10-01, validation to 2019-11-02, test to 2023-10-13. All popularity counts, means and
user histories are computed from the earlier side of each cutoff only. The test split was
written to `data/reserved_test/` with a `SEALED.md` notice and was not read until the
model was frozen (§4.1).

**What the data showed (D2):**

1. **The rating scale changed** on 18 Feb 2003, when half-stars were introduced. They make
   up 42% of later ratings, so a 3.5★ threshold means slightly different things before and
   after.
2. **Popularity is extremely concentrated.** The top 1% of films hold 39% of training
   ratings and the top 10% hold 90%. That makes popularity a strong opponent, not a straw
   man.
3. **Most of the catalog is too sparse to learn from.** 49,419 of 85,080 films have no
   training ratings at all.
4. **Most future ratings come from new users.** 84% of validation ratings and 90% of test
   ratings come from users with no earlier history, and nothing built on a user's past can
   score them. We therefore evaluate an explicit **scorable cohort**: users with ≥20
   earlier ratings and ≥5 in the evaluation period. These users rate about 0.25★ lower
   than the users excluded (3.35 vs 3.60), so the cohort is not representative (§6).

## 3. Methods

### 3.1 Baseline: tuned popularity

Recommend the **most-rated films of the last 365 days** among films with ≥20 ratings and a
damped mean ≥3.5★ (each mean is pulled toward the global mean by 50 pseudo-ratings), and
skip anything the user has already rated. The window was tuned rather than assumed: all
years 0.105 → 5 years 0.144 → 2 years 0.156 → **1 year 0.168** (precision@10, tune half).
A recent window beats an all-time list by a wide margin, so this is the strongest
popularity rule we found.

### 3.2 Candidate models (D3)

| Model | Author | Idea |
|---|---|---|
| **Gradient boost hybrid** | Yager Range | Predicts each user's star rating from 20 features: user and film statistics, genre/actor/director affinity, item-kNN similarity and IMDb rating. Uses `HistGradientBoostingRegressor`, then ranks by predicted stars + 0.7 × log(recent popularity). |
| **List taste model** | Ishika Bhattarai | Truncated SVD (192 factors) of the user × film matrix, with columns damped by popularity^0.3, gives every film a "taste fingerprint." A user's profile is the signed sum of their films' fingerprints (liked +1.0, so-so +0.25, disliked −1.0). Films are scored against the profile, standardized per user and blended with recent popularity. |

### 3.3 Evaluation protocol (identical for every model)

- **Cohort.** D2's scorable validation cohort (5,906 users), shuffled with seed 3602 and
  split 50/50. Every setting was chosen on the **tune half** (2,953 users), and every
  number reported here comes from the **report half** (2,953 users). The shared code
  (`per_user_metrics`, `summarize`, `bootstrap_diff` in
  `Gradient boost model/evaluate_hybrid.py`) reproduces the baseline's 0.1647 exactly in
  every folder, which confirms that all models saw the same users and scoring.
- **Hit** = a recommended film the user later rated ≥3.5★. The **main metric is
  precision@10**, because the decision is "which ten films to show" and a wasted slot is
  the cost. Recall@10, NDCG@10 and hit rate (at least one good pick) are secondary. RMSE is
  only a diagnostic.
- **Uncertainty.** We use a 95% paired bootstrap over users for every model − baseline
  difference, since the same users are scored by both.
- **Bias metrics** sit alongside accuracy, as D2 required: distinct films recommended,
  share of picks from the top-1% films, overlap between two users' lists, and precision on
  films outside the top 1% ("long-tail precision").

### 3.4 Model selection (D3)

| Report half, 2,953 users | Precision@10 | Recall@10 | NDCG@10 | Hit rate | Distinct films | Top-1% share |
|---|---|---|---|---|---|---|
| Popularity baseline | 0.165 | 0.039 | 0.180 | 59.4% | 223 | 63% |
| Gradient boost hybrid | 0.175 | 0.043 | 0.190 | 62.6% | 763 | 70% |
| **List taste model** | **0.208** | **0.052** | **0.225** | **71.0%** | 569 | **51%** |

We selected the **list taste model**. It was best on every ranking metric. It beat the
baseline by +0.043 (95% CI +0.038, +0.049) and the gradient boost model by +0.033. It won
in 11 of 12 user groups and leaned on blockbusters less than either alternative. It is
also simpler: no IMDb features and no second training stage. It can fold in a new website
user's lists at request time without retraining, which the website requires. The gradient
boost model kept two real advantages, star-rating predictions (RMSE 0.81) and coverage of
films with almost no MovieLens history, but neither was the decision we were evaluating.

### 3.5 The frozen model: FilmPalate pipeline (D4)

The deployed model is a scikit-learn `Pipeline` (`FilmPalate/D4_submission/watchlist/pipeline.py`):

```
Pipeline([("encode", WatchlistEncoder()),   # custom transformer: lists → signed taste weights
          ("rank",   WatchlistRanker())])   # list taste model + newer films + cold start
```

It made three deliberate changes from Ishika's evaluated settings. Each was checked on the
tune/report halves before freezing:

1. **Popularity weight 2 instead of 6** for users with full histories. Precision@10 went
   from 0.208 to **0.204** (−0.005, CI −0.008 to −0.002) in exchange for **1,400 vs 569
   distinct films** and a top-1% share of 39% instead of 51%. We judged the small accuracy
   cost worth the gain in variety, which is the popularity-bias concern D1 raised.
2. **A higher popularity weight for short histories** (40 → 25 → 15 → 8 → 4 → 2 as ratings
   grow). At weight 2 the model *loses* to popularity until about 20 ratings (1 rating:
   0.105 vs 0.165). With the tuned schedule it beats popularity from the first rating
   (+0.003 at 1 rating, +0.015 at 10, +0.028 at 50).
3. **Films released 2017–2026** (2,684 with ≥10,000 IMDb votes) borrow the averaged
   fingerprint of their 10 most similar catalog films. As a new-release row they are
   ranked by IMDb popularity + rating, which beat most-voted in a 2015–16 hold-out
   backtest (0.188 vs 0.170).

The artifact is loaded once per server process and never retrained while serving. The API
validates input (422 for bad stars, duplicates or unknown IDs; 404 for an unknown film;
503 if the artifact is missing). `FilmPalate/extras/verify_deployment.py` confirms that
local and deployed outputs match exactly.

## 4. Validation and final test results

### 4.1 Test procedure

**Every choice above was frozen before the test split was opened.** The test protocol
mirrors validation, moved forward one period:

- **History** = all ratings before 2019-11-02 (train + validation), the same data the
  deployed model was refreshed on. **Test period** = 2019-11-02 → 2023-10-13.
- **Cohort** = D2's scorable test cohort: users with ≥20 ratings before the cutoff and ≥5
  ratings in the test period (D2 counted **3,302 users**; all are scored, since there is
  nothing left to tune).
- **Models:** the frozen FilmPalate pipeline (exact deployed settings) and the popularity
  baseline (365-day window, ending 2019-11-02). Same metric code, hit threshold and
  bootstrap as validation.
- The test file was read **[once / N times — FILL IN]** on **[DATE]** by
  `[script name]`. **[Disclose any re-runs here.]**

### 4.2 Results

| Metric | Popularity baseline | **FilmPalate (frozen)** | Difference (95% CI) |
|---|---|---|---|
| Precision@10 *(validation)* | 0.165 | **0.204** | +0.039 (+0.032, +0.045) |
| **Precision@10 *(test)*** | [ ] | [ ] | [ ] ([ ], [ ]) |
| Recall@10 *(test)* | [ ] | [ ] | [ ] |
| NDCG@10 *(test)* | [ ] | [ ] | [ ] |
| Hit rate *(test)* | [ ] | [ ] | [ ] |
| Distinct films *(test)* | [ ] | [ ] | |
| Share of picks from top-1% films *(test)* | [ ] | [ ] | |
| Long-tail precision@10 *(test)* | [ ] | [ ] | |

**Reading the result. [Rewrite once the numbers are in.]** If the test difference has a
CI above zero, the validation finding held up on unseen users and a later period. A
smaller margin than on validation would not be surprising: the test period includes
2020–2023, when viewing habits shifted, and its cohort is drawn from a later, smaller
group of users. If the CI includes zero, the honest conclusion is that the advantage did
not generalize. The rubric explicitly allows that finding, and we would report it as
such.

### 4.3 Error analysis

Precision@10 by user group (validation report half, list taste model at weight 6. The
weight-2 FilmPalate setting follows the same pattern, about 0.005 lower):

| Group | Bucket (users) | Baseline | List taste model |
|---|---|---|---|
| History size | 20–49 (251) | 0.186 | **0.227** |
| | 100–249 (731) | 0.147 | **0.203** |
| | 1000+ (349) | 0.190 | **0.214** |
| Rater generosity | harsh (985) | 0.140 | **0.176** |
| | generous (984) | 0.173 | **0.229** |
| Taste | niche (985) | 0.139 | **0.188** |
| | mainstream (984) | 0.196 | **0.231** |
| Gap before returning | < 1 month (563) | 0.181 | **0.244** |
| | **5+ years (147)** | **0.337** | 0.282 (CI −0.088, −0.020) |

- **The biggest gains go to the people a popular list serves worst:** niche-taste users
  (+0.049) and mid-history users (+0.056).
- **The one clear loss is users returning after 5+ years.** Their history describes old
  taste, and what they watch on return is mostly what is popular *now*. Both personalized
  models tie or lose here. A production system should weight recent activity more.
- **By genre**, hit rates are weakest for War (0.16), Romance (0.19) and Animation (0.20)
  picks, and strongest for Sci-Fi and action/adventure.
- **Short histories:** see §3.5 (2). Without the popularity schedule, a brand-new user
  would get worse picks than a popular list for their first ~20 ratings.
- Test-period group breakdown: **[ADD if computed; otherwise note it was not run.]**

## 5. Conclusions and usefulness

1. **The answer to the question is yes, with qualifications.** For users with ≥20 earlier
   ratings, personalization beat a tuned popularity list by about 4 percentage points of
   precision on validation **[and by X on test]**. That is roughly 0.4 more good films in
   every row of ten, about a 24% relative gain. Ranking metrics, hit rate and all 11 of 12
   user groups point the same way.
2. **Much of the gain is "better popular films for you," not hidden gems.** The
   taste-only variant (no popularity blend) scores 0.183. Still better than the baseline,
   but popularity adds another +0.025. Popularity answers *will they watch it*; taste
   answers *will they like it*. Both matter for the decision.
3. **The best accuracy setting and the best product setting differ.** We gave up 0.005
   precision for 6× the catalog coverage. A metric alone would not have made that choice;
   the bias check in D2 did.
4. **The model is not useful for brand-new users or for long-absent ones.** For them a
   popularity list, or a popularity-heavy blend, is the honest default, and the website
   does exactly that.

**Practical use.** For a site with returning users, the model is worth running: one
request per page view, no retraining, and it works from the lists people already keep.
FilmPalate demonstrates this end to end. Sign up, rate or list a few films, and the "For
you" page calls the live API, shows loading and error states and updates as lists change.

## 6. Bias and fallacy risks, with checks

| Risk | Why it matters here | Check we ran | Result |
|---|---|---|---|
| **Popularity bias** | With 39% of ratings on 1% of films, a model can score well by re-ranking blockbusters and add nothing new. | Distinct films, top-1% share, user-to-user overlap and long-tail precision, reported next to accuracy. | FilmPalate (weight 2) recommends 1,400 distinct films vs 223, takes 39% of picks from the top 1% vs 63%, and its top-10s overlap 2.1% between users vs 17.4%. Long-tail precision is 0.120 vs 0.074. Picks with <1,000 ratings still almost never appear. |
| **Selection (cohort) bias** | Only users with ≥20 earlier ratings can be scored. They are 15% of validation ratings and rate 0.25★ lower than everyone else. | Report results for the cohort only; break down by history size and generosity; test short histories separately. | The gain holds across history sizes and generosity groups, but the result **does not generalize to new users**. With ≤20 ratings the unadjusted model loses to popularity. |
| **Exposure / feedback-loop bias** | MovieLens already showed users recommendations, so what they rated was partly what they were shown. A good niche pick they were never shown counts as a miss. | Interpret precision as a *lower bound* for niche picks; report taste-only and long-tail variants. | Offline metrics likely **understate** the value of discovery and **overstate** popularity. Only a live A/B test could measure the effect of recommending. |
| **Prediction ≠ causation** | A hit means the user *later rated* the film highly, not that our recommendation *caused* it. | Language discipline: no causal claims; framed as "predicts which films they will like." | Holds; all claims in §5 are predictive. |
| **Overfitting the tuning set ("forking paths")** | 108 list-model settings and dozens of baseline, blend and weight settings were tried. | Tune half vs report half for every choice, then a sealed test split read once after freezing. | Tune → report drops were small (0.213 → 0.208). Test: **[FILL IN]**. |
| **Simulated lists** | MovieLens has ratings, not "Want to Watch" lists, so that weight (+0.5) was never validated. | Disclosed; Want to Watch counted as half a rating in the popularity schedule. | Untested offline. Needs real site data. |
| **Measurement drift** | Half-stars appeared in 2003, and the IMDb snapshot used for newer films is from 2026. | The list model uses neither IMDb features nor the star scale beyond the 3.5 cut. | Low risk for the main model. The new-release row uses current IMDb votes, which a real 2019 system could not have had. |

## 7. Limitations

- **Offline, historical evaluation only.** No live users were randomized. The results
  describe 2016–2023 MovieLens raters, a community of enthusiasts and not typical viewers.
- **Cold start is solved with popularity, not personalization.** The model's contribution
  starts at roughly 1–20 ratings, depending on the popularity schedule.
- **Catalog reach.** The core model only covers films with ≥20 ratings before Nov 2019.
  Newer films are reached by description similarity, which was not tested as a
  recommendation source for individual users (a within-user AUC of 0.51 suggests taste
  adds little there).
- **Two websites exist.** `Combined website/` (list model + gradient boost) reached 0.210
  at weight 2 on validation. It was built after the model choice was frozen, was not
  tested on the reserved split and is not the submitted model.

## 8. Reproducibility

Everything is in the repository. Raw data is gitignored; the README and each folder's
README list download links. Dependencies are pinned in `pyproject.toml` / `uv.lock`.
Credentials live only in `.env`, `web/.env.local` and the Vercel/Modal settings, never in
code.

```bash
uv sync
uv run python prepare_data.py                                   # D2: clean, split, seal test
uv run python "Gradient boost model/baseline/evaluate_baseline.py"
uv run python "List taste model/evaluate_list_model.py"         # D3 (~13 min)
uv run python "FilmPalate/D4_submission/build_model.py"         # D4 artifact (~10 min)
uv run python "FilmPalate/extras/verify_deployment.py"          # local = deployed
uv run python [D6 TEST SCRIPT]                                  # D6: reads the sealed split once
```

## 9. Contributions and AI assistance

| Member | Contribution |
|---|---|
| A.J. Jensen | D2 data preparation, audit, data dictionary, EDA figures, Supabase-backed Streamlit dashboard on Modal; [D6 — FILL IN] |
| Ishika Bhattarai | List taste model (D3, the selected model), its evaluation, three-way comparison, demo app; [FILL IN] |
| Yager Range | Gradient boost model and popularity baseline (D3); FilmPalate pipeline, API and Modal deployment (D4); website, Postman tests (D5); [FILL IN] |
| Srijith G Menon | [FILL IN] |
| Yashvi Patel | [FILL IN] |
| Max Valentine | [FILL IN] |

**AI assistance.** Claude Code was used throughout, within the course AI policy. It helped
write pipeline, evaluation, API and website code, debug deployments and draft
documentation, including this report. Every analytical choice (the research question,
cohort definition, metrics, model selection, popularity weights) was made and checked by
the group, and every number is produced by a script in the repository.

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context.
  *ACM Transactions on Interactive Intelligent Systems, 5*(4), Article 19.
  https://doi.org/10.1145/2827872
- GroupLens Research. *MovieLens 32M dataset.* https://grouplens.org/datasets/movielens/32m/
- IMDb. *IMDb non-commercial datasets.* https://developer.imdb.com/non-commercial-datasets/
- Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender algorithms on
  top-N recommendation tasks. *Proceedings of RecSys '10*, 39–46.
- Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). Controlling popularity bias in
  learning-to-rank recommendation. *Proceedings of RecSys '17*, 42–46.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*,
  2825–2830.
- This product uses the TMDB API but is not endorsed or certified by TMDB.
