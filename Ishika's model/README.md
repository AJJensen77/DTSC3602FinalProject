# List taste model (candidate model #2)

**Author:** Ishika Bhattarai · **Live demo:** https://bhattaraiishika--ishika-list-model-demo-run.us-east.modal.run

A personalized recommender driven by a user's **watch lists** — *Completed*, *Currently
Watching*, *Plan to Watch*, *Dropped* — plus a **"More like this"** lookup for any film.
It is the second candidate model for D3, scored with exactly the same protocol as the
popularity baseline and the gradient boost model.

## How it works

1. **Taste fingerprints.** From the MovieLens training split, a truncated SVD
   (scikit-learn `TruncatedSVD`) gives every film a short vector of latent "taste
   factors". Films watched by the same kinds of people get similar vectors. Before the
   SVD, each film's column is divided by `n_ratings ** popularity_damping` so
   blockbusters don't dominate every factor.
2. **Lists become signed weights.**

   | List | Weight | Meaning |
   |---|---|---|
   | Completed | +1.0 | strongest evidence of taste (+0.25 if rated only 2.5–3★) |
   | Currently Watching | +0.8 | chose it and is still watching |
   | Plan to Watch | +0.5 | interest, not yet confirmed |
   | Dropped | −1.0 | evidence *against* similar films |

3. **"Because of your lists…"** The weighted lists are folded through the fingerprints
   into one taste score per film, then blended with recent popularity (the same
   365-day count the baseline uses). Films already on any list are never recommended.
4. **"More like this."** Cosine similarity between film fingerprints.

No model retraining is needed for a new user — their lists are folded in at request
time, which is what the website/API needs.

**MovieLens has ratings, not lists.** For training and evaluation, a rating ≥ 3.5 counts
as *Completed (liked)*, 2.5–3 as *Completed (so-so)*, and ≤ 2 as *Dropped*. *Watching* and
*Plan to Watch* never occur in MovieLens, so their weights are design choices that
**could not be validated offline**.

## Evaluation protocol (same as every group model)

- D2's chronological split; trained on the training split only; reserved test data never
  opened (checked by file access time, `reserved_test_untouched` in the results).
- D2's scorable cohort (≥ 20 training ratings, ≥ 5 validation ratings), shuffled with seed
  3602, split 50/50: **settings tuned on the tune half, every number reported from the
  report half**.
- Hit = recommended film the user later rated ≥ 3.5. Metrics from the shared code in
  `evaluate_hybrid.py`: precision@10 (main), recall@10, NDCG@10, hit rate, plus coverage.
- Paired bootstrap 95% CI on the per-user difference vs. the baseline.

Tuned on the tune half: popularity damping, number of factors, popularity blend weight,
and an optional 3.5★ quality floor (same as the baseline's).

## Results

Report half, 2,953 users: **precision@10 0.208 vs 0.165** for the popularity baseline
(95% CI on the difference +0.038 to +0.049), 569 vs 223 distinct films recommended.
Full comparison, variants, error analysis and limitations: **[`RESULTS.md`](RESULTS.md)**.

Note: like `Gradient boost model/`, this folder runs from a checkout of the `D2` branch
(it uses D2's `data/processed/`, `src/` package and `pyproject.toml`) with
`Gradient boost model/` alongside it (shared baseline + metric code).

## Files

| File | What it is |
|---|---|
| `listmodel/recommender.py` | `ListTasteRecommender` — fit, list encoding, `recommend_from_lists`, `similar_movies` |
| `evaluate_list_model.py` | Tuning, report-half comparison, variants, error analysis, figures, saves the artifact |
| `demo.py` | Try lists / "more like this" by hand |
| `demo_app/` | Live Streamlit demo (`app.py`) and its Modal deployment (`modal_app.py`) |
| `results/` | JSON + CSV outputs |
| `figures/` | fig9–fig11 |
| `artifacts/list_model.joblib` | Fitted model + film metadata incl. `tmdbId` (gitignored; rebuild with the evaluation script) |

## Run it

```bash
uv run python prepare_data.py                               # D2 prep, if data/processed is empty
uv run python "Ishika's model/evaluate_list_model.py"           # full run
uv run python "Ishika's model/evaluate_list_model.py" --quick   # 15k-user sample, ~2 min
uv run python "Ishika's model/demo.py"
uv run python "Ishika's model/demo.py" --similar "Spirited Away"
uv run streamlit run "Ishika's model/demo_app/app.py"           # demo page locally
uv run modal deploy "Ishika's model/demo_app/modal_app.py"     # redeploy the live demo
```

## For the website (D4/D5)

- `model.recommend_from_lists({"completed": [...], "watching": [...], "plan_to_watch": [...], "dropped": [...]}, k=10)`
- `model.similar_movies(movie_id, k=10)`
- The artifact includes each film's `tmdbId`, which the site can use with TMDB's
  `/movie/{id}/videos` endpoint to fetch a YouTube trailer for the hover preview.
  Lists themselves would be stored per user in Supabase (as the ReelPicks site already
  does for ratings).

*AI assistance: built with Claude Code (model code, evaluation script, README), per the
course AI policy.*
