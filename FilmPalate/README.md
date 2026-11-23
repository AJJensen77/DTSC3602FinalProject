# FilmPalate: website and API for the List taste model (D4 / D5)

**Model:** Ishika Bhattarai's List taste model (`../List taste model/`), the model the
group chose in D3. **Pipeline, API and website:** Yager Range.

- Website: https://filmpalate.vercel.app
- API: https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run · docs at
  https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run/docs

## What the site does

People keep two lists:

| List | Replaces (List taste model) | Weight in the model |
|---|---|---|
| **My Ratings**: 0.5-5 stars once a film is finished | Completed (and Dropped) | >= 3.5 stars +1.0, 2.5-3 +0.25, <= 2 -1.0 |
| **Want to Watch**: films they're interested in | Plan to Watch | +0.5 |

*Currently Watching* and *Dropped* are not used; a low star rating plays Dropped's part.
Rating a Want to Watch film moves it to My Ratings.

Pages: home (popular rows, no account needed), For you, Browse (search, genres, new
releases), My lists, and a movie page with "More like this".

## The fitted pipeline (`watchlist/pipeline.py`)

```
Pipeline([
    ("encode", WatchlistEncoder()),   # custom transformer: lists -> signed taste weights
    ("rank",   WatchlistRanker()),    # List taste model + new films + cold start
])
```

- **`WatchlistEncoder`** turns `{"ratings": {movieId: stars}, "want_to_watch": [...]}` into one
  sparse row of list weights in the model's film order, and rejects unknown films or
  invalid stars (the API returns these as HTTP 422).
- **`FilmContentFeatures`** (custom transformer) turns a film's IMDb description (genres,
  directors and top-3 cast seen in 3+ films, decade) into a feature vector.
- **`WatchlistRanker`** wraps `ListTasteRecommender` with Ishika's tuned settings, and adds:
  - **New films (2017-2026).** The model only knows films with 20+ MovieLens ratings before
    the sealed test period (Nov 2019). 2,684 films released 2017+ with 10,000+ IMDb votes
    are added; each borrows the averaged fingerprint of its 10 closest catalog films by
    description (used for "More like this" and when someone rates one). "More like this"
    for a new film compares descriptions directly.
  - **Popularity by history size.** A per-user popularity weight (see results below).
  - **Cold start.** Nothing positive on the lists -> popular, well-liked films.

### Changes from Ishika's evaluated settings, and why

1. **Popularity weight 2 instead of 6 for heavy raters**, for variety (results below).
2. **Higher popularity weight for users with few ratings** (40 -> 25 -> 15 -> 8 -> 4 -> 2 as
   ratings grow; a Want to Watch counts as half a rating), because with few ratings the
   model otherwise loses to plain popular picks.
3. **The "New releases" row** is ranked by IMDb popularity + rating, not personal taste, and
   says so on the page.

The model code itself (`listmodel/recommender.py`) is unchanged.

## Results (report half, group protocol: D2 split, D2 cohort, hit = validation rating >= 3.5)

| Check | Result | Script -> output |
|---|---|---|
| Popularity weight 2 vs 6 vs baseline, full history (2,953 users) | 0.204 vs 0.208 vs 0.165 precision@10; w2 - baseline +0.039 (95% CI +0.032 to +0.045); w2 - w6 -0.005 (-0.008 to -0.002); 1,400 vs 569 vs 223 distinct films | `evaluate_popularity_weight.py` -> `results/popularity_weight_check.json` |
| Few ratings, weight 2 | Loses to popular picks up to ~20 ratings (1 rating: 0.105 vs 0.165) | `evaluate_min_ratings.py` -> `results/min_ratings_check.json` |
| Few ratings, tuned weight per history size | Beats popular picks from the first rating: +0.003 at 1 rating (CI +0.000 to +0.005), +0.015 at 10, +0.028 at 50 | `evaluate_blend.py` -> `results/blend_check.json` |
| Ranking new films (2015-16 films held out as "new", 2,665 users) | Popularity + IMDb rating 0.188 vs most-voted 0.170 (CI +0.016 to +0.021); personal taste adds nothing (within-user AUC 0.51) | `evaluate_new_films.py` -> `results/new_films_backtest.json` |

Want to Watch cannot be evaluated offline: MovieLens has ratings, not lists.

## Rebuild and run (from the repo root)

Data: as in D2 (`data/processed/` from `prepare_data.py`, MovieLens ml-32m in
`data/raw/ml-32m/`) plus IMDb's `title.basics`, `title.ratings`, `title.principals` and
`title.crew` (`.tsv.gz`, from https://datasets.imdbws.com/) in `data/raw/`.

```bash
uv sync
uv run python "FilmPalate/build_model.py"          # ~10 min -> api/artifacts/ (gitignored)
uv run uvicorn --app-dir "FilmPalate" api.serve:app --port 8000   # http://localhost:8000/docs
uv run modal deploy "FilmPalate/api/modal_serve.py"               # Modal app "filmpalate-api"

cd "FilmPalate/web"
npm ci
npm run dev                                             # http://localhost:3000
```

Website environment (`web/.env.local`, never committed): `NEXT_PUBLIC_MOVIE_API_URL`,
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `TMDB_API_KEY`. The Supabase
table and its row-level security are in `web/supabase/schema.sql`.

`api/artifacts/sample_predictions.json` holds reference outputs for three requests.
`uv run python "FilmPalate/verify_deployment.py"` sends them to the deployed API and
checks the outputs match exactly (plus health, pipeline info, 422 and 404 responses).

## Postman

`postman/FilmPalate.postman_collection.json`: 13 requests, 32 assertions against the
deployed API (health, pipeline info, valid recommendations, empty lists, invalid input
-> 422, unknown movie -> 404). Import it into Postman and click **Run collection**, or run
`npx newman run "FilmPalate/postman/FilmPalate.postman_collection.json"`.

## API

| Endpoint | |
|---|---|
| `GET /health` | liveness + pipeline steps |
| `GET /pipeline` | pipeline steps, settings, training metadata |
| `GET /genres`, `GET /movies/starter`, `/movies/search`, `/movies/browse`, `/movies/lookup` | catalog |
| `GET /movies/{id}`, `GET /movies/{id}/similar` | one film (404 if unknown), more like this |
| `POST /recommend` | lists -> top picks + new releases (the pipeline's `predict`) |
| `POST /recommend/rows` | lists -> rows for the site |

Invalid input returns 422 with a message (stars outside 0.5-5 or not in half steps,
duplicate films, unknown movie ids, unknown genre, out-of-range `k`).

## AI assistance

Built with Claude Code (pipeline, API, website, evaluation scripts, this README), per the
course AI policy. Analytical choices (lists, weights, popularity settings) were made by
Yager Range based on the results above.
