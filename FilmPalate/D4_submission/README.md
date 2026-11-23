# FilmPalate API: fitted pipeline served on Modal (D4)

**Model:** Ishika Bhattarai's List taste model, the model the group chose in D3.
**Pipeline and API:** Yager Range.

- **Modal API:** https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run
- **API docs (/docs):** https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run/docs

## What's in this folder

| File | What it is |
|---|---|
| `watchlist/pipeline.py` | The fitted scikit-learn `Pipeline` and its custom transformers (below) |
| `watchlist/imdb.py` | Loads IMDb data for films released 2017-2026 |
| `build_model.py` | Rebuilds the saved artifact from the data |
| `api/serve.py` | FastAPI app (endpoints and error handling below) |
| `api/modal_serve.py` | Modal deployment of `api/serve.py` |

The pipeline also uses code from the group's other folders, unchanged:
`List taste model/listmodel/recommender.py` (Ishika's model) and
`Gradient boost model/gbmodel/recommenders.py` (the shared recommender base class). The
Python environment is pinned by `pyproject.toml` and `uv.lock` at the repo root.

Supporting material that isn't part of this submission (backtest scripts and results, a
local-vs-deployed check, the website and Postman tests) is in `../extras/`.

## Input: a person's two lists

| List | Replaces (List taste model) | Weight in the model |
|---|---|---|
| **My Ratings**: 0.5-5 stars once a film is finished | Completed (and Dropped) | >= 3.5 stars +1.0, 2.5-3 +0.25, <= 2 -1.0 |
| **Want to Watch**: films they're interested in | Plan to Watch | +0.5 |

*Currently Watching* and *Dropped* are not used; a low star rating plays Dropped's part.

## The fitted pipeline (`watchlist/pipeline.py`)

```
Pipeline([
    ("encode", WatchlistEncoder()),   # custom transformer: lists -> signed taste weights
    ("rank",   WatchlistRanker()),    # List taste model + new films + cold start
])
```

- **`WatchlistEncoder`** (custom transformer) turns
  `{"ratings": {movieId: stars}, "want_to_watch": [...]}` into one sparse row of list
  weights in the model's film order, and rejects unknown films or invalid stars (the API
  returns these as HTTP 422).
- **`FilmContentFeatures`** (custom transformer) turns a film's IMDb description (genres,
  directors and top-3 cast seen in 3+ films, decade) into a feature vector.
- **`WatchlistRanker`** wraps `ListTasteRecommender` with Ishika's tuned settings, and adds:
  - **New films (2017-2026).** The model only knows films with 20+ MovieLens ratings before
    the sealed test period (Nov 2019). 2,684 films released 2017+ with 10,000+ IMDb votes
    are added; each borrows the averaged fingerprint of its 10 closest catalog films by
    description. "More like this" for a new film compares descriptions directly.
  - **Popularity by history size.** A per-user popularity weight (see results).
  - **Cold start.** Nothing positive on the lists -> popular, well-liked films.

### Changes from Ishika's evaluated settings, and why

1. **Popularity weight 2 instead of 6 for heavy raters**, for variety.
2. **Higher popularity weight for users with few ratings** (40 -> 25 -> 15 -> 8 -> 4 -> 2 as
   ratings grow; a Want to Watch counts as half a rating), because with few ratings the
   model otherwise loses to plain popular picks.
3. **New releases** are ranked by IMDb popularity + rating, not personal taste.

The model code itself (`listmodel/recommender.py`) is unchanged.

### Results behind these choices

Report half, group protocol (D2 split, D2 cohort, hit = validation rating >= 3.5). Scripts
and outputs are in `../extras/`.

| Check | Result |
|---|---|
| Popularity weight 2 vs 6 vs baseline, full history (2,953 users) | 0.204 vs 0.208 vs 0.165 precision@10; w2 - baseline +0.039 (95% CI +0.032 to +0.045); w2 - w6 -0.005 (-0.008 to -0.002); 1,400 vs 569 vs 223 distinct films |
| Few ratings, weight 2 | Loses to popular picks up to ~20 ratings (1 rating: 0.105 vs 0.165) |
| Few ratings, tuned weight per history size | Beats popular picks from the first rating: +0.003 at 1 rating (CI +0.000 to +0.005), +0.015 at 10, +0.028 at 50 |
| Ranking new films (2015-16 films held out as "new", 2,665 users) | Popularity + IMDb rating 0.188 vs most-voted 0.170 (CI +0.016 to +0.021); personal taste adds nothing (within-user AUC 0.51) |

Want to Watch cannot be evaluated offline: MovieLens has ratings, not lists.

## Rebuild the artifact and run the API (from the repo root)

The fitted artifact (`api/artifacts/watchlist_pipeline.joblib`, ~57 MB, plus
`catalog.parquet`) is not committed; rebuild it with `build_model.py`.

**Data:** as in D2 (`data/processed/` from `prepare_data.py`, MovieLens ml-32m in
`data/raw/ml-32m/`) plus IMDb's `title.basics`, `title.ratings`, `title.principals` and
`title.crew` (`.tsv.gz`, from https://datasets.imdbws.com/) in `data/raw/`.

```bash
uv sync
uv run python "FilmPalate/D4_submission/build_model.py"     # ~10 min -> api/artifacts/
uv run uvicorn --app-dir "FilmPalate/D4_submission" api.serve:app --port 8000
#   -> http://localhost:8000/docs
uv run modal deploy "FilmPalate/D4_submission/api/modal_serve.py"   # Modal app "filmpalate-api"
```

On Windows, set `PYTHONUTF8=1` before `modal deploy` (its progress output uses symbols the
default console encoding can't print).

## API

| Endpoint | |
|---|---|
| `GET /health` | liveness + pipeline steps |
| `GET /pipeline` | pipeline steps, settings, training metadata |
| `POST /recommend` | lists -> top picks + new releases (the pipeline's `predict`) |
| `POST /recommend/rows` | lists -> rows of recommendations for the website |
| `GET /movies/{id}`, `GET /movies/{id}/similar` | one film (404 if unknown), more like this |
| `GET /genres`, `/movies/starter`, `/movies/search`, `/movies/browse`, `/movies/lookup` | catalog |

Example request:

```json
POST /recommend
{"ratings": [{"movie_id": 2571, "stars": 5}, {"movie_id": 1721, "stars": 1.5}],
 "want_to_watch": [79132], "k": 10}
```

**Errors:** invalid input returns **422** with a message (stars outside 0.5-5 or not in half
steps, the same film rated twice, unknown movie ids, unknown genre, `k` out of range); an
unknown movie returns **404**; if the artifact failed to load, every model endpoint
returns **503**.

## AI assistance

Built with Claude Code (pipeline, API, evaluation scripts, this README), per the course AI
policy. Analytical choices (lists, weights, popularity settings) were made by Yager Range
based on the results above.
