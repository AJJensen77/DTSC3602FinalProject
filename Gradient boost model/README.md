# Gradient boost model (Yager Range)

A candidate model for the group comparison, plus the ReelPicks website that serves it.
Everything for it lives in this folder; deleting the folder removes it. The only change
outside it is five added dependencies in the root `pyproject.toml`/`uv.lock`
(scikit-learn, scipy, joblib, fastapi, uvicorn).

**Group summary:** [`MODEL_SUMMARY.md`](MODEL_SUMMARY.md) ·
**Live site:** https://reelpicks.vercel.app ·
**API docs:** https://yagerr05--movie-recommender-api-fastapi-app.us-east.modal.run/docs

## What's here

| Path | What it is |
|---|---|
| `gbmodel/hybrid.py` | `TasteProfileTransformer` (features) and `HybridRecommender` (gradient boosting) |
| `gbmodel/recommenders.py` | Shared recommender contract; item-kNN similarity used as a feature |
| `baseline/` | **Popularity baseline** all models are compared against: code, its own evaluation, results ([README](baseline/README.md)) |
| `gbmodel/imdb_features.py` | IMDb rating, top-billed cast, director loaders |
| `prepare_imdb.py` | Builds `data/processed/movie_features.parquet` |
| `evaluate_hybrid.py` | Training, tuning, evaluation vs baseline, ablations, error analysis |
| `results/`, `figures/` | Outputs of `evaluate_hybrid.py` (fig6–fig8) |
| `build_site_model.py` | Builds the artifacts the website API serves |
| `api/` | FastAPI service (`serve.py`) and its Modal deployment (`modal_serve.py`) |
| `web/` | Next.js + shadcn/ui website on Vercel, Supabase accounts and ratings |

## Reproducing (run from the repo root)

Prerequisite: the D2 pipeline (`prepare_data.py`) has been run. Download three more IMDb files:

```bash
for f in title.ratings title.principals title.crew; do
  curl -o data/raw/$f.tsv.gz https://datasets.imdbws.com/$f.tsv.gz
done
uv sync
uv run python "Gradient boost model/prepare_imdb.py"      # IMDb features
uv run python "Gradient boost model/evaluate_hybrid.py"   # ~45 min, ~8 GB RAM (--quick: ~3 min sample)
uv run python "Gradient boost model/build_site_model.py"  # website model -> api/artifacts/
```

Evaluation reads training and validation ratings only; the reserved test split is never opened.

## Website

| Piece | Service | Deploy |
|---|---|---|
| Model API | Modal | `uv run modal deploy "Gradient boost model/api/modal_serve.py"` |
| Accounts + ratings | Supabase (separate project from D2) | run `web/supabase/schema.sql` once in the SQL editor |
| Website | Vercel | `cd "Gradient boost model/web" && npx vercel deploy --prod` |

The site depends only on the API contract `POST /recommend` (ratings in, up to 10 movies out),
so another model can be served behind the same API without changing the website.

Settings live in `web/.env.local` (gitignored) and in the Vercel project:
`NEXT_PUBLIC_MOVIE_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
(publishable key, safe in the browser; ratings are protected by row-level security), and
`TMDB_API_KEY` (server-only, for posters).

Local development:

```bash
uv run uvicorn --app-dir "Gradient boost model" api.serve:app --reload   # API on :8000
cd "Gradient boost model/web" && npm install && npm run dev               # site on :3000
```

Data credits: MovieLens 32M (GroupLens), IMDb non-commercial datasets, posters from TMDB
(this product uses the TMDB API but is not endorsed or certified by TMDB).
