# FilmPalate extras (not part of the D4 submission)

Supporting material for `../D4_submission/`. Commands run from the repo root.

| Item | What it is |
|---|---|
| `evaluate_popularity_weight.py`, `evaluate_min_ratings.py`, `evaluate_blend.py`, `evaluate_new_films.py` | Backtests behind the choices and numbers in the D4 README |
| `results/*.json` | Their outputs |
| `sample_predictions.json` | The fitted pipeline's outputs for three requests (written by `build_model.py`) |
| `verify_deployment.py` | Sends those requests to the deployed API and checks the answers match exactly, plus health, pipeline info, 422 and 404 responses |
| `web/` | The FilmPalate website (D5): https://filmpalate.vercel.app |
| `postman/FilmPalate.postman_collection.json` | Postman tests against the deployed API (D5) |

```bash
uv run python "FilmPalate/extras/evaluate_blend.py"          # likewise for the other backtests
uv run python "FilmPalate/extras/verify_deployment.py"
npx newman run "FilmPalate/extras/postman/FilmPalate.postman_collection.json"
```

## Website (`web/`)

```bash
cd FilmPalate/extras/web
npm ci
npm run dev        # http://localhost:3000
```

Environment (`web/.env.local`, never committed): `NEXT_PUBLIC_MOVIE_API_URL`,
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `TMDB_API_KEY`. The Supabase
table and its row-level security are in `web/supabase/schema.sql`. Deployed to Vercel
(`web/vercel.json` declares the Next.js framework).

## Postman

13 requests, 32 assertions against the deployed API: health, pipeline info, valid
recommendations, empty lists, invalid input -> 422, unknown movie -> 404. Import the
collection into Postman and click **Run collection**.
