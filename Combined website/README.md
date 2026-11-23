# Combined website: List taste model + Gradient boost model (D5)

The group decided to mix the two D3 candidate models for the website:
**Ishika Bhattarai's List taste model** and **Yager Range's Gradient boost model**. The site
is built from Yager's ReelPicks website and API, reworked into a list site in the style of
MyAnimeList / MyDramaList: people keep a **Finished watching** and a **Plan to watch** list,
score what they've seen, and get recommendations from their lists.

- **Website:** https://reelpicks-combined.vercel.app
- **API (Modal):** https://bhattaraiishika--reelpicks-combined-api-fastapi-app.us-east.modal.run
  ([/docs](https://bhattaraiishika--reelpicks-combined-api-fastapi-app.us-east.modal.run/docs))

This folder doesn't change anything outside it. It reuses `List taste model/listmodel/`,
`Gradient boost model/gbmodel/`, `FilmPalate/D4_submission/watchlist/` and the shared
evaluation code unchanged. FilmPalate
(Yager's D4 app on the List taste model alone) is separate, in `FilmPalate/`.

## How the two models are mixed (`combined/blend.py`)

1. **The List taste model picks 100 candidates** from everything on the person's lists:
   Finished watching (+1.0), Plan to watch (+0.5), and star ratings (3.5+ = +1.0,
   2.5-3 = +0.25, 2 or less = -1.0, i.e. a low score counts like "Dropped").
2. **The Gradient boost model predicts the person's star rating** for each candidate from
   their ratings (IMDb rating, cast, director, similar films, ...).
3. **Final score = list score + 1.0 × predicted stars (standardized per person).**

Each recommendation also says *why*: "Because of X on your list" (the list film closest
in taste). "More like this" on a movie page uses the List taste model's fingerprints.

## Newer films: 2017 to today (`combined/site.py`)

The List taste model only knows films with 20+ MovieLens ratings by Nov 2019. The site
adds every IMDb feature film released **2017-2026 with 1,000+ IMDb votes** (IMDb snapshot
2026-11-23), reusing Yager's `WatchlistRanker` from `FilmPalate/D4_submission/watchlist/`:

- **15,151 newer films** (7,038 MovieLens never had + 8,113 MovieLens has but with too few
  ratings by 2019) each borrow a taste fingerprint from their 10 closest catalog films by
  IMDb description (genres, director, top cast, decade). They can be searched, put on
  lists and scored, and they count toward the person's taste.
- **"New releases for you"** (the 2,695 with 10,000+ votes, Yager's tested cut-off) are
  ranked by IMDb buzz + 0.25 × IMDb rating + **2.0 × the person's taste**. Yager's
  backtest (2015-16 films held out as "new") found taste adds little offline precision
  (weight 0 → 0.197, 0.25 → 0.194, 1.0 → 0.168 tune precision@10, vs 0.170 for "most
  voted"). But at his chosen 0.25 the row was nearly the same for everyone: our 8 test
  users' new-release rows overlapped 75%, and only 10% of picks matched their genre. At
  2.0 the overlap is 11% and the genre match is 41% (5% for an empty list). We chose
  following the person's list. 2.0 itself is beyond his grid and not backtested.
- "More like this" for a newer film compares descriptions (same director, cast, genres).
- Posters: films MovieLens doesn't have get their TMDB id from TMDB's `/find` by IMDb id
  (all 92,103 catalog films have one).
- Main picks are unchanged: they still come from the evaluated combined model on the
  18,258 catalog films.

## Results (report half, 2,953 users, same protocol as both model folders)

D2 split, D2 cohort, hit = a recommended film the user rated 3.5+ in validation; blend
settings picked on the tune half. Reserved test split never read.

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit rate |
|---|---|---|---|---|
| **Combined (popularity weight 6)** | **0.216** | **0.056** | **0.233** | **72.7%** |
| List taste model (weight 6) | 0.208 | 0.052 | 0.225 | 71.0% |
| Gradient boost model | 0.176 | 0.044 | 0.191 | 63.3% |
| Popularity baseline | 0.165 | 0.039 | 0.180 | 59.4% |

Combined vs list taste model: **+0.008** precision@10 (95% CI +0.005 to +0.010); vs
gradient boost **+0.040** (+0.035 to +0.045); vs popularity **+0.051** (+0.046 to +0.057).
(Gradient boost refit here from Yager's code and settings: 0.176 vs his reported 0.175;
the IMDb files were re-downloaded on 2026-11-23, so its features differ slightly.)

## Bias check, and the setting the site uses

D1 flagged popularity bias, and D2 found the top 1% of films hold 39% of all training
ratings, and promised to check catalog coverage and less-popular films alongside
accuracy. `audit_bias.py` does that:

| Model | Precision@10 | Distinct films | Picks from top-1% films | Overlap of 2 users' top 10 | Hits on non-top-1% films |
|---|---|---|---|---|---|
| Popularity baseline | 0.165 | 223 | 63% | 17.4% | 0.074 |
| List taste model, weight 6 | 0.208 | 569 | 51% | 4.4% | 0.106 |
| Combined, weight 6 | 0.216 | 591 | 55% | 4.5% | 0.102 |
| List taste model, weight 2 | 0.204 | 1,400 | 39% | 2.1% | 0.120 |
| **Combined, weight 2 (the site)** | **0.210** | **1,415** | **43%** | **2.3%** | **0.116** |
| Gradient boost (Yager's reported) | 0.175 | 763 | 70% | — | — |

The site uses **combined with list popularity weight 2**: the least popularity-driven
option that is still more accurate than every single model. It gives up 0.006
precision@10 vs combined at weight 6, and in return recommends 2.4× as many different
films, two users' lists overlap half as much, and it finds more hits among less-popular
films. (FilmPalate settled on the same weight 2 independently.) It beats popularity in
every user group, including niche-taste users (0.205 vs 0.143) and harsh raters
(0.184 vs 0.140); see `results/bias_audit_by_group.csv`.

**Do recommendations follow the list?** `check_personas.py` runs the deployed model on 8
made-up users (horror, animation, documentary, romance, western, sci-fi, war, musicals,
12 films each on Finished watching):

- 70-100% of each persona's picks are in their genre, vs 0-30% for an empty list and
  2-15% of the catalog.
- Personas' top-10s overlap 0.4% on average (at most 1 film).
- A Finished watching list with no stars works (60-100%), and so does Plan to watch
  alone (a horror Plan to watch list gives 100% horror).
- Rating the same films 1★ instead of 4.5★ drops that genre to 0-20%.
- Nothing already on a list is recommended back.
- New releases: 41% of each persona's picks are in their genre (5% for an empty list).

Details are in `results/persona_check.json`.

### Short histories (honest caveats)

- **Same information for both** (each model only knows the user's N most recent films):
  the List taste model beats popularity from **1** rating (0.068 vs 0.057), and with 10
  ratings it scores 0.092 vs 0.059. The gradient boost model alone does **not** beat
  popularity with 10 ratings (0.058 vs 0.059). With short histories the blend equals the
  List taste model alone (differences within ±0.002); its gain needs fuller histories.
  The gradient boost step is used from the first rating since it never hurt.
  See `results/combined_few_ratings.csv` (measured at list weight 6).
- **Both skip every film the user has seen** (FilmPalate's protocol): the List taste
  model at weight 2 loses to popularity until ~20 ratings (FilmPalate's
  `extras/results/min_ratings_check.json`), which is why FilmPalate raises the
  popularity weight for new users. This site keeps weight 2 so a new user's picks
  follow their list rather than the global top 10. Re-running both protocols at
  weight 2 for the blend (`audit_bias.py`'s last step) was not finished: it was stopped
  by low memory on the build machine.
- MovieLens has ratings, not lists, so Plan to watch can't be evaluated offline; its
  +0.5 weight is the List taste model's D3 default.

## Website (`web/`)

Next.js + shadcn/ui on Vercel, accounts and lists in Supabase (`web/supabase/schema.sql`:
`ratings` and `list_items` tables, row-level security so each person only sees their own).

| Page | What it does |
|---|---|
| `/` | Logged out: sign up / browse. Logged in: "You might be interested in", "New releases for you", "Because you finished X", Plan to watch shelf |
| `/browse` | Search any of 92,103 films (through 2026); add to a list or score it from the card |
| `/movie/[id]` | Your entry (status + score), info, trailer, **More like this** |
| `/list` | My list: All / Finished watching / Plan to watch tabs, stats, scores, status, sort |
| `/for-you` | Recommendations with genre filters and predicted stars, plus new releases |
| `/login`, `/signup` | Supabase email + password accounts |

Hovering a poster plays a muted ~30 s trailer clip (YouTube via TMDB). Posters and
trailers need `TMDB_API_KEY` in the Vercel project. Without it, cards show the title
and the movie page links to a YouTube trailer search.

## Reproduce (repo root)

Data: D2's `data/processed/` and `data/raw/ml-32m/`, plus IMDb `title.basics`,
`title.ratings`, `title.principals`, `title.crew` (`.tsv.gz`) in `data/raw/`, and the List
taste model artifact (`uv run python "List taste model/evaluate_list_model.py"`).

```bash
uv run python "Gradient boost model/prepare_imdb.py"     # IMDb features (~5 min)
uv run python "Combined website/evaluate_combined.py"    # fit GB as evaluated, tune + test blend (~1 h)
uv run python "Combined website/audit_bias.py"           # bias audit -> results/site_settings.json (~8 GB RAM)
uv run python "Combined website/build_site.py"           # site model + newer films -> api/artifacts/ (~25 min; TMDB_API_KEY for posters)
uv run python "Combined website/check_personas.py"       # do picks follow the list?
PYTHONUTF8=1 uv run modal deploy "Combined website/api/modal_serve.py"
cd "Combined website/web" && npm ci && npx vercel deploy --prod
```

Vercel environment: `NEXT_PUBLIC_MOVIE_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`,
`NEXT_PUBLIC_SUPABASE_ANON_KEY` (publishable key), optional `TMDB_API_KEY`. Local
development: put the same in `web/.env.local` (gitignored), then `npm run dev`, and run the
API with `uv run uvicorn --app-dir "Combined website" api.serve:app --reload`.

## Files

| Path | What it is |
|---|---|
| `combined/blend.py` | `CombinedRecommender`: the mix, used by the evaluation and the API |
| `evaluate_combined.py` | Fits the gradient boost model as Yager evaluated it; tunes and tests the blend |
| `audit_bias.py` | Popularity / coverage / personalization / user-group audit; picks the site setting |
| `combined/site.py` | `SiteRecommender`: the combined model + newer films (via FilmPalate's `WatchlistRanker`) |
| `build_site.py` | Refreshes both models with ratings through 2019-11, adds 2017-2026 films, looks up posters |
| `check_personas.py` | Made-up users: do recommendations follow their lists? |
| `api/serve.py`, `api/modal_serve.py` | FastAPI service and its Modal deployment |
| `web/` | The website |
| `results/` | Outputs of the scripts above |

Built with Claude Code, per the course AI policy.
