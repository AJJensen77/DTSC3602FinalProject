"""FilmPalate API: movie recommendations from the List taste model pipeline.

A website user has two lists:
* **My Ratings**: films they finished, 0.5-5 stars.
* **Want to Watch**: films they are interested in (counts as mild positive taste).

``POST /recommend`` runs the fitted scikit-learn pipeline (WatchlistEncoder ->
WatchlistRanker) and returns the top films. ``POST /recommend/rows`` powers the
website's home page: top picks, new releases, "because you liked / want to watch"
rows and genre rows. Both work with empty lists (popular, well-liked films).

Local dev (from the repo root):
    uv run uvicorn --app-dir "FilmPalate" api.serve:app --reload
    -> http://localhost:8000/docs
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# The pickled pipeline refers to classes in these packages.
APP_DIR = Path(__file__).resolve().parents[1]
for p in (APP_DIR, APP_DIR.parent / "List taste model", APP_DIR.parent / "Gradient boost model"):
    if p.exists():
        sys.path.insert(0, str(p))

ARTIFACTS = Path(__file__).parent / "artifacts"
ROW_SIZE, NEW_PER_GENRE, MIN_ROW = 10, 3, 4
LIKED_ROWS = 2   # "Because you liked ..." rows
MAX_LIST = 2000
HIDDEN_GENRES = {"IMAX", "(no genres listed)"}

# MovieLens files "The Dark Knight" as "Dark Knight, The"; people search the other way.
_ARTICLE = re.compile(r"^(.+?), (The|A|An)( \(.*\))?$")


def display_title(title: str) -> str:
    return _ARTICLE.sub(r"\2 \1\3", str(title))


def load():
    try:
        bundle = joblib.load(ARTIFACTS / "watchlist_pipeline.joblib")
        catalog = pd.read_parquet(ARTIFACTS / "catalog.parquet")
    except Exception as exc:  # noqa: BLE001 - any load failure should surface as 503
        return None, None, str(exc)
    catalog["title"] = catalog["title"].map(display_title)
    catalog["search_key"] = catalog["title"].str.lower()
    catalog["genre_list"] = catalog["genres"].fillna("").str.split("|")
    # Browse order: the starter grid first, then the best-known films.
    catalog = catalog.sort_values(["starter_rank", "imdb_votes", "n_ratings"],
                                  ascending=[True, False, False], na_position="last")
    return bundle, catalog.set_index("movieId", drop=False), None


# Loaded once per server process, never refit per request.
_bundle, _catalog, _load_error = load()

app = FastAPI(
    title="FilmPalate API",
    version="1.0.0",
    description=("Recommendations from a person's **My Ratings** (0.5-5 stars) and "
                 "**Want to Watch** list, using the group's List taste model in a "
                 "fitted scikit-learn pipeline. Works with empty lists too."),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def require():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifact not loaded: {_load_error}")
    return _bundle["pipeline"], _catalog


# -- schemas ------------------------------------------------------------------


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    imdb_id: str | None
    imdb_rating: float | None
    is_new: bool = Field(..., description="Released 2017+ and added from IMDb (not in MovieLens)")


class RatingIn(BaseModel):
    movie_id: int = Field(..., description="From /movies/search, /movies/browse or /movies/starter")
    stars: float = Field(..., ge=0.5, le=5.0, description="0.5 to 5.0 in half-star steps")

    @field_validator("stars")
    @classmethod
    def half_steps(cls, v: float) -> float:
        if round(v * 2) != v * 2:
            raise ValueError("stars must be in half-star steps (0.5, 1.0, ..., 5.0)")
        return v


class ListsIn(BaseModel):
    ratings: list[RatingIn] = Field(default_factory=list, max_length=MAX_LIST,
                                    description="My Ratings")
    want_to_watch: list[int] = Field(default_factory=list, max_length=MAX_LIST,
                                     description="Want to Watch movie ids, oldest first")

    @model_validator(mode="after")
    def no_duplicates(self):
        ids = [r.movie_id for r in self.ratings]
        if len(set(ids)) != len(ids):
            raise ValueError("each movie may be rated only once")
        if len(set(self.want_to_watch)) != len(self.want_to_watch):
            raise ValueError("each movie may be on Want to Watch only once")
        return self

    def as_pipeline_input(self) -> dict:
        return {"ratings": {r.movie_id: r.stars for r in self.ratings},
                "want_to_watch": list(self.want_to_watch)}


class RecommendRequest(ListsIn):
    k: int = Field(10, ge=1, le=50)


class Recommendation(Movie):
    rank: int


class RecommendResponse(BaseModel):
    personalized: bool
    n_ratings: int
    n_want_to_watch: int
    recommendations: list[Recommendation]
    new_releases: list[Recommendation]


class Row(BaseModel):
    key: str = Field(..., description='"top", "new", "because-<id>", "want-<id>" or a genre')
    title: str
    movies: list[Movie]
    seed: Movie | None = Field(None, description='The film a "Because you..." row is built from')
    note: str | None = None


class RowsRequest(ListsIn):
    rotate: int = Field(0, ge=0, le=10_000, description=(
        'Which of your films the "Because you..." rows start from; the site '
        "increases it on every Refresh so the rows cycle through your favourites."))


class RowsResponse(BaseModel):
    personalized: bool
    rows: list[Row]


class BrowseResponse(BaseModel):
    movies: list[Movie]
    has_more: bool


class Similar(Movie):
    similarity: float


def to_movie(row) -> dict:
    def opt(v, cast):
        return None if pd.isna(v) else cast(v)

    return {
        "movie_id": int(row["movieId"]),
        "title": row["title"],
        "year": opt(row["year"], int),
        "genres": [g for g in row["genre_list"] if g and g not in HIDDEN_GENRES],
        "tmdb_id": opt(row["tmdbId"], int),
        "imdb_id": opt(row["imdb_id"], str),
        "imdb_rating": opt(row["imdb_rating"], lambda x: round(float(x), 1)),
        "is_new": bool(row["is_new"]),
    }


def movies(cat, ids) -> list[dict]:
    return [to_movie(cat.loc[m]) for m in ids if m in cat.index]


def encode(pipe, lists: ListsIn):
    """Run the pipeline's encoder; its ValueErrors (unknown ids, bad stars) become 422."""
    try:
        return pipe[:-1].transform([lists.as_pipeline_input()])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


# -- routes -------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "service": "FilmPalate API",
        "docs": "/docs",
        "model": "ok" if _bundle is not None else "artifact not loaded",
        "endpoints": {
            "GET /health": "liveness check",
            "GET /pipeline": "pipeline steps, settings and training metadata",
            "GET /genres": "genres usable as filters",
            "GET /movies/starter": "well-known films to start a new user's lists",
            "GET /movies/search": "title search",
            "GET /movies/browse": "films a page at a time, optionally one genre or only new",
            "GET /movies/lookup": "details for a list of movie ids",
            "GET /movies/{movie_id}": "one film",
            "GET /movies/{movie_id}/similar": "more like this",
            "POST /recommend": "lists -> top recommendations and new releases",
            "POST /recommend/rows": "lists -> rows for the home page",
        },
    }


@app.get("/health")
def health():
    pipe, cat = require()
    return {"status": "ok", "films": int(len(cat)), "pipeline_steps": [n for n, _ in pipe.steps]}


@app.get("/pipeline")
def pipeline_info():
    pipe, _ = require()
    enc, rank = pipe.named_steps["encode"], pipe.named_steps["rank"]
    return {
        **_bundle["metadata"],
        "encoder_params": {k: v for k, v in enc.get_params().items() if k != "film_ids"},
        "ranker_params": rank.get_params(),
    }


@app.get("/genres", response_model=list[str])
def genres():
    _, cat = require()
    return GENRE_ORDER


@app.get("/movies/starter", response_model=list[Movie])
def starter(n: int = Query(48, ge=1, le=72)):
    _, cat = require()
    rows = cat.dropna(subset=["starter_rank"]).sort_values("starter_rank").head(n)
    return [to_movie(r) for _, r in rows.iterrows()]


@app.get("/movies/search", response_model=list[Movie])
def search(q: str = Query(..., min_length=2, max_length=100), limit: int = Query(20, ge=1, le=50)):
    _, cat = require()
    hits = cat[cat["search_key"].str.contains(q.strip().lower(), regex=False)]
    hits = hits.sort_values(["imdb_votes", "n_ratings"], ascending=False).head(limit)
    return [to_movie(r) for _, r in hits.iterrows()]


@app.get("/movies/browse", response_model=BrowseResponse)
def browse(offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=60),
           genre: str | None = Query(None, description="Only this genre"),
           new_only: bool = Query(False, description="Only 2017+ films")):
    _, cat = require()
    if genre is not None and genre not in GENRE_ORDER:
        raise HTTPException(status_code=422, detail=f"unknown genre: {genre}")
    rows = cat
    if genre is not None:
        rows = rows[rows["genre_list"].map(lambda gs: genre in gs)]
    if new_only:
        rows = rows[rows["is_new"]].sort_values("imdb_votes", ascending=False)
    page = rows.iloc[offset: offset + limit]
    return BrowseResponse(movies=[to_movie(r) for _, r in page.iterrows()],
                          has_more=offset + limit < len(rows))


@app.get("/movies/lookup", response_model=list[Movie])
def lookup(ids: str = Query(..., description="Comma-separated movie ids, max 500")):
    _, cat = require()
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="ids must be comma-separated integers") from None
    if len(wanted) > 500:
        raise HTTPException(status_code=422, detail="at most 500 ids per request")
    return movies(cat, wanted)


@app.get("/movies/{movie_id}", response_model=Movie)
def movie(movie_id: int):
    _, cat = require()
    if movie_id not in cat.index:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found")
    return to_movie(cat.loc[movie_id])


@app.get("/movies/{movie_id}/similar", response_model=list[Similar])
def similar(movie_id: int, k: int = Query(10, ge=1, le=30)):
    pipe, cat = require()
    try:
        sims = pipe.named_steps["rank"].similar(movie_id, k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found") from None
    return [Similar(**to_movie(cat.loc[m]), similarity=round(s, 3)) for m, s in sims]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    pipe, cat = require()
    H = encode(pipe, req)
    rank = pipe.named_steps["rank"]
    top = rank.predict(H, k=req.k)[0]
    new = rank.predict_new(H, k=req.k)[0]

    def recs(ids):
        return [Recommendation(**m, rank=i + 1) for i, m in enumerate(movies(cat, ids))]

    return RecommendResponse(personalized=bool(rank.has_taste(H)[0]),
                             n_ratings=len(req.ratings), n_want_to_watch=len(req.want_to_watch),
                             recommendations=recs(top), new_releases=recs(new))


def _genre_order(cat) -> list[str]:
    g = cat["genre_list"].explode()
    watched = cat["n_ratings"].reindex(g.index).groupby(g.to_numpy()).sum()
    return [x for x in watched.sort_values(ascending=False).index if x and x not in HIDDEN_GENRES]


GENRE_ORDER = _genre_order(_catalog) if _catalog is not None else []


@app.post("/recommend/rows", response_model=RowsResponse)
def recommend_rows(req: RowsRequest):
    pipe, cat = require()
    H = encode(pipe, req)
    rank = pipe.named_steps["rank"]
    personal = bool(rank.has_taste(H)[0])
    base_s, new_s = (s[0] for s in rank.score(H))
    nb = rank.n_base_
    base_ids, new_ids = rank.film_ids_[:nb], rank.film_ids_[nb:]
    seen = {r.movie_id for r in req.ratings} | set(req.want_to_watch)

    def top_of(scores, ids, mask=None, n=ROW_SIZE):
        s = scores if mask is None else np.where(mask, scores, -np.inf)
        order = np.argsort(-s)[:n]
        return [int(ids[j]) for j in order if np.isfinite(s[j])]

    rows = [Row(key="top", title="Top picks for you" if personal else "Popular and well liked",
                movies=movies(cat, top_of(base_s, base_ids))),
            # Not personal: the backtest found taste does not help rank new films
            # (evaluate_new_films.py), so this row is popular + well reviewed.
            Row(key="new", title="New releases",
                movies=movies(cat, top_of(new_s, new_ids)))]

    # "More like this" rows: up to LIKED_ROWS from 4+ star ratings and one from Want
    # to Watch. Newest first, but films the model has viewing data for come before
    # 2017+ films, whose "similar" list is matched only by genre, director and cast.
    # ``rotate`` (bumped on each Refresh) cycles through the rest.
    def seeds(ids, n):
        ids = list(reversed(ids))                      # newest first
        ordered = [m for m in ids if not cat.at[m, "is_new"]] + [m for m in ids if cat.at[m, "is_new"]]
        if not ordered:
            return []
        start = (req.rotate * n) % len(ordered)
        return [ordered[(start + i) % len(ordered)] for i in range(min(n, len(ordered)))]

    liked = [r.movie_id for r in req.ratings if r.stars >= 4.0]
    for key, label, chosen in (("because", "Because you liked", seeds(liked, LIKED_ROWS)),
                               ("want", "Because you want to watch", seeds(req.want_to_watch, 1))):
        for m in chosen:
            sims = [x for x, _ in rank.similar(m, ROW_SIZE + len(seen) + 1) if x not in seen]
            seed = to_movie(cat.loc[m])
            rows.append(Row(key=f"{key}-{m}", title=f"{label} {seed['title']}",
                            movies=movies(cat, sims[:ROW_SIZE]), seed=seed,
                            note=("Newer film: matched by genre, director and cast"
                                  if seed["is_new"] else None)))

    base_genres = cat.loc[base_ids, "genre_list"].to_numpy()
    new_genres = cat.loc[new_ids, "genre_list"].to_numpy()
    genre_rows = []
    for g in GENRE_ORDER:
        b = top_of(base_s, base_ids, np.array([g in gs for gs in base_genres]))
        n = top_of(new_s, new_ids, np.array([g in gs for gs in new_genres]), NEW_PER_GENRE)
        for i, m in enumerate(n):          # new films at positions 2, 5 and 8
            b.insert(min(2 + 3 * i, len(b)), m)
        if len(b) >= MIN_ROW:
            # Personal mode: genres the user is most into first.
            strength = float(np.mean([base_s[rank.film_index_.get_loc(m)] for m in b
                                      if rank.film_index_.get_loc(m) < nb][:5] or [0]))
            genre_rows.append((strength, Row(key=g, title=g, movies=movies(cat, b[:ROW_SIZE]))))
    if personal:
        genre_rows.sort(key=lambda t: -t[0])
    rows += [r for _, r in genre_rows]
    return RowsResponse(personalized=personal, rows=[r for r in rows if r.movies])
