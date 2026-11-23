"""Combined movie recommender API (list taste model + gradient boost model).

Built on Yager's ReelPicks API: the movie endpoints are the same, and /recommend now
takes a person's watch lists as well as their ratings. Adds "more like this".

Local dev (from the repo root):
    uv run uvicorn --app-dir "Combined website" api.serve:app --reload
    -> http://localhost:8000/docs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# The pickled models refer to classes in the `combined`, `listmodel` and `gbmodel` packages.
HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1] / "List taste model", HERE.parents[1] / "Gradient boost model"):
    sys.path.insert(0, str(p))

ARTIFACTS = HERE / "artifacts"
HIDDEN_GENRES = {"IMAX"}  # a screen format, not a genre people filter by
MAX_ITEMS = 5000

# MovieLens files "The Dark Knight" as "Dark Knight, The"; people search the other way.
_ARTICLE = re.compile(r"^(.+?), (The|A|An)( \(.*\))?$")


def display_title(title: str) -> str:
    return _ARTICLE.sub(r"\2 \1\3", title)


def load():
    try:
        bundle = joblib.load(ARTIFACTS / "site_bundle.joblib")
        catalog = pd.read_parquet(ARTIFACTS / "catalog.parquet")
    except Exception as exc:  # noqa: BLE001 - any load failure should surface as 503
        return None, None, None, str(exc)
    catalog["title"] = catalog["title"].map(display_title)
    catalog["search_key"] = catalog["title"].str.lower()
    catalog = catalog.set_index("movieId", drop=False)
    # Genre filter masks over the list model's films (it generates the candidates).
    lm = bundle["combined"].list_model
    g = catalog["genres"].reindex(lm.item_ids_).fillna("").astype(str)
    masks = {name: g.str.contains(name, regex=False).to_numpy()
             for name in bundle["combined"].gb_model.genres_}
    return bundle, catalog, masks, None


# Loaded once per server process - never reloaded or refit per request.
_bundle, _catalog, _genre_masks, _load_error = load()

app = FastAPI(
    title="ReelPicks Combined Recommender API",
    version="2.0.0",
    description=("Watch lists and ratings in, recommendations out. Candidates come from the "
                 "list taste model; the gradient boost model predicts stars and re-ranks them."),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def require():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifact not loaded: {_load_error}")
    return _bundle["combined"], _catalog


# -- schemas ------------------------------------------------------------------


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    imdb_rating: float | None


class MovieDetail(Movie):
    imdb_votes: int | None
    movielens_ratings: int
    has_similar: bool


class RatingIn(BaseModel):
    movie_id: int = Field(..., description="MovieLens movieId (from /movies/search or /movies/starter)")
    rating: float = Field(..., ge=0.5, le=5.0, description="0.5 to 5.0 in half-star steps")

    @field_validator("rating")
    @classmethod
    def half_steps(cls, v: float) -> float:
        if round(v * 2) != v * 2:
            raise ValueError("rating must be in half-star steps (0.5, 1.0, ..., 5.0)")
        return v


class RecommendRequest(BaseModel):
    ratings: list[RatingIn] = Field(default_factory=list, max_length=MAX_ITEMS)
    finished: list[int] = Field(default_factory=list, max_length=MAX_ITEMS,
                                description="Finished watching (rated films count as finished too)")
    plan_to_watch: list[int] = Field(default_factory=list, max_length=MAX_ITEMS)
    genres: list[str] = Field(default_factory=list, description="Optional: only these genres (any match)")
    k: int = Field(10, ge=1, le=30)


class BecauseOf(BaseModel):
    movie_id: int
    title: str


class Recommendation(Movie):
    predicted_rating: float | None
    because: BecauseOf | None


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    n_ratings_used: int
    n_list_items_used: int
    used_gradient_boost: bool
    min_ratings_for_gradient_boost: int
    genres: list[str]
    note: str | None = None


def to_movie(row) -> dict:
    def opt(v, cast):
        return None if pd.isna(v) else cast(v)

    return {
        "movie_id": int(row["movieId"]),
        "title": row["title"],
        "year": opt(row["year"], int),
        "genres": [g for g in str(row["genres"]).split("|") if g and g != "(no genres listed)"],
        "tmdb_id": opt(row["tmdbId"], int),
        "imdb_rating": opt(row["imdb_rating"], lambda x: round(float(x), 1)),
    }


def check_ids(ids, cat, status=422):
    unknown = [i for i in ids if i not in cat.index]
    if unknown:
        raise HTTPException(status_code=status, detail=f"unknown movie ids: {unknown[:10]}")


# -- routes -------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "service": "ReelPicks Combined Recommender API",
        "docs": "/docs",
        "model": "ok" if _bundle is not None else "artifact not loaded",
        "endpoints": {
            "GET /health": "liveness check",
            "GET /info": "model metadata",
            "GET /genres": "genres usable as filters",
            "GET /movies/starter": "well-known films for a new user",
            "GET /movies/search": "title search",
            "GET /movies/lookup": "details for a list of movie ids",
            "GET /movies/{movie_id}": "one film",
            "GET /movies/{movie_id}/similar": "more like this",
            "POST /recommend": "lists + ratings -> recommendations",
        },
    }


@app.get("/health")
def health():
    require()
    return {"status": "ok"}


@app.get("/info")
def info():
    require()
    return _bundle["metadata"]


@app.get("/genres", response_model=list[str])
def genres():
    comb, _ = require()
    return sorted(g for g in comb.gb_model.genres_ if g not in HIDDEN_GENRES)


@app.get("/movies/starter", response_model=list[Movie])
def starter(n: int = Query(48, ge=1, le=72)):
    _, cat = require()
    rows = cat.dropna(subset=["starter_rank"]).sort_values("starter_rank").head(n)
    return [to_movie(r) for _, r in rows.iterrows()]


@app.get("/movies/search", response_model=list[Movie])
def search(q: str = Query(..., min_length=2, max_length=100), limit: int = Query(20, ge=1, le=50)):
    _, cat = require()
    hits = cat[cat["search_key"].str.contains(q.strip().lower(), regex=False)]
    # Best-known first: most people searching mean the famous film.
    hits = hits.sort_values(["n_ratings", "imdb_votes"], ascending=False).head(limit)
    return [to_movie(r) for _, r in hits.iterrows()]


@app.get("/movies/lookup", response_model=list[Movie])
def lookup(ids: str = Query(..., description="Comma-separated movie ids, max 500")):
    _, cat = require()
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="ids must be comma-separated integers")
    if len(wanted) > 500:
        raise HTTPException(status_code=422, detail="at most 500 ids per request")
    found = cat.reindex([i for i in wanted if i in cat.index])
    return [to_movie(r) for _, r in found.iterrows()]


@app.get("/movies/{movie_id}", response_model=MovieDetail)
def movie(movie_id: int):
    _, cat = require()
    check_ids([movie_id], cat, status=404)
    row = cat.loc[movie_id]
    return MovieDetail(**to_movie(row),
                       imdb_votes=None if pd.isna(row["imdb_votes"]) else int(row["imdb_votes"]),
                       movielens_ratings=int(row["n_ratings"]),
                       has_similar=bool(row["in_list_model"]))


@app.get("/movies/{movie_id}/similar", response_model=list[Movie])
def similar(movie_id: int, k: int = Query(12, ge=1, le=30)):
    comb, cat = require()
    check_ids([movie_id], cat, status=404)
    return [to_movie(cat.loc[m]) for m in comb.similar(movie_id, k) if m in cat.index]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    comb, cat = require()
    ratings = {r.movie_id: r.rating for r in req.ratings}
    if len(ratings) != len(req.ratings):
        raise HTTPException(status_code=422, detail="each movie may be rated only once")
    check_ids([*ratings, *req.finished, *req.plan_to_watch], cat)
    bad = sorted(set(req.genres) - set(comb.gb_model.genres_) | (set(req.genres) & HIDDEN_GENRES))
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown genres: {bad}")

    allowed = None
    if req.genres:
        allowed = np.logical_or.reduce([_genre_masks[g] for g in req.genres])
    recs = comb.recommend_one(ratings, req.finished, req.plan_to_watch, allowed=allowed, k=req.k)

    out = []
    for r in recs:
        b = r["because"]
        because = BecauseOf(movie_id=b, title=cat.at[b, "title"]) if b in cat.index else None
        out.append(Recommendation(**to_movie(cat.loc[r["movieId"]]),
                                  predicted_rating=r["predicted_rating"], because=because))
    n_items = len(set(ratings) | set(req.finished) | set(req.plan_to_watch))
    used_gb = len(ratings) >= comb.min_ratings_for_gb
    note = None
    if n_items == 0:
        note = "Add movies to your lists or rate a few to personalize these. For now: popular, well-liked films."
    elif not used_gb:
        need = comb.min_ratings_for_gb - len(ratings)
        note = f"Rate {need} more movie{'s' if need != 1 else ''} to turn on star predictions."
    elif len(out) < req.k:
        note = f"Only {len(out)} films match this genre selection."
    return RecommendResponse(recommendations=out, n_ratings_used=len(ratings),
                             n_list_items_used=n_items, used_gradient_boost=used_gb,
                             min_ratings_for_gradient_boost=comb.min_ratings_for_gb,
                             genres=req.genres, note=note)
