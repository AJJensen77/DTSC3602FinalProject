"""Movie recommendation API: a person's ratings in, ten films they should like out.

The website only depends on this contract, so the group can swap in a different
model behind it without touching the site.

Local dev (from the repo root):
    uv run uvicorn --app-dir "Gradient boost model" api.serve:app --reload
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
from pydantic import BaseModel, Field, field_validator

# The pickled model refers to classes in the sibling `gbmodel` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARTIFACTS = Path(__file__).parent / "artifacts"
MIN_RATINGS = 10
HIDDEN_GENRES = {"IMAX"}  # a screen format, not a genre people filter by

# MovieLens files "The Dark Knight" as "Dark Knight, The"; people search the other way.
_ARTICLE = re.compile(r"^(.+?), (The|A|An)( \(.*\))?$")


def display_title(title: str) -> str:
    return _ARTICLE.sub(r"\2 \1\3", title)


def load():
    try:
        bundle = joblib.load(ARTIFACTS / "hybrid_bundle.joblib")
        catalog = pd.read_parquet(ARTIFACTS / "catalog.parquet")
    except Exception as exc:  # noqa: BLE001 - any load failure should surface as 503
        return None, None, str(exc)
    catalog["title"] = catalog["title"].map(display_title)
    catalog["search_key"] = catalog["title"].str.lower()
    return bundle, catalog.set_index("movieId", drop=False), None


# Loaded once per server process - never reloaded or refit per request.
_bundle, _catalog, _load_error = load()

app = FastAPI(
    title="Movie Recommender API",
    version="1.0.0",
    description="Rate at least 10 films; get up to 10 more predicted at 3.5 stars or higher.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def require():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifact not loaded: {_load_error}")
    return _bundle["model"], _catalog


# -- schemas ------------------------------------------------------------------


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    imdb_rating: float | None


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
    ratings: list[RatingIn] = Field(..., max_length=5000)
    genres: list[str] = Field(default_factory=list, description="Optional: only these genres (any match)")
    k: int = Field(10, ge=1, le=20)


class Recommendation(Movie):
    predicted_rating: float


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    n_ratings_used: int
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


# -- routes -------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "service": "Movie Recommender API",
        "docs": "/docs",
        "model": "ok" if _bundle is not None else "artifact not loaded",
        "endpoints": {
            "GET /health": "liveness check",
            "GET /info": "model metadata",
            "GET /genres": "genres usable as filters",
            "GET /movies/starter": "well-known films for a new user to rate",
            "GET /movies/search": "title search",
            "GET /movies/lookup": "details for a list of movie ids",
            "POST /recommend": f"{MIN_RATINGS}+ ratings -> recommendations",
        },
    }


@app.get("/health")
def health():
    require()
    return {"status": "ok"}


@app.get("/info")
def info():
    model, _ = require()
    return {**_bundle["metadata"], "min_ratings": MIN_RATINGS}


@app.get("/genres", response_model=list[str])
def genres():
    model, _ = require()
    return sorted(g for g in model.genres_ if g not in HIDDEN_GENRES)


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


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    model, cat = require()

    ids = [r.movie_id for r in req.ratings]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="each movie may be rated only once")
    if len(ids) < MIN_RATINGS:
        raise HTTPException(status_code=422,
                            detail=f"rate at least {MIN_RATINGS} movies (got {len(ids)})")
    unknown = [i for i in ids if i not in cat.index]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown movie ids: {unknown[:10]}")
    bad = sorted(set(req.genres) - set(model.genres_) | (set(req.genres) & HIDDEN_GENRES))
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown genres: {bad}")

    recs = model.recommend_from_ratings([(r.movie_id, r.rating) for r in req.ratings],
                                        genres=req.genres or None, k=req.k)
    out = [Recommendation(**to_movie(cat.loc[m]), predicted_rating=round(p, 2)) for m, p in recs]
    note = None
    if len(out) < req.k:
        note = (f"Only {len(out)} films in this selection are predicted at "
                f"{model.min_predicted} stars or higher for you.")
    return RecommendResponse(recommendations=out, n_ratings_used=len(ids),
                             genres=req.genres, note=note)
