"""IMDb metadata for films the MovieLens data never saw (2017-2026 releases).

Adapted from ``Gradient boost model/gbmodel/imdb_features.py`` (copied, not imported,
so this app does not depend on the gradient boost model). IMDb's files are large, so
each one is read in chunks and filtered to the titles we need straight away.
"""

import pandas as pd

# IMDb writes missing values as a literal backslash-N and never quotes fields.
_READ = dict(sep="\t", na_values=r"\N", quoting=3, dtype="str")

N_TOP_ACTORS = 3
ACTOR_CATEGORIES = frozenset({"actor", "actress"})

# IMDb genre names -> MovieLens genre names. IMDb-only genres (Biography, History,
# Sport, Music, ...) have no MovieLens counterpart and are dropped.
IMDB_TO_MOVIELENS_GENRE = {
    "Action": "Action", "Adventure": "Adventure", "Animation": "Animation",
    "Comedy": "Comedy", "Crime": "Crime", "Documentary": "Documentary", "Drama": "Drama",
    "Family": "Children", "Fantasy": "Fantasy", "Horror": "Horror", "Musical": "Musical",
    "Mystery": "Mystery", "Romance": "Romance", "Sci-Fi": "Sci-Fi", "Thriller": "Thriller",
    "War": "War", "Western": "Western",
}


def _filtered_chunks(path, usecols, keep: set[str], chunksize: int = 2_000_000):
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, **_READ):
        yield chunk[chunk["tconst"].isin(keep)]


def load_ratings(path, keep: set[str]) -> pd.DataFrame:
    df = pd.concat(_filtered_chunks(path, ["tconst", "averageRating", "numVotes"], keep))
    return df.astype({"averageRating": "float32", "numVotes": "int64"})


def load_top_actors(path, keep: set[str], n: int = N_TOP_ACTORS) -> pd.DataFrame:
    """tconst -> list of the ``n`` first-billed actors/actresses (IMDb nconst ids)."""
    parts = []
    for chunk in _filtered_chunks(path, ["tconst", "ordering", "nconst", "category"], keep):
        parts.append(chunk[chunk["category"].isin(ACTOR_CATEGORIES)])
    cast = pd.concat(parts)
    cast["ordering"] = cast["ordering"].astype("int32")
    cast = cast.sort_values(["tconst", "ordering"]).groupby("tconst").head(n)
    return cast.groupby("tconst")["nconst"].agg(list).rename("actors").reset_index()


def load_directors(path, keep: set[str]) -> pd.DataFrame:
    crew = pd.concat(_filtered_chunks(path, ["tconst", "directors"], keep))
    crew = crew.dropna(subset=["directors"])
    crew["directors"] = crew["directors"].str.split(",")
    return crew[["tconst", "directors"]]


def load_recent_films(path, min_year: int, exclude: set[str]) -> pd.DataFrame:
    """Feature films from ``min_year`` on in ``title.basics``, minus ``exclude`` and
    adult titles. Returns tconst, title, year and MovieLens-style genres."""
    parts = []
    cols = ["tconst", "titleType", "primaryTitle", "isAdult", "startYear", "genres"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=2_000_000, **_READ):
        year = pd.to_numeric(chunk["startYear"], errors="coerce")
        keep = (chunk["titleType"] == "movie") & (year >= min_year) & (chunk["isAdult"] == "0")
        parts.append(chunk[keep & ~chunk["tconst"].isin(exclude)].assign(year=year[keep]))
    films = pd.concat(parts)
    films["genres"] = films["genres"].fillna("").str.split(",").apply(
        lambda gs: "|".join(dict.fromkeys(IMDB_TO_MOVIELENS_GENRE[g] for g in gs
                                          if g in IMDB_TO_MOVIELENS_GENRE))
        or "(no genres listed)")
    return films.rename(columns={"primaryTitle": "title"})[["tconst", "title", "year", "genres"]]
