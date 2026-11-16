"""IMDb movie metadata for the hybrid model: rating, votes, top-billed cast, director.

IMDb's files are large (``title.principals`` alone is ~800 MB compressed), so each
is read in chunks and immediately filtered to the tconsts in our catalog.
"""

import pandas as pd

# Same read conventions as catalog.load_imdb_types: IMDb writes missing values as
# a literal backslash-N and never quotes fields.
_READ = dict(sep="\t", na_values=r"\N", quoting=3, dtype="str")

N_TOP_ACTORS = 3
ACTOR_CATEGORIES = frozenset({"actor", "actress"})


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
    cast = cast.sort_values(["tconst", "ordering"])
    cast = cast.groupby("tconst").head(n)
    return cast.groupby("tconst")["nconst"].agg(list).rename("actors").reset_index()


def load_directors(path, keep: set[str]) -> pd.DataFrame:
    crew = pd.concat(_filtered_chunks(path, ["tconst", "directors"], keep))
    crew = crew.dropna(subset=["directors"])
    crew["directors"] = crew["directors"].str.split(",")
    return crew[["tconst", "directors"]]
