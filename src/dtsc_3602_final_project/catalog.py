"""Build the movie catalog and decide which MovieLens entries are TV.

MovieLens has no title-type field, so the catalog is joined to IMDb's official
``title.basics.tsv.gz`` through ``links.csv`` (which carries an imdbId for every
row). ``titleType`` is then authoritative for dropping television.
"""

import re

import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

# A "movie" for this project is a self-contained film. tvMovie (made-for-TV
# film) and video (direct-to-video film) are films; short is a short film.
FILM_TYPES = frozenset({"movie", "tvMovie", "video", "short"})

# Episodic / non-film television, plus the handful of games MovieLens carries.
TV_TYPES = frozenset(
    {"tvSeries", "tvMiniSeries", "tvEpisode", "tvSpecial", "tvShort", "videoGame"}
)


def load_imdb_types(path) -> pd.DataFrame:
    """Return tconst -> titleType for every IMDb title."""
    basics = pd.read_csv(
        path,
        sep="\t",
        usecols=["tconst", "titleType"],
        dtype={"tconst": "str", "titleType": "category"},
        na_values=r"\N",  # IMDb encodes missing values as a literal backslash-N
        quoting=3,  # IMDb does not quote fields; csv.QUOTE_NONE
    )
    return basics


def build_catalog(movies: pd.DataFrame, links: pd.DataFrame, imdb: pd.DataFrame) -> pd.DataFrame:
    """Join MovieLens movies to IMDb title types and parse title/year/genres."""
    catalog = movies.merge(links[["movieId", "imdbId"]], on="movieId", how="left")

    # links.csv stores imdbId as a bare integer; IMDb uses zero-padded "tt" ids.
    catalog["tconst"] = "tt" + catalog["imdbId"].astype("Int64").astype("str").str.zfill(7)
    catalog = catalog.merge(imdb, on="tconst", how="left")

    catalog["year"] = catalog["title"].str.extract(YEAR_RE)[0].astype("Float64")
    catalog["clean_title"] = catalog["title"].str.replace(YEAR_RE, "", regex=True).str.strip()
    # Unmatched ids must land in a real "unknown" bucket, not stay NaN: spot
    # checks show they are ordinary films with a stale imdbId, so they are kept.
    catalog["titleType"] = catalog["titleType"].astype("object").fillna("unknown").astype("str")

    catalog["is_tv"] = catalog["titleType"].isin(TV_TYPES)
    catalog["is_film"] = catalog["titleType"].isin(FILM_TYPES)
    return catalog
