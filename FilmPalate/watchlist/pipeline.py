"""The fitted scikit-learn Pipeline the website's API serves.

    Pipeline([
        ("encode", WatchlistEncoder()),   # site lists -> signed taste weights
        ("rank",   WatchlistRanker()),    # List taste model + new-film fingerprints
    ])

Input to the pipeline is one dict per website user::

    {"ratings": {movieId: stars, ...}, "want_to_watch": [movieId, ...]}

The site has two lists instead of the List taste model's four:

* **My Ratings** (replaces *Completed*): 0.5-5 stars once a film is finished. The
  star rating picks the weight exactly as the model was trained on MovieLens:
  >= 3.5 liked (+1.0), 2.5-3 so-so (+0.25), <= 2 disliked (-1.0, which replaces the
  old *Dropped* list).
* **Want to Watch** (replaces *Plan to Watch*): +0.5. Interest, but weaker evidence
  than a good rating. Rating a film moves it from Want to Watch to My Ratings.

*Currently Watching* and *Dropped* are not used.

New films. The List taste model only knows films with >= 20 MovieLens ratings
before the reserved test period (Nov 2019), so nothing recent. ``WatchlistRanker``
gives each 2017-2026 film the averaged fingerprint of its closest catalog films by
IMDb description (genres, director, top cast, decade). That places it in the taste
space for "More like this" and lets a user's rating of a new film feed their taste.

Ranking *which* new films to show is a different matter: ``evaluate_new_films.py``
found that predicted fingerprints (and genre preferences) do not tell liked from
disliked new films, while IMDb rating does. So the "New releases" row ranks by IMDb
buzz plus rating, not by personal taste (``new_taste_weight`` = 0 by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from listmodel.recommender import ListTasteRecommender

LIKED_AT = 3.5            # D1's definition of a liked film
DISLIKED_AT_OR_BELOW = 2.0


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.zeros(0, np.float32)
    return ((x - x.mean()) / (x.std() or 1)).astype(np.float32)


# -- 1. site lists -> signed weights ---------------------------------------------------


class WatchlistEncoder(BaseEstimator, TransformerMixin):
    """Turn each user's My Ratings + Want to Watch into one signed sparse row.

    ``film_ids`` is the ranker's film order (set when the pipeline is assembled).
    Rated films outrank Want to Watch: a film that is both counts as rated.
    Raises ``ValueError`` on unknown films or invalid stars, which the API turns
    into HTTP 422.
    """

    def __init__(self, film_ids=None, liked_weight=1.0, so_so_weight=0.25,
                 disliked_weight=-1.0, want_to_watch_weight=0.5):
        self.film_ids = film_ids
        self.liked_weight = liked_weight
        self.so_so_weight = so_so_weight
        self.disliked_weight = disliked_weight
        self.want_to_watch_weight = want_to_watch_weight

    def fit(self, X=None, y=None):
        if self.film_ids is None:
            raise ValueError("WatchlistEncoder needs film_ids (the ranker's film order)")
        self.film_index_ = pd.Index(np.asarray(self.film_ids, dtype=np.int64))
        return self

    def star_weight(self, stars: float) -> float:
        if stars >= LIKED_AT:
            return self.liked_weight
        if stars <= DISLIKED_AT_OR_BELOW:
            return self.disliked_weight
        return self.so_so_weight

    def transform(self, X) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, user in enumerate(X):
            ratings = {int(m): float(s) for m, s in (user.get("ratings") or {}).items()}
            for stars in ratings.values():
                if not (0.5 <= stars <= 5.0) or round(stars * 2) != stars * 2:
                    raise ValueError(f"stars must be 0.5-5.0 in half steps, got {stars}")
            weights = {int(m): self.want_to_watch_weight
                       for m in (user.get("want_to_watch") or ())}
            weights.update({m: self.star_weight(s) for m, s in ratings.items()})
            ids = list(weights)
            idx = self.film_index_.get_indexer(ids)
            if (idx < 0).any():
                unknown = [m for m, j in zip(ids, idx) if j < 0]
                raise ValueError(f"unknown movie ids: {unknown[:10]}")
            rows += [i] * len(ids)
            cols += idx.tolist()
            # A weight of exactly 0 would vanish from the sparse matrix and let the
            # film be recommended back; keep a tiny positive weight instead.
            vals += [w or 1e-3 for w in weights.values()]
        return sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                             shape=(len(X), len(self.film_index_)))


# -- 2. film description -> features ----------------------------------------------------


class FilmContentFeatures(BaseEstimator, TransformerMixin):
    """IMDb description of a film -> sparse feature row.

    Genres (multi-hot), directors and top-3 cast who appear in at least
    ``min_person_films`` films (multi-hot), decade (one-hot), plus standardized IMDb
    rating and log vote count. People seen too rarely carry no reusable signal and are
    dropped, which keeps the vocabulary to a few thousand columns.
    """

    def __init__(self, min_person_films=3):
        self.min_person_films = min_person_films

    @staticmethod
    def _people(df) -> pd.Series:
        def as_list(v):
            return list(v) if isinstance(v, (list, tuple, np.ndarray)) else []
        d = df["directors"].map(lambda v: ["d:" + p for p in as_list(v)])
        a = df["actors"].map(lambda v: ["a:" + p for p in as_list(v)])
        return d + a

    @staticmethod
    def _genres(df) -> pd.Series:
        return df["genres"].fillna("").map(
            lambda g: [x for x in g.split("|") if x and x != "(no genres listed)"])

    def fit(self, X: pd.DataFrame, y=None):
        self.genres_ = sorted({g for gs in self._genres(X) for g in gs})
        counts = self._people(X).explode().value_counts()
        self.people_ = counts[counts >= self.min_person_films].index.tolist()
        self.decades_ = list(range(1910, 2040, 10))
        r = X["averageRating"].astype(float)
        v = np.log1p(X["numVotes"].astype(float))
        self.rating_mean_, self.rating_sd_ = float(r.mean()), float(r.std())
        self.votes_mean_, self.votes_sd_ = float(v.mean()), float(v.std())
        self.vocab_ = {f: j for j, f in enumerate(
            ["g:" + g for g in self.genres_] + self.people_
            + [f"decade:{d}" for d in self.decades_])}
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        rows, cols = [], []
        decade = (X["year"].fillna(2000) // 10 * 10).astype(int).clip(1910, 2030)
        for i, (gs, ps, dec) in enumerate(zip(self._genres(X), self._people(X), decade)):
            for f in ["g:" + g for g in gs] + ps + [f"decade:{dec}"]:
                j = self.vocab_.get(f)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        onehot = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                               shape=(len(X), len(self.vocab_)))
        r = (X["averageRating"].astype(float).fillna(self.rating_mean_) - self.rating_mean_) / self.rating_sd_
        v = (np.log1p(X["numVotes"].astype(float).fillna(0)) - self.votes_mean_) / self.votes_sd_
        dense = sp.csr_matrix(np.column_stack([r, v]).astype(np.float32))
        return sp.hstack([onehot, dense], format="csr")


# -- 3. ranking -------------------------------------------------------------------------


class WatchlistRanker(BaseEstimator):
    """Ishika's List taste model, extended to films it never saw and to empty lists.

    Catalog films (``>= min_item_ratings`` MovieLens ratings) are scored exactly as
    ``ListTasteRecommender.score`` does: taste folded in from the lists, standardized
    per user, plus ``popularity_weight`` x recent popularity.

    New films get the averaged fingerprint of their ``content_neighbors`` most
    similar catalog films (by IMDb description) and are ranked for their own
    "New releases" row by IMDb buzz (log votes per year since release, standardized)
    + ``new_rating_weight`` x standardized IMDb rating
    + ``new_taste_weight`` x the user's standardized taste score.
    """

    def __init__(self, n_factors=192, popularity_damping=0.3, popularity_weight=6.0,
                 window_days=365, min_item_ratings=20, content_neighbors=10,
                 min_person_films=3, new_rating_weight=0.25, new_taste_weight=0.0,
                 content_popularity_weight=0.05, popularity_schedule=None, newest_year=2026):
        self.n_factors = n_factors
        self.popularity_damping = popularity_damping
        self.popularity_weight = popularity_weight
        self.window_days = window_days
        self.min_item_ratings = min_item_ratings
        self.content_neighbors = content_neighbors
        self.min_person_films = min_person_films
        self.new_rating_weight = new_rating_weight
        self.new_taste_weight = new_taste_weight
        self.content_popularity_weight = content_popularity_weight
        self.popularity_schedule = popularity_schedule
        self.newest_year = newest_year

    # -- fitting ---------------------------------------------------------------

    def fit(self, ratings: pd.DataFrame, films: pd.DataFrame, new_films: pd.DataFrame | None = None):
        """``films``: IMDb description (movieId, genres, year, actors, directors,
        averageRating, numVotes) for MovieLens films. ``new_films``: same columns for
        films to add from their description alone."""
        base = ListTasteRecommender(
            n_factors=self.n_factors, popularity_damping=self.popularity_damping,
            popularity_weight=self.popularity_weight, window_days=self.window_days,
            min_item_ratings=self.min_item_ratings).fit(ratings)
        self.base_ = base
        V, s = base.item_factors_, base.singular_values_

        # Content features of the catalog films that have IMDb data.
        desc = films.set_index("movieId").reindex(base.item_ids_)
        known = np.flatnonzero(desc["numVotes"].notna().to_numpy())
        self.content_features_ = FilmContentFeatures(self.min_person_films).fit(
            desc.iloc[known].reset_index())
        catalog_x = self._content_unit(desc.iloc[known].reset_index())

        if new_films is None or len(new_films) == 0:
            new_films = films.iloc[:0]
        new_films = new_films[~new_films["movieId"].isin(base.item_ids_)].reset_index(drop=True)
        V_new = np.zeros((len(new_films), V.shape[1]), np.float32)
        if len(new_films):
            # Each new film borrows the similarity-weighted mean fingerprint of its
            # closest catalog films by description.
            sim = (self._content_unit(new_films) @ catalog_x.T).toarray()
            k = min(self.content_neighbors, sim.shape[1] - 1)
            top = np.argpartition(-sim, k, axis=1)[:, :k]
            w = np.maximum(np.take_along_axis(sim, top, axis=1), 0)
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
            V_new = np.einsum("nk,nkf->nf", w, V[known][top]).astype(np.float32)

        self.n_base_ = len(base.item_ids_)
        self.film_ids_ = np.concatenate([base.item_ids_, new_films["movieId"].to_numpy(np.int64)])
        self.film_index_ = pd.Index(self.film_ids_)
        self.factors_ = np.vstack([V, V_new]).astype(np.float32)
        # Fold-in scale for new films: a typical catalog film's (they have no counts).
        self.scale_ = np.concatenate([base.column_scale_, np.full(len(new_films),
                                      np.median(base.column_scale_), np.float32)])
        self.vectors_ = normalize(self.factors_ * s).astype(np.float32)
        # Description vectors for every film, for "More like this" on new films.
        all_desc = pd.concat([desc.reset_index(), new_films], ignore_index=True)
        self.content_x_ = self._content_unit(all_desc).astype(np.float32)
        votes = np.log1p(all_desc["numVotes"].astype(float).fillna(0).to_numpy())
        self.votes_z_ = _z(votes)

        year = new_films["year"].astype(float).fillna(self.newest_year)
        age = (self.newest_year + 1 - year).clip(lower=1)
        buzz = np.log1p(new_films["numVotes"].astype(float).fillna(0) / age).to_numpy()
        self.new_buzz_z_ = _z(buzz)
        self.new_imdb_rating_ = new_films["averageRating"].astype(float).to_numpy()
        self.new_rating_z_ = _z(np.nan_to_num(self.new_imdb_rating_,
                                              nan=np.nanmean(self.new_imdb_rating_) if len(buzz) else 0))
        return self

    def _content_unit(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Unit-length description vectors (genres, people, decade) for matching."""
        x = self.content_features_.transform(df)[:, :-2]   # drop rating/votes columns
        return normalize(x).tocsr()

    # -- scoring ---------------------------------------------------------------

    def _check(self, H) -> sp.csr_matrix:
        H = sp.csr_matrix(H)
        if H.shape[1] != len(self.film_ids_):
            raise ValueError(f"expected {len(self.film_ids_)} columns, got {H.shape[1]}")
        return H

    def taste_z(self, H) -> np.ndarray:
        """Per-user standardized taste for every film (catalog first, then new)."""
        h = self._check(H) @ sp.diags(self.scale_)
        u = np.asarray(h @ self.factors_)
        t = (u @ self.factors_.T).astype(np.float32)
        # Standardize on catalog films only, so this matches ListTasteRecommender.score.
        b = t[:, : self.n_base_]
        sd = b.std(axis=1, keepdims=True)
        return (t - b.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)

    def user_popularity_weight(self, H) -> np.ndarray:
        """Popularity weight per user, from how much their lists say.

        ``evaluate_blend.py`` found that with few ratings the model only beats popular
        picks if popularity counts a lot, and that the best weight falls as ratings
        grow. "How much the lists say" = sum of |list weights|, so a rating counts 1
        and a Want to Watch 0.5. Beyond the last step, ``popularity_weight`` applies.
        """
        H = self._check(H)
        evidence = np.asarray(abs(H).sum(axis=1)).ravel()
        w = np.full(len(evidence), float(self.popularity_weight))
        for upto, weight in reversed(getattr(self, "popularity_schedule", None) or ()):
            w[evidence <= upto] = weight
        return w

    def has_taste(self, H) -> np.ndarray:
        """True per user if their lists contain anything positive to go on."""
        H = self._check(H)
        return np.asarray((H.maximum(0)).sum(axis=1)).ravel() > 0

    def score(self, H) -> tuple[np.ndarray, np.ndarray]:
        """(catalog scores, new-film scores); films already on a list get -inf.

        Users with nothing positive on their lists get the cold-start ranking:
        catalog = recent popularity among well-liked films (Ishika's fallback),
        new films = IMDb buzz among well-reviewed films.
        """
        H = self._check(H)
        z = self.taste_z(H)
        base = z[:, : self.n_base_] + self.user_popularity_weight(H)[:, None] * self.base_.popularity_z_
        new = (self.new_taste_weight * z[:, self.n_base_:] + self.new_buzz_z_
               + self.new_rating_weight * self.new_rating_z_)
        cold = ~self.has_taste(H)
        if cold.any():
            liked = self.base_.damped_mean_ >= LIKED_AT
            base[cold] = np.where(liked, self.base_.popularity_z_, -np.inf)
            good = np.nan_to_num(self.new_imdb_rating_, nan=0) >= 6.5
            new[cold] = np.where(good, self.new_buzz_z_ + self.new_rating_weight
                                 * self.new_rating_z_, -np.inf)
        seen = H.nonzero()
        full = np.hstack([base, new])
        full[seen] = -np.inf
        return full[:, : self.n_base_], full[:, self.n_base_:]

    @staticmethod
    def _top(s: np.ndarray, ids: np.ndarray, k: int) -> list[list[int]]:
        out = []
        for row in s:
            k_ = min(k, len(row))
            top = np.argpartition(-row, k_ - 1)[:k_] if k_ else np.array([], int)
            top = top[np.argsort(-row[top])]
            out.append([int(ids[j]) for j in top if np.isfinite(row[j])])
        return out

    def predict(self, H, k: int = 10) -> list[list[int]]:
        """Top-k catalog films per user (the evaluated "Top picks" list)."""
        base, _ = self.score(H)
        return self._top(base, self.film_ids_[: self.n_base_], k)

    def predict_new(self, H, k: int = 10) -> list[list[int]]:
        """Top-k new (2017-2026) films per user."""
        _, new = self.score(H)
        return self._top(new, self.film_ids_[self.n_base_:], k)

    def similar(self, movie_id: int, k: int = 10) -> list[tuple[int, float]]:
        """"More like this".

        Catalog films: nearest taste fingerprints (films watched by the same kinds of
        people). New films have only a borrowed fingerprint, so for them similarity
        comes straight from the description (shared director, cast, genres, decade),
        with a small nudge toward well-known films so obscure titles that merely
        share a genre don't crowd out the obvious matches.
        """
        j = self.film_index_.get_indexer([movie_id])[0]
        if j < 0:
            raise KeyError(movie_id)
        if j >= self.n_base_:
            sims = np.asarray((self.content_x_ @ self.content_x_[j].T).todense()).ravel()
            sims = sims + self.content_popularity_weight * self.votes_z_
        else:
            sims = self.vectors_ @ self.vectors_[j]
        sims[j] = -np.inf
        k = min(k, len(sims) - 1)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.film_ids_[t]), float(sims[t])) for t in top]


def build_pipeline(ranker: WatchlistRanker, **encoder_params) -> Pipeline:
    """Assemble the served pipeline around an already fitted ranker."""
    encoder = WatchlistEncoder(film_ids=ranker.film_ids_, **encoder_params).fit()
    return Pipeline([("encode", encoder), ("rank", ranker)])
