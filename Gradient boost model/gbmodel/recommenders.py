"""Shared recommender contract, plus the item-kNN similarity the hybrid model uses.

The popularity baseline built on this contract lives in ``baseline/popularity.py``.

Both follow the same contract:

* ``fit(ratings)`` learns only from a ratings frame (``userId, movieId, rating``),
  in practice the training split.
* The catalog is the set of films with at least ``min_item_ratings`` training
  ratings.
* ``recommend(history, k)`` takes a user-by-film sparse matrix of the ratings each
  user has already placed (built with ``history_matrix``) and returns the top-k
  ``movieId`` per user, never repeating a film the user has already rated.

Scoring from a user's *history* rather than a stored user id means the same code
serves a user the model has never seen, which is what the D4 API will need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator

# D1's definition of "a movie the user liked".
LIKE_THRESHOLD = 3.5


class BaseRecommender(BaseEstimator):
    """Shared catalog handling, history encoding, and top-k selection."""

    min_item_ratings: int

    # -- fitting -----------------------------------------------------------

    def fit(self, ratings: pd.DataFrame, y=None):
        counts = ratings.groupby("movieId").size()
        keep = counts[counts >= self.min_item_ratings]
        self.item_ids_ = keep.index.to_numpy(dtype=np.int64)
        self.item_index_ = pd.Index(self.item_ids_)
        self.item_counts_ = keep.to_numpy(dtype=np.int64)

        r = ratings[ratings["movieId"].isin(self.item_index_)]
        user_codes, self.train_user_ids_ = pd.factorize(r["userId"], sort=True)
        item_codes = self.item_index_.get_indexer(r["movieId"])
        X = sp.csr_matrix(
            (r["rating"].to_numpy(np.float32), (user_codes, item_codes)),
            shape=(len(self.train_user_ids_), len(self.item_ids_)),
        )
        self._fit_matrix(X)
        return self

    def _fit_matrix(self, X: sp.csr_matrix) -> None:
        raise NotImplementedError

    # -- scoring -----------------------------------------------------------

    def history_matrix(self, ratings: pd.DataFrame, user_ids) -> sp.csr_matrix:
        """Encode each user's known ratings as one sparse row in model item space.

        Ratings on films outside the recommendable catalog are dropped: the model
        has no parameters for them.
        """
        user_index = pd.Index(user_ids)
        r = ratings[ratings["userId"].isin(user_index)]
        cols = self.item_index_.get_indexer(r["movieId"])
        ok = cols >= 0
        rows = user_index.get_indexer(r["userId"].to_numpy()[ok])
        return sp.csr_matrix(
            (r["rating"].to_numpy(np.float32)[ok], (rows, cols[ok])),
            shape=(len(user_index), len(self.item_ids_)),
        )

    def score(self, history: sp.csr_matrix) -> np.ndarray:
        """Dense (n_users, n_items) scores; higher means recommend sooner."""
        raise NotImplementedError

    def recommend(self, history: sp.csr_matrix, k: int = 10, batch: int = 1000) -> np.ndarray:
        """Top-k movieIds per row of ``history``; -1 pads rows with fewer than k."""
        out = np.full((history.shape[0], k), -1, dtype=np.int64)
        for start in range(0, history.shape[0], batch):
            h = history[start : start + batch]
            s = self.score(h).astype(np.float32, copy=False)
            # Never re-recommend something the user has already rated.
            s[h.nonzero()] = -np.inf
            top = np.argpartition(-s, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            ids = self.item_ids_[top]
            ids[~np.isfinite(np.take_along_axis(s, top, axis=1))] = -1
            out[start : start + batch] = ids
        return out


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with cosine similarity.

    Two films are similar when the same users rated them. Each film keeps its
    ``n_neighbors`` most similar films; a user's score for a film is the summed
    similarity to the films already in their history.
    """

    def __init__(self, min_item_ratings=20, n_neighbors=200, weighting="binary",
                 shrinkage=10.0, block=1000):
        self.min_item_ratings = min_item_ratings
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.shrinkage = shrinkage
        self.block = block

    def _weight(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.copy().astype(np.float32)
        if self.weighting == "binary":
            X.data[:] = 1.0
        elif self.weighting != "rating":
            raise ValueError(f"unknown weighting {self.weighting!r}")
        return X

    def _fit_matrix(self, X):
        Xw = self._weight(X)
        norms = np.sqrt(np.asarray(Xw.multiply(Xw).sum(axis=0))).ravel()
        Xc = Xw.tocsc()
        XT = Xw.T.tocsr()
        n_items, k = Xw.shape[1], self.n_neighbors
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            dots = (XT @ Xc[:, start:stop]).toarray()
            # Shrinkage discounts similarities resting on few co-ratings.
            sims = dots / (np.outer(norms, norms[start:stop]) + self.shrinkage)
            sims[np.arange(start, stop), np.arange(stop - start)] = 0.0
            top = np.argpartition(-sims, k, axis=0)[:k]
            rows.append(top.ravel(order="F"))
            cols.append(np.repeat(np.arange(start, stop), k))
            vals.append(np.take_along_axis(sims, top, axis=0).ravel(order="F"))
        self.similarity_ = sp.csr_matrix(
            (np.concatenate(vals).astype(np.float32),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )

    def truncated(self, n_neighbors: int) -> "ItemKNNRecommender":
        """Copy keeping only the top ``n_neighbors`` per film, without refitting."""
        S = self.similarity_.tocsc()
        r, c, v = [], [], []
        for j in range(S.shape[1]):
            lo, hi = S.indptr[j], S.indptr[j + 1]
            idx, val = S.indices[lo:hi], S.data[lo:hi]
            keep = np.argsort(-val)[:n_neighbors]
            r.append(idx[keep]); c.append(np.full(len(keep), j)); v.append(val[keep])
        clone = ItemKNNRecommender(self.min_item_ratings, n_neighbors, self.weighting,
                                   self.shrinkage, self.block)
        for attr in ("item_ids_", "item_index_", "item_counts_", "train_user_ids_"):
            setattr(clone, attr, getattr(self, attr))
        clone.similarity_ = sp.csr_matrix(
            (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=S.shape
        )
        return clone

    def score(self, history):
        return (self._weight(history) @ self.similarity_).toarray()

