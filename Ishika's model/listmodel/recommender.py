"""The list-based taste model.

Idea
----
Every film gets a short "taste fingerprint" (a vector of latent factors) learned from
who watched what in the MovieLens training split. Films watched by the same kinds of
people end up with similar fingerprints.

* **Recommend from lists.** A user's watch lists (Completed, Currently Watching,
  Plan to Watch, Dropped) become one signed weight per film. Folding those weights
  through the fingerprints gives a score for every film in the catalog; Dropped films
  carry a negative weight, so films like them are pushed *down*.
* **More like this.** The films whose fingerprints point the same way as the clicked
  film's (cosine similarity).

The fingerprints come from a truncated SVD of the user-by-film matrix (scikit-learn
``TruncatedSVD``), the "PureSVD" recommender of Cremonesi et al. (2010). Scoring folds
a user in from their lists, so it works for users the model has never seen -- which is
what a website needs, since site users are not MovieLens users.

MovieLens has ratings, not lists, so for training and offline evaluation each rating is
mapped onto the list it most likely stands for (see ``rating_to_list``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gbmodel.recommenders import LIKE_THRESHOLD, BaseRecommender

LISTS = ("completed", "watching", "plan_to_watch", "dropped")

# How much each list says about a user's taste. Completed is the strongest positive
# signal; Plan to Watch only shows interest; Dropped is evidence *against* similar films.
DEFAULT_LIST_WEIGHTS = {
    "completed": 1.0,
    "watching": 0.8,
    "plan_to_watch": 0.5,
    "dropped": -1.0,
}

# Stored zeros vanish from a sparse matrix, which would let an already-listed film be
# recommended again. Any listed film keeps at least this much weight.
_EPS = 1e-3


def rating_to_list(rating: np.ndarray, dropped_at_or_below: float = 2.0) -> np.ndarray:
    """MovieLens stand-in for lists: low ratings -> 'dropped', the rest -> 'completed'."""
    return np.where(rating <= dropped_at_or_below, "dropped", "completed")


class ListTasteRecommender(BaseRecommender):
    """Matrix-factorization recommender driven by signed watch-list weights.

    Parameters
    ----------
    n_factors : length of each film's taste fingerprint.
    min_item_ratings : films need this many training ratings to be recommendable
        (same catalog rule as the baseline and the hybrid model).
    popularity_damping : before the SVD, each film's column is divided by
        ``n_ratings ** popularity_damping``. 0 = plain PureSVD, which is dominated by
        blockbusters; larger values let niche films shape the factors more.
    list_weights : weight per list (see ``DEFAULT_LIST_WEIGHTS``).
    meh_weight : weight for a completed film the user rated 2.5-3 stars (watched it,
        did not love it). Only used when a rating is known.
    dropped_at_or_below : rating at or below which a MovieLens rating counts as Dropped.
    popularity_weight : how much recent popularity is blended into the final ranking.
        Taste decides which films a user would *like*; recent popularity helps decide
        which of those they are likely to actually *watch* next.
    window_days : the popularity window (last N days of training), as in the baseline.
    min_mean : optional quality floor -- skip films whose damped mean rating is below it.
    """

    def __init__(self, n_factors=128, min_item_ratings=20, popularity_damping=0.5,
                 list_weights=None, meh_weight=0.25, dropped_at_or_below=2.0,
                 popularity_weight=0.0, window_days=365, min_mean=None,
                 prior_strength=50, random_state=3602):
        self.n_factors = n_factors
        self.min_item_ratings = min_item_ratings
        self.popularity_damping = popularity_damping
        self.list_weights = list_weights
        self.meh_weight = meh_weight
        self.dropped_at_or_below = dropped_at_or_below
        self.popularity_weight = popularity_weight
        self.window_days = window_days
        self.min_mean = min_mean
        self.prior_strength = prior_strength
        self.random_state = random_state

    # -- list encoding -------------------------------------------------------------

    @property
    def weights_(self) -> dict:
        return {**DEFAULT_LIST_WEIGHTS, **(self.list_weights or {})}

    def rating_weights(self, rating: np.ndarray) -> np.ndarray:
        """Signed list weight for each MovieLens rating."""
        w = self.weights_
        rating = np.asarray(rating, dtype=np.float32)
        out = np.where(rating >= LIKE_THRESHOLD, w["completed"], self.meh_weight)
        out = np.where(rating <= self.dropped_at_or_below, w["dropped"], out)
        out = np.where(out == 0, _EPS, out)
        return out.astype(np.float32)

    def history_matrix(self, ratings: pd.DataFrame, user_ids) -> sp.csr_matrix:
        """Users' MovieLens ratings encoded as signed list weights (one row per user)."""
        H = super().history_matrix(ratings, user_ids)
        H.data = self.rating_weights(H.data)
        return H

    def lists_to_history(self, lists: dict, ratings: dict | None = None) -> sp.csr_matrix:
        """Encode one website user's lists as a 1-row history.

        ``lists`` maps list name -> iterable of movieIds, e.g.
        ``{"completed": [1, 260], "plan_to_watch": [2571], "dropped": [480]}``.
        ``ratings`` optionally maps movieId -> stars for completed films; a known
        rating refines the Completed weight the same way it does in training.
        Films outside the catalog are ignored. If a film appears on several lists,
        the later list in ``LISTS`` order wins (e.g. Dropped beats Watching).
        """
        unknown = set(lists) - set(LISTS)
        if unknown:
            raise ValueError(f"unknown list(s) {sorted(unknown)}; expected {LISTS}")
        w, ratings = self.weights_, ratings or {}
        weight_by_film = {}
        for name in LISTS:
            for movie_id in lists.get(name, ()):
                weight = w[name]
                if name == "completed" and movie_id in ratings:
                    weight = float(self.rating_weights(np.array([ratings[movie_id]]))[0])
                weight_by_film[int(movie_id)] = weight or _EPS
        cols = self.item_index_.get_indexer(list(weight_by_film))
        vals = np.array(list(weight_by_film.values()), dtype=np.float32)
        ok = cols >= 0
        return sp.csr_matrix((vals[ok], (np.zeros(ok.sum(), dtype=int), cols[ok])),
                             shape=(1, len(self.item_ids_)))

    # -- fitting -------------------------------------------------------------------

    def fit(self, ratings: pd.DataFrame, y=None):
        super().fit(ratings)  # catalog + calls _fit_matrix
        idx = self.item_index_
        mu = ratings["rating"].mean()
        stats = ratings.groupby("movieId")["rating"].agg(["sum", "size"]).reindex(idx)
        m = self.prior_strength
        self.damped_mean_ = ((stats["sum"] + m * mu) / (stats["size"] + m)).to_numpy()

        recent = ratings
        if self.window_days is not None:
            cutoff = ratings["timestamp"].max() - self.window_days * 86400
            recent = ratings[ratings["timestamp"] >= cutoff]
        pop = recent.groupby("movieId").size().reindex(idx).fillna(0).to_numpy(np.float64)
        self.recent_popularity_ = pop
        lp = np.log1p(pop)
        self.popularity_z_ = ((lp - lp.mean()) / lp.std()).astype(np.float32)
        return self

    def _fit_matrix(self, X: sp.csr_matrix) -> None:
        W = X.copy()
        W.data = self.rating_weights(W.data)
        self.column_scale_ = (self.item_counts_.astype(np.float64)
                              ** -self.popularity_damping).astype(np.float32)
        W = W @ sp.diags(self.column_scale_)
        svd = TruncatedSVD(n_components=self.n_factors, algorithm="randomized",
                           n_iter=7, random_state=self.random_state)
        svd.fit(W)
        V = svd.components_.T.astype(np.float32)          # (n_items, n_factors)
        self.item_factors_ = V
        self.singular_values_ = svd.singular_values_.astype(np.float32)
        # Fingerprints for "more like this": factors scaled by their strength.
        self.item_vectors_ = normalize(V * self.singular_values_).astype(np.float32)

    def with_factors(self, n_factors: int) -> "ListTasteRecommender":
        """Copy keeping only the first ``n_factors`` factors, without refitting.

        Randomized SVD returns factors strongest-first, so the leading columns of a
        larger fit approximate a smaller fit; this makes tuning ``n_factors`` cheap.
        """
        clone = ListTasteRecommender(**{**self.get_params(), "n_factors": n_factors})
        for attr, val in vars(self).items():
            if attr.endswith("_") and not attr.startswith("__"):
                setattr(clone, attr, val)
        V = self.item_factors_[:, :n_factors]
        s = self.singular_values_[:n_factors]
        clone.item_factors_, clone.singular_values_ = V, s
        clone.item_vectors_ = normalize(V * s).astype(np.float32)
        return clone

    # -- scoring -------------------------------------------------------------------

    def taste(self, history: sp.csr_matrix) -> np.ndarray:
        """Pure taste score: the user's lists folded through the film fingerprints."""
        h = history @ sp.diags(self.column_scale_)
        user_factors = np.asarray(h @ self.item_factors_)  # (n_users, n_factors)
        return user_factors @ self.item_factors_.T

    def score(self, history: sp.csr_matrix) -> np.ndarray:
        s = self.taste(history).astype(np.float32)
        # Put taste on a per-user standard scale so one popularity weight fits everyone.
        sd = s.std(axis=1, keepdims=True)
        s = (s - s.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)
        if self.popularity_weight:
            s += self.popularity_weight * self.popularity_z_
        if self.min_mean is not None:
            s[:, self.damped_mean_ < self.min_mean] = -np.inf
        return s

    # -- website-facing helpers ------------------------------------------------------

    def recommend_from_lists(self, lists: dict, k: int = 10, ratings: dict | None = None):
        """Top-k movieIds for one website user, given their lists ("Because you added...")."""
        H = self.lists_to_history(lists, ratings)
        if H.nnz == 0 or (H.data <= 0).all():
            # Nothing positive to go on yet: fall back to popular, well-liked films.
            s = np.where(self.damped_mean_ >= LIKE_THRESHOLD, self.popularity_z_, -np.inf)
            s = np.asarray(s, dtype=np.float32)[None, :]
            s[H.nonzero()] = -np.inf
            top = np.argsort(-s[0])[:k]
            return self.item_ids_[top[np.isfinite(s[0, top])]]
        recs = self.recommend(H, k)[0]
        return recs[recs >= 0]

    def similar_movies(self, movie_id: int, k: int = 10) -> pd.DataFrame:
        """"More like this": the k films whose taste fingerprints are closest."""
        j = self.item_index_.get_indexer([movie_id])[0]
        if j < 0:
            raise KeyError(f"movieId {movie_id} is not in the recommendable catalog "
                           f"(needs >= {self.min_item_ratings} training ratings)")
        sims = self.item_vectors_ @ self.item_vectors_[j]
        sims[j] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return pd.DataFrame({"movieId": self.item_ids_[top], "similarity": sims[top]})
