"""Hybrid rating predictor: a user's taste profile + movie metadata -> predicted stars.

For a (user, movie) pair the model sees

* **who the user is** - how many films they have rated, their average, spread, and
  how often they rate 3.5+;
* **what the movie is** - its MovieLens rating count and mean, a *blended prior*
  that falls back on the movie's IMDb rating when MovieLens ratings are thin, the
  raw IMDb rating and vote count, and release year;
* **how this user feels about this kind of movie** - the user's average surprise
  (rating minus what their average and the movie's prior predicted) on films that
  share the movie's genres, its top-billed actors, and its director;
* **what similar movies say** - an item-kNN score: the user's surprise on films
  that the same people tend to rate, weighted by similarity.

A gradient-boosted regressor maps these to a star rating. ``HybridRecommender``
then recommends the highest-predicted films that clear 3.5 stars.

Everything is computed from a user's *ratings*, never from a stored user id, so the
same code can score a brand-new user who has just rated a few films.

Speed comes from expressing every profile feature as a sparse matrix product: for a
batch of users with history matrix ``H`` (users x films) and a film-to-genre
incidence matrix ``G``, ``H @ G`` is every user's per-genre total at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor

from .recommenders import LIKE_THRESHOLD, ItemKNNRecommender

NO_GENRE = "(no genres listed)"


def _incidence(lists: pd.Series, min_films: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Film-by-group 0/1 matrix from a Series of lists (genres, actors, directors).

    Groups attached to fewer than ``min_films`` films are dropped: a person seen in
    one film carries no information about any *other* film.
    """
    exploded = lists.explode().dropna()
    counts = exploded.value_counts()
    exploded = exploded[exploded.isin(counts[counts >= min_films].index)]
    codes, groups = pd.factorize(exploded)
    M = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (exploded.index.to_numpy(), codes)),
        shape=(len(lists), len(groups)),
    )
    return M, np.asarray(groups)


def _per_row(M: sp.csr_matrix) -> np.ndarray:
    return np.diff(M.indptr).astype(np.float32)


class TasteProfileTransformer(BaseEstimator, TransformerMixin):
    """Turns users' rating histories into per-(user, movie) features.

    ``fit(history)`` learns population-level state from a ratings frame: movie rating
    statistics, the IMDb-to-MovieLens calibration, genre/cast/director incidence, and
    item-kNN similarities. Users' own profiles are computed at scoring time from
    whatever history is passed in.

    ``feature_groups`` switches parts off for ablation: any of
    ``{"imdb", "people", "knn"}`` may be omitted.
    """

    def __init__(self, movie_features: pd.DataFrame, feature_groups=("imdb", "people", "knn"),
                 prior_strength=20, genre_shrinkage=10.0, people_shrinkage=3.0,
                 knn_neighbors=100, knn_min_ratings=20, min_votes_floor=500,
                 recent_days=365):
        self.movie_features = movie_features
        self.feature_groups = feature_groups
        self.prior_strength = prior_strength
        self.genre_shrinkage = genre_shrinkage
        self.people_shrinkage = people_shrinkage
        self.knn_neighbors = knn_neighbors
        self.knn_min_ratings = knn_min_ratings
        self.min_votes_floor = min_votes_floor
        self.recent_days = recent_days

    # -- fitting -------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        groups = set(self.feature_groups)
        mf = self.movie_features.reset_index(drop=True)
        self.movie_ids_ = mf["movieId"].to_numpy(np.int64)
        self.movie_index_ = pd.Index(self.movie_ids_)
        n_all = len(mf)

        cols = self.movie_index_.get_indexer(X["movieId"])
        if (cols < 0).any():
            raise ValueError("history contains movieIds missing from movie_features")
        r = X["rating"].to_numpy(np.float64)
        self.mu_ = float(r.mean())
        count = np.bincount(cols, minlength=n_all).astype(np.float64)
        total = np.bincount(cols, weights=r, minlength=n_all)
        ml_mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        # What is being watched *now*: ratings in the last ``recent_days`` of history.
        ts = X["timestamp"].to_numpy()
        recent = ts >= ts.max() - self.recent_days * 86400
        self.recent_count_ = np.bincount(cols[recent], minlength=n_all).astype(np.float64)

        imdb = mf["averageRating"].to_numpy(np.float64)
        votes = mf["numVotes"].fillna(0).to_numpy(np.float64)
        use_imdb = "imdb" in groups
        if use_imdb:
            # Map IMDb's 1-10 scale onto MovieLens stars with a line fit on films
            # that have plenty of MovieLens ratings; that line is the fill-in.
            ok = (count >= 100) & ~np.isnan(imdb)
            slope, intercept = np.polyfit(imdb[ok], ml_mean[ok], 1)
            self.imdb_calibration_ = (float(slope), float(intercept))
            prior = np.where(np.isnan(imdb), self.mu_, intercept + slope * imdb)
        else:
            self.imdb_calibration_ = None
            prior = np.full(n_all, self.mu_)
        m = self.prior_strength
        self.blended_ = (total + m * prior) / (count + m)
        self.movie_bias_ = (self.blended_ - self.mu_).astype(np.float32)

        # Films the model can say anything useful about.
        self.eligible_ = np.flatnonzero((count > 0) | (use_imdb & (votes >= self.min_votes_floor)))
        e = self.eligible_
        self.count_ = count
        self.votes_ = votes
        self.year_ = mf["year"].to_numpy(np.float64)

        static = {
            "movie_log_count": np.log1p(count[e]),
            "movie_log_recent_count": np.log1p(self.recent_count_[e]),
            "movie_mean": ml_mean[e],
            "movie_blended_prior": self.blended_[e],
        }
        if use_imdb:
            static["imdb_rating"] = imdb[e]
            static["imdb_log_votes"] = np.log1p(votes[e])
        static["year"] = self.year_[e]
        self._static_names = list(static)
        self._static = np.column_stack(list(static.values())).astype(np.float32)

        genres = mf["genres"].fillna("").str.split("|").apply(
            lambda g: [x for x in g if x and x != NO_GENRE])
        self._groups = {"genre": (*_incidence(genres, 1), self.genre_shrinkage)}
        if "people" in groups:
            self._groups["actor"] = (*_incidence(mf["actors"], 2), self.people_shrinkage)
            self._groups["director"] = (*_incidence(mf["directors"], 2), self.people_shrinkage)
        # Eligible-film slices, transposed, for projecting a user's group scores
        # back onto films.
        self._group_e = {
            k: (M, M[e].T.tocsr(), _per_row(M[e]), shrink)
            for k, (M, _names, shrink) in self._groups.items()
        }

        if "knn" in groups:
            knn = ItemKNNRecommender(min_item_ratings=self.knn_min_ratings,
                                     n_neighbors=self.knn_neighbors).fit(X)
            # Re-index the similarity matrix from kNN film order to (all films ->
            # eligible films), so a user's full history multiplies straight in.
            rows = self.movie_index_.get_indexer(knn.item_ids_)
            P = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, np.arange(len(rows)))),
                              shape=(n_all, len(rows)))
            pos_in_e = pd.Index(e).get_indexer(rows)
            keep = pos_in_e >= 0
            Q = sp.csr_matrix((np.ones(keep.sum(), np.float32),
                               (np.flatnonzero(keep), pos_in_e[keep])),
                              shape=(len(rows), len(e)))
            S = (P @ knn.similarity_ @ Q).tocsr()
            self._knn = S
            self._knn_abs = abs(S)
        else:
            self._knn = None

        names = ["user_log_n", "user_mean", "user_std", "user_like_share", "baseline_pred"]
        names += self._static_names
        for k in self._group_e:
            names += [f"{k}_affinity", f"{k}_support"]
        if self._knn is not None:
            names += ["knn_score", "knn_support"]
        self.feature_names_ = names
        return self

    # -- scoring -------------------------------------------------------------

    def encode(self, ratings: pd.DataFrame, user_ids) -> sp.csr_matrix:
        """Users' known ratings as rows of a users x all-films sparse matrix."""
        user_index = pd.Index(user_ids)
        r = ratings[ratings["userId"].isin(user_index)]
        cols = self.movie_index_.get_indexer(r["movieId"])
        ok = cols >= 0
        return sp.csr_matrix(
            (r["rating"].to_numpy(np.float32)[ok],
             (user_index.get_indexer(r["userId"].to_numpy()[ok]), cols[ok])),
            shape=(len(user_index), len(self.movie_ids_)),
        )

    def block(self, H: sp.csr_matrix) -> np.ndarray:
        """Features for every (user in H, eligible film): shape (users, films, features)."""
        H = H.tocsr().astype(np.float32)
        H.sort_indices()
        B = H.copy()
        B.data[:] = 1.0
        n = np.asarray(B.sum(axis=1)).ravel()
        safe_n = np.maximum(n, 1)
        mean = np.where(n > 0, np.asarray(H.sum(axis=1)).ravel() / safe_n, self.mu_)
        sq = np.asarray(H.multiply(H).sum(axis=1)).ravel() / safe_n
        std = np.sqrt(np.maximum(sq - mean**2, 0))
        like = np.asarray((H >= LIKE_THRESHOLD).sum(axis=1)).ravel() / safe_n

        # Surprise: how far each rating sits from what the user's average and the
        # film's prior would have predicted. Affinities average these.
        R = H.copy()
        row_of = np.repeat(np.arange(H.shape[0]), np.diff(H.indptr))
        R.data = H.data - mean[row_of].astype(np.float32) - self.movie_bias_[H.indices]

        n_u, n_e = H.shape[0], len(self.eligible_)
        feats = [
            np.broadcast_to(np.log1p(n)[:, None], (n_u, n_e)),
            np.broadcast_to(mean[:, None], (n_u, n_e)),
            np.broadcast_to(std[:, None], (n_u, n_e)),
            np.broadcast_to(like[:, None], (n_u, n_e)),
            mean[:, None] + self.movie_bias_[self.eligible_][None, :],
        ]
        feats += [np.broadcast_to(self._static[:, j][None, :], (n_u, n_e))
                  for j in range(self._static.shape[1])]

        for M, MeT, per_film, shrink in self._group_e.values():
            sums = (R @ M).tocsr()
            cnt = (B @ M).tocsr()
            inv = cnt.copy()
            inv.data = 1.0 / (inv.data + shrink)
            aff = sums.multiply(inv).tocsr()
            with np.errstate(invalid="ignore", divide="ignore"):
                feats.append(_dense(aff @ MeT) / per_film)
                feats.append(_dense(cnt @ MeT) / per_film)

        if self._knn is not None:
            num = _dense(R @ self._knn)
            den = _dense(B @ self._knn_abs)
            with np.errstate(invalid="ignore", divide="ignore"):
                feats.append(np.where(den > 0, num / den, np.nan))
            feats.append(den)

        return np.stack(feats, axis=-1).astype(np.float32)

    def pair_features(self, history: pd.DataFrame, pairs: pd.DataFrame, batch: int = 128):
        """Feature rows for (userId, movieId) pairs, built from ``history`` only.

        Pairs on films outside the eligible set are dropped; returns the features
        and the positional index of the pairs that were kept.
        """
        e_index = pd.Index(self.movie_ids_[self.eligible_])
        col = e_index.get_indexer(pairs["movieId"])
        pairs = pairs.assign(_col=col, _pos=np.arange(len(pairs)))
        pairs = pairs[pairs["_col"] >= 0].sort_values("userId", kind="stable")
        pair_users = pairs["userId"].to_numpy()
        users = np.unique(pair_users)
        history = history[history["userId"].isin(users)].sort_values("userId", kind="stable")
        hist_users = history["userId"].to_numpy()
        out_X, out_pos = [], []
        for start in range(0, len(users), batch):
            ub = users[start : start + batch]
            lo = np.searchsorted(hist_users, ub[0], side="left")
            hi = np.searchsorted(hist_users, ub[-1], side="right")
            F = self.block(self.encode(history.iloc[lo:hi], ub))
            plo = np.searchsorted(pair_users, ub[0], side="left")
            phi = np.searchsorted(pair_users, ub[-1], side="right")
            p = pairs.iloc[plo:phi]
            ui = pd.Index(ub).get_indexer(p["userId"])
            out_X.append(F[ui, p["_col"].to_numpy()])
            out_pos.append(p["_pos"].to_numpy())
        return np.concatenate(out_X), np.concatenate(out_pos)


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


class HybridRecommender(BaseEstimator):
    """Taste-profile features -> HistGradientBoostingRegressor -> top-k films >= 3.5.

    Ranking knobs (tuned on validation):
      ``min_item_ratings`` / ``min_votes``: a film is a candidate if it has at least
          this many MovieLens ratings, *or* this many IMDb votes (the IMDb route is
          how films with no MovieLens ratings can still be recommended).
      ``popularity_weight`` / ``popularity_source``: adds ``w * log(1 + count)`` to
          the predicted rating when ordering candidates, where count is all-time
          ("all") or last-year ("recent") MovieLens ratings. A high predicted rating
          says the user would like a film; popularity says they are likely to
          actually watch it. Only films predicted >= 3.5 are ever eligible.
      ``max_year``: excludes films released after the history ends (they did not
          exist yet at prediction time).
    """

    def __init__(self, movie_features: pd.DataFrame, profile_params=None,
                 regressor_params=None, min_item_ratings=20, min_votes=None,
                 popularity_weight=0.0, popularity_source="recent", max_year=None,
                 min_predicted=LIKE_THRESHOLD):
        self.movie_features = movie_features
        self.profile_params = profile_params
        self.regressor_params = regressor_params
        self.min_item_ratings = min_item_ratings
        self.min_votes = min_votes
        self.popularity_weight = popularity_weight
        self.popularity_source = popularity_source
        self.max_year = max_year
        self.min_predicted = min_predicted

    def fit(self, history: pd.DataFrame, targets: pd.DataFrame):
        """Learn from ``targets`` ratings, describing each with ``history`` only."""
        X, y = self.fit_features(history, targets)
        return self.fit_regressor(X, y)

    def fit_features(self, history: pd.DataFrame, targets: pd.DataFrame):
        """Fit the profile on ``history`` and return the training matrix for ``targets``."""
        self.profile_ = TasteProfileTransformer(self.movie_features,
                                                **(self.profile_params or {})).fit(history)
        X, pos = self.profile_.pair_features(history, targets)
        return X, targets["rating"].to_numpy(np.float32)[pos]

    def fit_regressor(self, X: np.ndarray, y: np.ndarray):
        params = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                      early_stopping=True, random_state=3602)
        params.update(self.regressor_params or {})
        self.regressor_ = HistGradientBoostingRegressor(**params).fit(X, y)
        self.n_train_rows_ = len(y)
        return self

    def refresh(self, history: pd.DataFrame):
        """Rebuild population state from newer history; the regressor is unchanged."""
        self.profile_ = clone(self.profile_).fit(history)
        return self

    def predict_block(self, H: sp.csr_matrix, only: np.ndarray | None = None) -> np.ndarray:
        """Predicted stars for every eligible film; NaN outside ``only`` if given.

        Restricting to candidate films skips regressor work whose output ``rank``
        would discard anyway, so it changes speed, not recommendations.
        """
        F = self.profile_.block(H)
        if only is not None:
            F = F[:, only]
        n_u, n_c, n_f = F.shape
        pred = np.clip(self.regressor_.predict(F.reshape(-1, n_f)).reshape(n_u, n_c), 0.5, 5.0)
        if only is None:
            return pred.astype(np.float32)
        out = np.full((n_u, len(only)), np.nan, dtype=np.float32)
        out[:, only] = pred
        return out

    @property
    def genres_(self) -> list[str]:
        return [str(g) for g in self.profile_._groups["genre"][1]]

    def candidate_mask(self, min_item_ratings=None, min_votes=None, max_year=None,
                       genres=None) -> np.ndarray:
        p = self.profile_
        e = p.eligible_
        mir = self.min_item_ratings if min_item_ratings is None else min_item_ratings
        mv = self.min_votes if min_votes is None else min_votes
        mask = p.count_[e] >= mir
        if mv is not None and "imdb" in set(p.feature_groups):
            mask |= p.votes_[e] >= mv
        my = self.max_year if max_year is None else max_year
        if my is not None:
            # Anything already rated in the history existed; otherwise trust IMDb's year.
            mask &= (p.count_[e] > 0) | (p.year_[e] <= my)
        if genres:
            # Keep films carrying any of the requested genres.
            M, names, _ = p._groups["genre"]
            cols = pd.Index(names).get_indexer(list(genres))
            if (cols < 0).any():
                raise ValueError(f"unknown genre in {genres!r}")
            mask &= np.asarray(M[e][:, cols].sum(axis=1)).ravel() > 0
        return mask

    def rank(self, pred: np.ndarray, H: sp.csr_matrix, k=10, mask=None,
             popularity_weight=None, popularity_source=None) -> np.ndarray:
        """Top-k movieIds from a prediction block; -1 pads rows with fewer than k."""
        p = self.profile_
        w = self.popularity_weight if popularity_weight is None else popularity_weight
        mask = self.candidate_mask() if mask is None else mask
        src = self.popularity_source if popularity_source is None else popularity_source
        count = p.recent_count_ if src == "recent" else p.count_
        score = pred + w * np.log1p(count[p.eligible_])[None, :].astype(np.float32)
        score = np.where(mask[None, :] & (pred >= self.min_predicted), score, -np.inf)
        seen = H.tocsr()[:, p.eligible_].nonzero()
        score[seen] = -np.inf
        top = np.argpartition(-score, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(score, top, 1), 1), 1)
        ids = p.movie_ids_[p.eligible_][top]
        ids[~np.isfinite(np.take_along_axis(score, top, 1))] = -1
        return ids

    def recommend_from_ratings(self, ratings, genres=None, k=10) -> list[tuple[int, float]]:
        """One person's [(movieId, stars), ...] -> up to k (movieId, predicted stars).

        This is the whole contract the website relies on: ratings in, films out.
        Fewer than k come back when not enough films clear 3.5 stars (e.g. under a
        narrow genre filter).
        """
        df = pd.DataFrame(list(ratings), columns=["movieId", "rating"]).assign(userId=0)
        H = self.profile_.encode(df, [0])
        mask = self.candidate_mask(genres=genres)
        pred = self.predict_block(H, mask)
        ids = self.rank(pred, H, k, mask=mask)[0]
        col = pd.Index(self.profile_.movie_ids_[self.profile_.eligible_])
        return [(int(m), float(pred[0, col.get_loc(m)])) for m in ids if m >= 0]

    def recommend(self, history: pd.DataFrame, user_ids, k=10, batch=128) -> np.ndarray:
        out = []
        for start in range(0, len(user_ids), batch):
            H = self.profile_.encode(history, user_ids[start : start + batch])
            out.append(self.rank(self.predict_block(H, self.candidate_mask()), H, k))
        return np.concatenate(out)
