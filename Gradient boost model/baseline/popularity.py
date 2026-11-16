"""The popularity baseline: the same well-liked, popular films for everyone.

It follows the shared recommender contract in ``gbmodel/recommenders.py`` so it is
scored by exactly the same evaluation code as the gradient boost model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbmodel.recommenders import LIKE_THRESHOLD, BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Baseline: the same well-liked, popular films for everyone.

    Films are ranked by how many ratings they received, after keeping only films
    whose *damped* mean rating clears ``min_mean``. Damping pulls a film's mean
    toward the global mean by ``prior_strength`` pseudo-ratings, so a film with
    three 5-star ratings does not count as "highly rated".

    ``window_days`` restricts the popularity count to the last N days of the
    training data. Popularity drifts, so a recent window is a stronger opponent;
    it is tuned on validation like every other setting.
    """

    def __init__(self, min_item_ratings=20, min_mean=LIKE_THRESHOLD,
                 prior_strength=50, window_days=None):
        self.min_item_ratings = min_item_ratings
        self.min_mean = min_mean
        self.prior_strength = prior_strength
        self.window_days = window_days

    def fit(self, ratings: pd.DataFrame, y=None):
        super().fit(ratings)
        mu = ratings["rating"].mean()
        stats = ratings.groupby("movieId")["rating"].agg(["sum", "size"])
        stats = stats.reindex(self.item_ids_)
        m = self.prior_strength
        self.damped_mean_ = ((stats["sum"] + m * mu) / (stats["size"] + m)).to_numpy()

        recent = ratings
        if self.window_days is not None:
            cutoff = ratings["timestamp"].max() - self.window_days * 86400
            recent = ratings[ratings["timestamp"] >= cutoff]
        pop = recent.groupby("movieId").size().reindex(self.item_ids_).fillna(0)
        self.popularity_ = pop.to_numpy(dtype=np.float64)

        score = self.popularity_.copy()
        score[self.damped_mean_ < self.min_mean] = -np.inf
        score[self.popularity_ == 0] = -np.inf
        self.scores_ = score
        return self

    def _fit_matrix(self, X):
        pass

    def score(self, history):
        return np.broadcast_to(self.scores_, (history.shape[0], len(self.scores_))).copy()
