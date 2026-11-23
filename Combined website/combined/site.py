"""What the website serves: the combined model, extended with films from 2017 to today.

The List taste model only knows films with 20+ MovieLens ratings by Nov 2019. To cover
newer films we reuse Yager's ``WatchlistRanker`` (FilmPalate/D4_submission/watchlist):
it wraps the same List taste model and gives every newer film a borrowed taste
fingerprint (the averaged fingerprint of its 10 closest catalog films by IMDb
description: genres, director, top cast, decade). That lets newer films

* count toward a person's taste when they're on their lists,
* show up in "More like this" (by description, for new films), and
* be ranked in a separate "New releases for you" row: IMDb buzz + 0.25 x IMDb rating
  + 2.0 x the person's taste for the film. Yager's backtest (2015-16 films held out as
  "new") found taste adds little offline precision (taste 0 -> 0.197, 0.25 -> 0.194,
  1.0 -> 0.168 tune precision@10, vs 0.170 for "most voted"), but with taste <= 0.25 the
  row was nearly the same for everyone (8 made-up users: 75% overlap, 10% in their
  genre). At 2.0: 11% overlap, 41% in their genre. We chose following the list; 2.0
  itself was not backtested.

Main picks are unchanged from the evaluated combined model: List taste model candidates
(popularity weight 2) re-ranked with the gradient boost model's predicted stars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .blend import CombinedRecommender


class SiteRecommender:
    def __init__(self, combined: CombinedRecommender, ranker, new_recommendable: np.ndarray):
        """``ranker``: fitted WatchlistRanker whose ``base_`` is ``combined.list_model``.
        ``new_recommendable``: boolean mask over the ranker's new films (well-known ones)."""
        self.combined = combined
        self.ranker = ranker
        self.new_recommendable = new_recommendable

    @property
    def film_ids(self) -> np.ndarray:
        return self.ranker.film_ids_

    @property
    def new_ids(self) -> np.ndarray:
        return self.ranker.film_ids_[self.ranker.n_base_:]

    def history(self, ratings: dict, finished=(), plan_to_watch=()) -> sp.csr_matrix:
        """One person's lists as a row over every film the ranker knows (catalog + new)."""
        lm = self.combined.list_model
        w = lm.weights_
        weights: dict[int, float] = {}
        finished = set(map(int, finished)) | set(map(int, ratings))
        for m in set(map(int, plan_to_watch)) - finished:
            weights[m] = w["plan_to_watch"]
        for m in finished:
            weights[m] = w["completed"]
        for m, stars in ratings.items():
            weights[int(m)] = float(lm.rating_weights(np.array([stars]))[0])
        ids = np.array(list(weights), dtype=np.int64)
        cols = self.ranker.film_index_.get_indexer(ids)
        ok = cols >= 0
        vals = np.array(list(weights.values()), dtype=np.float32)[ok]
        return sp.csr_matrix((vals, (np.zeros(ok.sum(), int), cols[ok])),
                             shape=(1, len(self.film_ids)))

    def _because(self, H, movie_ids) -> list[int | None]:
        pos = H.indices[H.data > 0]
        if not len(pos):
            return [None] * len(movie_ids)
        V = self.ranker.vectors_
        cols = self.ranker.film_index_.get_indexer(movie_ids)
        sims = V[cols] @ V[pos].T
        return [int(self.film_ids[pos[int(np.argmax(r))]]) for r in sims]

    def recommend(self, ratings: dict, finished=(), plan_to_watch=(), allowed_catalog=None,
                  allowed_new=None, k: int = 10, k_new: int = 10):
        """(main picks, new releases) as lists of dicts: movieId, predicted_rating, because."""
        comb, r = self.combined, self.ranker
        H = self.history(ratings, finished, plan_to_watch)
        s_cat, s_new = r.score(H)


        # Main picks: the evaluated combined model on catalog films.
        if allowed_catalog is not None:
            s_cat = np.where(allowed_catalog[None, :], s_cat, -np.inf)
        n = min(comb.n_candidates, s_cat.shape[1] - 1)
        top = np.argpartition(-s_cat, n, axis=1)[:, :n]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s_cat, top, 1), 1), 1)
        scores = np.take_along_axis(s_cat, top, 1)
        ids = comb.list_model.item_ids_[top]
        ids[~np.isfinite(scores)] = -1
        pred = None
        has_taste = bool(r.has_taste(H)[0])
        if has_taste and len(ratings) >= comb.min_ratings_for_gb:
            df = pd.DataFrame({"userId": 0, "movieId": list(ratings),
                               "rating": np.array(list(ratings.values()), dtype=np.float32)})
            pred, _ = comb.gb_predictions(comb.gb_model.profile_.encode(df, [0]), ids)
        final = comb.blend(scores, pred)
        order = [j for j in np.argsort(-final[0], kind="stable")
                 if ids[0, j] >= 0 and np.isfinite(final[0, j])][:k]
        pick_ids = [int(ids[0, j]) for j in order]
        because = self._because(H, pick_ids) if has_taste else [None] * len(pick_ids)
        picks = [{"movieId": m, "because": b,
                  "predicted_rating": (None if pred is None or np.isnan(pred[0, j])
                                       else round(float(pred[0, j]), 2))}
                 for m, b, j in zip(pick_ids, because, order)]

        # New releases: Yager's new-film ranking with a small taste term.
        mask = self.new_recommendable.copy()
        if allowed_new is not None:
            mask &= allowed_new
        s = np.where(mask, s_new[0], -np.inf)
        kk = min(k_new, int(np.isfinite(s).sum()))
        nt = np.argsort(-s)[:kk]
        new_ids = [int(self.new_ids[j]) for j in nt]
        nb_because = self._because(H, new_ids) if has_taste else [None] * len(new_ids)
        new = [{"movieId": m, "because": b, "predicted_rating": None}
               for m, b in zip(new_ids, nb_because)]
        return picks, new

    def similar(self, movie_id: int, k: int = 12) -> list[int]:
        try:
            return [m for m, _ in self.ranker.similar(movie_id, k)]
        except KeyError:
            return []
