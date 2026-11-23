"""Combined recommender: Ishika's list taste model + Yager's gradient boost model.

How the two are mixed:

1. **Candidates come from the list taste model.** It folds everything on the person's
   lists (Finished watching, Plan to watch, and their star ratings) through the movie
   taste fingerprints and keeps the top ``n_candidates`` films. This is the model that
   beat popularity from the very first movie, and it can work from lists alone.
2. **Yager's gradient boost model predicts the person's star rating** for each candidate
   from their ratings (IMDb score, cast, director, similar films, ...).
3. **Final score = list score + gb_weight x (predicted stars, standardized per person).**
   Optionally, candidates predicted below 3.5 stars are moved to the back.

The gradient boost step needs ratings, so it is used only once the person has rated at
least ``min_ratings_for_gb`` films; before that, the list model ranks on its own.

The same class is used by ``evaluate_combined.py`` (to tune and test the blend) and by
the website API, so the site serves exactly what was evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

LIKE = 3.5
_DEMOTE = 1_000.0  # pushes "predicted < 3.5" films behind every other candidate


def _zrow(a: np.ndarray) -> np.ndarray:
    """Standardize each row, ignoring NaNs; NaNs become 0 (the row's average)."""
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, keepdims=True)
    z = (a - mu) / np.where(sd > 0, sd, 1)
    return np.nan_to_num(z, nan=0.0)


class CombinedRecommender:
    def __init__(self, list_model, gb_model, n_candidates: int = 100, gb_weight: float = 1.0,
                 demote_below_like: bool = False, min_ratings_for_gb: int = 5):
        self.list_model = list_model
        self.gb_model = gb_model
        self.n_candidates = n_candidates
        self.gb_weight = gb_weight
        self.demote_below_like = demote_below_like
        self.min_ratings_for_gb = min_ratings_for_gb
        p = gb_model.profile_
        self._gb_ids = pd.Index(p.movie_ids_[p.eligible_])

    # -- batch pieces (used by the evaluation) --------------------------------------

    def candidates(self, H_list: sp.csr_matrix, n: int | None = None,
                   allowed: np.ndarray | None = None,
                   exclude: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-n (movieIds, list scores) per row; films already on the lists are skipped.

        ``allowed`` is an optional boolean mask over the list model's films (genre filter).
        ``exclude`` optionally skips more films per row (same shape as ``H_list``).
        Rows with fewer than n allowed films are padded with movieId -1 / score -inf.
        """
        lm, n = self.list_model, n or self.n_candidates
        s = lm.score(H_list).astype(np.float32, copy=False)
        s[H_list.nonzero()] = -np.inf
        if exclude is not None:
            s[exclude.nonzero()] = -np.inf
        if allowed is not None:
            s[:, ~allowed] = -np.inf
        n = min(n, s.shape[1] - 1)
        top = np.argpartition(-s, n, axis=1)[:, :n]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)
        scores = np.take_along_axis(s, top, 1)
        ids = lm.item_ids_[top]
        ids[~np.isfinite(scores)] = -1
        return ids, scores

    def gb_predictions(self, H_gb: sp.csr_matrix, cand_ids: np.ndarray,
                       extra_mask: np.ndarray | None = None):
        """Predicted stars for each candidate (NaN where the gradient boost model has none).

        Returns (candidate predictions, full prediction block). ``extra_mask`` adds more
        films to the block, e.g. Yager's own candidate set when scoring his model alone.
        """
        cols = self._gb_ids.get_indexer(cand_ids.ravel()).reshape(cand_ids.shape)
        only = np.zeros(len(self._gb_ids), dtype=bool)
        only[cols[(cols >= 0) & (cand_ids >= 0)]] = True
        if extra_mask is not None:
            only |= extra_mask
        block = self.gb_model.predict_block(H_gb, only)
        pred = np.take_along_axis(block, np.where(cols >= 0, cols, 0), 1)
        pred[(cols < 0) | (cand_ids < 0)] = np.nan
        return pred, block

    def blend(self, list_scores: np.ndarray, pred: np.ndarray | None,
              gb_weight: float | None = None, demote: bool | None = None) -> np.ndarray:
        w = self.gb_weight if gb_weight is None else gb_weight
        d = self.demote_below_like if demote is None else demote
        out = list_scores.copy()
        if pred is None or (not w and not d):
            return out
        if w:
            out += w * _zrow(pred)
        if d:
            out -= _DEMOTE * (pred < LIKE)  # NaN compares False: unknown films are not demoted
        return out

    @staticmethod
    def top_k(ids: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        out = np.take_along_axis(ids, order, 1)
        out[~np.isfinite(np.take_along_axis(scores, order, 1))] = -1
        return out

    # -- one website user ------------------------------------------------------------

    def recommend_one(self, ratings: dict[int, float], finished=(), plan_to_watch=(),
                      allowed: np.ndarray | None = None, k: int = 10) -> list[dict]:
        """Recommendations for one person.

        ``ratings`` maps movieId -> stars. Rated films count as Finished watching.
        Returns dicts with movieId, list_score, predicted_rating (None when the gradient
        boost model is not used yet) and because (the list film it is most similar to).
        """
        lm = self.list_model
        finished = set(map(int, finished)) | set(map(int, ratings))
        plan = set(map(int, plan_to_watch)) - finished
        H = lm.lists_to_history({"completed": sorted(finished), "plan_to_watch": sorted(plan)},
                                ratings)
        if H.nnz == 0 or (H.data <= 0).all():
            ids = lm.recommend_from_lists({}, k=max(k * 3, 30))  # popular, well-liked films
            seen = finished | plan
            ids = [int(m) for m in ids if int(m) not in seen]
            if allowed is not None:
                ok = set(lm.item_ids_[allowed].tolist())
                ids = [m for m in ids if m in ok]
            return [{"movieId": m, "list_score": None, "predicted_rating": None, "because": None}
                    for m in ids[:k]]

        ids, scores = self.candidates(H, allowed=allowed)
        pred = None
        if len(ratings) >= self.min_ratings_for_gb:
            df = pd.DataFrame({"userId": 0, "movieId": list(ratings),
                               "rating": np.array(list(ratings.values()), dtype=np.float32)})
            H_gb = self.gb_model.profile_.encode(df, [0])
            pred, _ = self.gb_predictions(H_gb, ids)
        final = self.blend(scores, pred)
        order = np.argsort(-final[0], kind="stable")
        order = [j for j in order if ids[0, j] >= 0 and np.isfinite(final[0, j])][:k]

        # "Because you liked X": the positively weighted list film closest in taste.
        pos_cols = H.indices[H.data > 0]
        vecs = lm.item_vectors_
        out = []
        for j in order:
            m = int(ids[0, j])
            because = None
            if len(pos_cols):
                c = lm.item_index_.get_indexer([m])[0]
                sims = vecs[pos_cols] @ vecs[c]
                because = int(lm.item_ids_[pos_cols[int(np.argmax(sims))]])
            p = None if pred is None or np.isnan(pred[0, j]) else round(float(pred[0, j]), 2)
            out.append({"movieId": m, "list_score": float(scores[0, j]),
                        "predicted_rating": p, "because": because})
        return out

    def similar(self, movie_id: int, k: int = 12) -> list[int]:
        try:
            return self.list_model.similar_movies(movie_id, k)["movieId"].astype(int).tolist()
        except KeyError:
            return []
