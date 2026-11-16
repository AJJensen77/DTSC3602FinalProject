# Popularity baseline

The model every group candidate is compared against: **recommend the same popular,
well-liked movies to everyone** (D1's evaluation plan).

## Definition

1. Candidate films: at least 20 training ratings.
2. "Well liked": damped mean rating ≥ 3.5, where each film's mean is pulled toward the
   global mean by 50 pseudo-ratings (so three 5-star ratings don't count as "highly rated").
3. Rank by how many ratings the film received in the **last 365 days** of training.
4. Skip films the user has already rated; recommend the top 10.

The 365-day window was **tuned** on the tune half of the users, so the baseline is as
strong as we could make it rather than a straw man:

| Popularity window | Precision@10 (tune half) |
|---|---|
| All training years | 0.105 |
| Last 5 years | 0.144 |
| Last 2 years | 0.156 |
| **Last 1 year** | **0.168** |

## Results (report half, 2,953 users)

| Metric | Baseline |
|---|---|
| Precision@10 | 0.165 |
| Recall@10 | 0.039 |
| NDCG@10 | 0.180 |
| Hit rate (≥1 good pick of 10) | 59.4% |
| Liked share of picks the user watched | 78% |
| Distinct films recommended | 223 |

Full output: `results/baseline_results.json`, `results/baseline_tuning.csv`.

## Comparing another model against it

Evaluate on the **same users and split**: D2's scorable cohort (≥20 training ratings,
≥5 validation ratings), shuffled with seed 3602 and split 50/50 into tune/report halves.
Tune on the tune half, report on the report half. `evaluate_hybrid.py` exposes the shared
helpers (`load`, `cohort`, `per_user_metrics`, `summarize`, `bootstrap_diff`), and
`evaluate_baseline.py` shows the minimal pattern.

## Files

| File | What it is |
|---|---|
| `popularity.py` | `PopularityRecommender` |
| `evaluate_baseline.py` | Tunes the window and scores the report half |
| `results/` | Its outputs |

```bash
uv run python "Gradient boost model/baseline/evaluate_baseline.py"   # ~1 minute
```
