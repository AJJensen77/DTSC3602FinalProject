# List taste model — results (for the D3 modeling note)

**Author:** Ishika Bhattarai · Full numbers: `results/list_model_results.json` · Figures: `figures/fig9–fig11`

**Short answer:** On the 2,953 report-half users (never used for tuning), the list taste
model got **2.1 good picks out of 10 vs 1.6 for the popularity baseline** (precision@10
0.208 vs 0.165, 95% CI on the difference +0.038 to +0.049) and recommended 2.6× more
distinct films. Same users, split, and metric code as the baseline and the gradient
boost model, so all three are directly comparable. Reserved test data untouched (checked).

## Selected settings (chosen on the tune half)

| Setting | Tried | Selected |
|---|---|---|
| Popularity damping before SVD | 0.3, 0.6 | **0.3** |
| Taste factors | 64, 128, 192 | **192** |
| Popularity blend weight | 0 – 10 | **6.0** |
| 3.5★ quality floor | off / on | **off** |

Tune-half precision@10 0.213 (baseline on tune half: 0.168). 108 settings tried; full grid in
`results/list_model_tuning.csv`.

## Report half: model vs. baseline (2,953 users)

| Metric | List taste model | Popularity baseline | Difference (95% CI) |
|---|---|---|---|
| Precision@10 (main) | **0.208** | 0.165 | +0.044 (+0.038, +0.049) |
| Recall@10 | **0.052** | 0.039 | +0.013 (+0.011, +0.016) |
| NDCG@10 | **0.225** | 0.180 | +0.046 (+0.039, +0.051) |
| Hit rate (≥1 good pick of 10) | **71.0%** | 59.4% | +11.6 pts (+10.0, +13.2) |
| Liked share of picks the user watched | **84%** | 78% | |
| Distinct films recommended | **569** | 223 | |
| Share of recs from top-1% films | **51%** | 63% | |
| Median training ratings of a recommended film | **13,734** | 25,961 | |

For reference, the gradient boost model's published report-half numbers on the same users
are precision@10 0.175, recall 0.043, NDCG 0.190, hit rate 62.6%, 763 distinct films.

## What each design choice contributes (report half)

| Variant | Precision@10 | Distinct films |
|---|---|---|
| **Selected list model** | **0.208** | 569 |
| Taste only (no popularity blend) | 0.183 | 2,696 |
| All lists weighted equally (no negative Dropped) | 0.194 | 547 |
| Dropped ignored (weight 0) | 0.208 | 564 |
| Popularity baseline | 0.165 | 223 |

- **Taste alone already beats the baseline** (0.183 vs 0.165) while recommending 12× more
  distinct films. Blending in recent popularity adds +0.025 by helping pick which liked
  films the user is likely to actually watch next — at the cost of narrower coverage.
- **Signed lists matter:** treating every listed film as positive drops precision to 0.194.
  Low-rated films carrying less (or negative) weight is what helps.
- **The Dropped weight specifically adds ~nothing offline** (0.208 either way). In
  MovieLens "Dropped" is only a stand-in (ratings ≤ 2★, ~10% of ratings), so this does not
  test real Dropped lists — see limitations.

## Error analysis (report half, precision@10)

| Group | Bucket (n users) | List model | Baseline | Diff 95% CI |
|---|---|---|---|---|
| History size | 20–49 (251) | 0.227 | 0.186 | +0.019, +0.061 |
| | 100–249 (731) | 0.203 | 0.147 | +0.044, +0.067 |
| | 1000+ (349) | 0.214 | 0.190 | +0.011, +0.039 |
| Rater generosity | harsh (985) | 0.176 | 0.140 | +0.028, +0.045 |
| | generous (984) | 0.229 | 0.173 | +0.046, +0.066 |
| Taste | niche (985) | 0.188 | 0.139 | +0.040, +0.057 |
| | mainstream (984) | 0.231 | 0.196 | +0.025, +0.046 |
| Share of low ratings | many "dropped" (984) | 0.173 | 0.131 | +0.034, +0.051 |
| Gap before next rating | <1 month (563) | 0.244 | 0.181 | +0.051, +0.076 |
| | 1–5 years (993) | 0.186 | 0.159 | +0.018, +0.036 |
| | **5+ years (147)** | **0.282** | **0.337** | **−0.088, −0.020** |

(All buckets in `results/list_model_error_by_user_group.csv`.)

- **Wins in every group except one.** The gain is largest for niche-taste users (+0.049)
  and mid-history users — the people a "most popular" list serves worst.
- **Loses for users returning after 5+ years** (147 users). Their old lists describe old
  taste, and what they watch on return is mostly what's popular *now*, which the baseline
  captures directly. A real site could weight recent list activity more heavily.
- **Popularity of recommended films:** 62% of the model's picks are 10k+-rating films vs 73%
  for the baseline; its hit rate on those blockbusters is higher (0.204 vs 0.153), i.e. it
  picks *better* popular films, not just more of them. It almost never recommends films with
  <1,000 training ratings (8 of 29,530 recs) — the popularity blend crowds them out.
- **By genre of recommended film** (`results/list_model_hits_by_genre.csv`): weakest hit rates
  for War (0.16), Romance (0.19), Animation (0.20); strongest for Sci-Fi and the main
  action/adventure genres.

## "More like this" (qualitative)

`results/list_model_results.json → more_like_this_examples`, e.g. Spirited Away → Princess
Mononoke, My Neighbor Totoro, Howl's Moving Castle…; Toy Story → Toy Story 2, The Lion King,
A Bug's Life, Monsters, Inc.… There is no ground truth for "similar", so this part is only
checked by eye.

## Limitations

- **Lists are simulated.** MovieLens has ratings, not lists: ≥3.5★ → Completed (liked),
  2.5–3★ → Completed (so-so), ≤2★ → Dropped. *Currently Watching* and *Plan to Watch* never
  occur in the data, so their weights (+0.8, +0.5) are design choices, not validated.
- **Popularity-heavy ranking.** The selected blend leans heavily on recent popularity; the
  model is best described as "popular films re-ranked by your taste". Long-tail films are
  rarely recommended. The taste-only variant (0.183, 2,696 distinct films) is the option if
  the group values discovery over raw precision.
- **Offline test favors popular films:** a good niche pick the user never happened to rate
  counts as a miss.
- **Experienced users only:** D2's scorable cohort (≥20 training ratings). A brand-new site
  user with an empty list gets the popularity fallback.
- **Validation, not test:** all numbers are on the validation period; the reserved test set
  is for D6.

*AI assistance: built with Claude Code, per the course AI policy.*
