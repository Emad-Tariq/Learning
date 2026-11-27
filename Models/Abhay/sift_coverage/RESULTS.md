# SIFT coverage: measured result

The experiment keeps Dalvin Model 3d’s baseline shortlist and final ranking rule. It adds up to two SIFT checks for each shortlisted identity with zero checked gallery photos, adds availability features, and retrains all three stages. No baseline artifacts or submissions are changed.

## How often was the alleged gap present?

Across 1,829 query appearances, 991 were known animals and 970 had the correct identity shortlisted. Only **5 of 970 (0.52%)** had no checked reference photo for the correct shortlisted identity. These were **3 distinct leopard photos**; no hyena query had this gap.

Additional verification found at least seven geometric inliers in 2 affected appearances, both the same distinct leopard photo. Four of the five appearances originally missed the top five; the other was ranked fifth. Among all 6,607 unchecked candidate rows, supplementation produced at least seven inliers in 18 rows: two correct and 16 incorrect candidate appearances. Seven inliers is therefore supporting evidence, not an identity decision.

## Complete three-stage comparison

| Metric | Baseline | Added verification + availability | Difference |
| --- | ---: | ---: | ---: |
| Overall MAP@5 | 0.934272 | 0.935147 | +0.000875 |
| Known MAP@5 | 0.897191 | 0.900320 | +0.003128 |
| Unknown MAP@5 | 0.978123 | 0.976333 | -0.001790 |

| Episode | Baseline | Intervention | Difference |
| --- | ---: | ---: | ---: |
| 22052 | 0.929718 | 0.930914 | +0.001196 |
| 22053 | 0.938067 | 0.938431 | +0.000364 |
| 22054 | 0.935255 | 0.936299 | +0.001043 |

| Species | Baseline | Intervention | Difference |
| --- | ---: | ---: | ---: |
| hyena | 0.963533 | 0.962947 | -0.000586 |
| leopard | 0.921058 | 0.922593 | +0.001534 |

The identity-cluster bootstrap 95% interval for the overall difference is [-0.002433, +0.004745]. 30 appearances improved and 32 worsened. The predeclared gate failed.

## Affected queries

| Episode | Photo | True identity | Extra best inliers | Baseline rank | Intervention rank |
| --- | --- | --- | ---: | ---: | ---: |
| 22053 | tr_06884 | LEO0257 | 3 | outside 5 | outside 5 |
| 22053 | tr_06660 | LEO0213 | 11 | outside 5 | 1 |
| 22054 | tr_06884 | LEO0257 | 3 | 5 | 3 |
| 22054 | tr_06660 | LEO0213 | 11 | outside 5 | 1 |
| 22054 | tr_00636 | LEO0428 | 0 | outside 5 | outside 5 |

## What this establishes

Keep Dalvin’s baseline as the default. This intervention did not meet the development gate.

The specific no-checked-photo gap is real but uncommon under this protocol. Its presence alone does not establish that missing SIFT caused a wrong prediction. The paired comparison tests the combined effect of extra checks, availability features and retraining; it does not isolate these effects. Even perfectly repairing only the five affected appearances, with all other predictions fixed, could add at most about 0.002624 MAP@5 in these episodes. This is not an upper bound for all possible SIFT improvements.

## Checks and limitations

- All baseline top-five predictions reproduce the saved checkpoint output exactly. Every saved reciprocal-rank score and pooled metric is independently recomputed from the prediction strings.
- Runtime assertions preserve candidate membership, query order, truth and non-SIFT candidate evidence. Two tests check sparse-zero provenance and label-independent pair selection.
- Validation identities were examined in earlier EDA/experiments. Episodes reuse photos and identities; these are development results, not a pristine final evaluation or leaderboard claim.
- The identity-disjoint roles have smaller galleries than original full-data competition inference. These measurements cannot establish the gap frequency or performance loss for that original model.
- Only previously unchecked candidates receive at most two checks. Partially checked candidates and all-photo verification were not tested.
- The bootstrap reflects these fixed models/identities and does not estimate retraining variance.

Reproduction and exact intervention: [README.md](README.md). Detailed results and per-query predictions are under `results/`.
