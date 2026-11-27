# Model 5: Open-Set Score Fusion (OSSF) â€” validation analysis

Recorded 24 November 2026 from the completed notebook output. This is an open-set validation result, not a public/private Kaggle leaderboard result. The exact run environment and notebook version were not supplied with the pasted table; `results/reported_metrics.json` preserves that provenance and the displayed precision. Repository defaults are a reproducibility reference, not a verified record of this execution.

## Comparison

| Method | Overall MAP@5 | Known MAP@5 | Unknown MAP@5 | Known incorrectly new at rank 1 | Unknown correctly new at rank 1 |
|---|---:|---:|---:|---:|---:|
| Dalvin rule | **0.916206** | 0.887106 | **0.980892** | 0.128940 | **0.961783** |
| Detector only | 0.909947 | 0.889971 | 0.954352 | 0.114613 | 0.910828 |
| Model 5 / learned fusion | 0.911594 | **0.890449** | 0.958599 | **0.106017** | 0.923567 |

Evaluation contains 506 queries: 349 known and 157 unknown (31.0277%). Candidate recall for known queries is 0.9799426934, equivalent to 342 of 349 correct identities present in the shortlist. All three methods use the same stack and candidate identity ordering. Differences arise from the position of `new_individual`.

Fusion minus baseline MAP@5 is **-0.0046113306982873414**, or about **-0.4611 percentage points**. The exact delta was printed by the notebook; table metrics are rounded to six decimals. Known-query MAP@5 increases by about 0.003343 while unknown-query MAP@5 falls by about 0.022293. Known-animal queries incorrectly assigned new at rank one decrease from 45 to 37. Unknown queries correctly assigned new at rank one decrease from 151 to 145. A reduction in known-new errors is not necessarily an increase in correct identity top-one predictions.

## Species analysis

| Species / query type | n | Dalvin rule | Detector only | Model 5 |
|---|---:|---:|---:|---:|
| Hyena / known | 158 | 0.903059 | **0.906224** | 0.900949 |
| Hyena / unknown | 79 | **0.974684** | 0.940928 | 0.959916 |
| Leopard / known | 191 | 0.873909 | 0.876527 | **0.881763** |
| Leopard / unknown | 78 | **0.987179** | 0.967949 | 0.957265 |

Known-leopard MAP@5 improves, whereas known-hyena MAP@5 decreases slightly. Unknown performance decreases for both species. Several viewpoint slices contain only 1-6 queries, so they do not support reliable viewpoint-specific tuning.

## Interpretation

The hypothesis was that learning the final combination of unknown evidence would improve ranking over the fixed average. This run does not support that hypothesis overall. Fusion reduces some false unknown decisions for known animals but gives up more reciprocal-rank credit on truly unseen animals. The legacy heuristic's lack of a direct probabilistic interpretation does not imply that it is ineffective for MAP@5.

The coefficient table shows strong positive weights for the existing detector and legacy unknown score; correlated inputs prevent interpreting coefficient magnitudes as causal importance. Logistic loss optimizes binary likelihood rather than MAP@5, and a pooled linear fusion model may not generalize across species, gallery size, or unknown prevalence. These are possible explanations, not established causes from this table alone.

The identity-disjoint sequential-stage protocol also differs from Dalvin's original full-data simulation stacking. Its 0.916206 baseline cannot be compared directly with the logged 0.85078 public leaderboard result. No significance claim or leaderboard gain is made from a single split.

## Decision and follow-up

Dalvin's rule remains the strongest method on this evaluation. Model 5 is retained as a documented ablation and negative result. A subsequent experiment could constrain changes to the baseline probability through conservative blending or a simpler calibration model. Blend weights and hyperparameters would be selected using calibration/development data only. This examined validation set should not be repeatedly optimized against and described as untouched; a new reserved outer evaluation is required for a final improvement claim.
