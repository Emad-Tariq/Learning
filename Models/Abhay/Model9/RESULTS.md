# Model 9: coverage-aware ranking and unknown detection

The development-selected correction did not pass the validation gate. Keep Dalvin as the default.

Full-pipeline MAP@5: 0.934272 → 0.932905, difference -0.001367; descriptive 95% paired identity-cluster bootstrap interval [-0.003271, +0.000229].

The heads were fitted on previously unused calibration identities; the original three-stage stack is frozen. Strengths were selected using four identity-disjoint development folds before validation. Validation identities and the labelled dataset have been examined before, and episodes reuse photos; this is not an untouched holdout or a Kaggle leaderboard result. The experimental heads use additional supervised calibration examples, so this is a test of that addition, not an equal-supervised-data architecture comparison.

Development used 5,587 query appearances from 897 distinct photos and 109 query identities across 12 episodes. Ranking-only corrections at strengths 0.25, 0.5 and 1 improved the targeted development slices but reduced overall development MAP@5 by 0.000227 to 0.001008; they failed the selection constraint. The selected ranking strength is therefore zero, so the rank-only validation ablation is exactly the baseline. The selected unknown strength is 0.5; it is evaluated without further validation tuning.

Ranking strength selected on development: 0.0. Unknown strength: 0.5.

| Slice | Appearances | Unique photos | Dalvin | Rank only | Unknown only | Combined | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1829 | 1147 | 0.934272 | 0.934272 | 0.932905 | 0.932905 | -0.001367 |
| Known animals | 991 | 783 | 0.897191 | 0.897191 | 0.896687 | 0.896687 | -0.000505 |
| Unknown animals | 838 | 663 | 0.978123 | 0.978123 | 0.975736 | 0.975736 | -0.002387 |
| Missing side | 49 | 37 | 0.560204 | 0.560204 | 0.560204 | 0.560204 | +0.000000 |
| Small gallery (1–3 photos) | 134 | 100 | 0.691169 | 0.691169 | 0.691169 | 0.691169 | +0.000000 |
| Missing side + small gallery | 41 | 30 | 0.542683 | 0.542683 | 0.542683 | 0.542683 | +0.000000 |
| Same side available | 659 | 518 | 0.950405 | 0.950405 | 0.950405 | 0.950405 | +0.000000 |
| 4+ gallery photos | 857 | 683 | 0.929405 | 0.929405 | 0.928821 | 0.928821 | -0.000583 |

Both heads use observable candidate evidence at inference. The ranking correction is bounded to ±1 log-score; the unknown correction to ±log(2) log odds before blending. The shortlist and all baseline feature extraction, SIFT, three-stage stacking and final ranking rule remain unchanged.

The baseline exactly reproduces the prior run across all 1829 validation query appearances. Tests check correction bounds, preservation at zero strength and exclusion of truth columns from inference features.

Detailed configuration, development strength search, identity folds, fitted heads, source hashes and per-query predictions are retained in `local_runs/coverage_2052`. See `README.md` for the frozen selection procedure.