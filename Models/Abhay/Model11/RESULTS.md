# Model 11: full-vector MLP new-animal detector

**The local experiment did not improve overall MAP@5. Keep the baseline.**

The fixed primary arm blends baseline P(new) equally with a logistic calibrator that receives full-vector pair-MLP evidence and baseline context. A summary-only calibrator provides the same baseline context without the MLP evidence. Full replacement is diagnostic; no arm or blend was selected using validation outcomes.

| Method | Overall MAP@5 | Known MAP@5 | New MAP@5 | New top-one detection |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.934272 | 0.897191 | 0.978123 | 95.70% |
| summary_blend | 0.932176 | 0.899378 | 0.970963 | 94.27% |
| mlp_blend | 0.932723 | 0.904087 | 0.966587 | 93.44% |
| mlp_replacement | 0.927529 | 0.904928 | 0.954256 | 91.17% |

Primary overall difference: -0.001549. Identity-cluster bootstrap 95% interval: [-0.0046694689501059735, 0.0013030443418964896]. Promotion gate passed: False.

Known-animal performance improved, but the loss on truly new animals was larger. The MLP arm beat the summary-only calibration control by just 0.000547 MAP@5, which does not establish a reliable incremental benefit. No settings were changed after inspecting these validation results.

The network has 138,337 parameters, trained on 9,852 distinct sampled pairs. Early stopping used 1,999 pairs from disjoint identities and selected epoch 3; six later epochs failed to improve stopping loss. Pair probabilities are not interpreted as query P(known). Gallery size and pair aggregation enter a separate query-level calibrator.

Validation contains 1,829 query appearances from 1,147 distinct photos and 105 identities across three episodes. Baseline predictions exactly reproduce the preserved checkpoint. Independent checks verify identity separation and recompute all final rankings, reciprocal ranks, class-specific MAP@5 and new-animal top-one rates.

## Scope

This is a local completed run using the preserved Dalvin baseline checkpoint at seed 2052 and its verified matching cache. The Kaggle notebook trains that baseline from scratch from embedded preserved sources, then executes the same verifier and calibration protocol. Exact numeric results can vary with package versions; local versions are in `independent_audit.json`. Kaggle execution itself has not been performed here.

The existing evaluation allocation was previously inspected in EDA and other experiments. These are development results, not an untouched test or a leaderboard result. Gallery sizes differ from full-data competition inference. Episodes reuse photos/identities, and the bootstrap does not estimate retraining variance.

## Files

- `MLP_NewAnimal_Kaggle.ipynb`: self-contained notebook, including source snapshots and tests.
- `README.md`: protocol and execution instructions.
- `local_runs/main/validation_predictions.csv`: per-query predictions for all four arms.
- `local_runs/main/metrics.json`: overall, species and episode metrics and uncertainty.
- `local_runs/main/independent_audit.json`: independently verified counts and environment.
