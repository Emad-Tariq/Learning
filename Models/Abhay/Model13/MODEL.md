# Model 13: Joint ranking, regularized heads and PLDA evidence

## Purpose and design motivation

Model 12 improved enlarged-gallery validation substantially, but known-leopard matching and small-gallery cases remained difficult. Independently trained detector/candidate heads and pairwise maximum evidence might still leave a mismatch with the final identity-plus-new ranking task. This motivated testing objective alignment, stronger regularization and covariance-based identity evidence.

## Main model and pipeline

Use Model 12 episode features and earlier-stage predictions. Compare a grouped joint identity-plus-new ranking objective, regularized replacements for the final baseline heads, and an all-candidate scoring control. PLDA enrichment adds numerical within/between-identity evidence while preserving original rows, candidates and baseline fields. Weight repeated photo appearances and record bounded development searches and blend strengths. Lock the selected recipes before preparing a separately trained seed-27613 identity evaluation; reuse geometric pair measurements rather than fitted weights across the split.

## Evaluation and what was learned

This folder contains multiple experimental arms rather than one interchangeable score. Development metrics/recipes and the fresh-split preparation are retained in local_runs. A consolidated final comparison is not present among the saved JSON summaries inspected for this documentation; no winning score is inferred. The fresh checkpoint also supports later paired comparisons.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [experiment.py](experiment.py)
- [compare.py](compare.py)
- [joint_ranker.py](joint_ranker.py)
- [regularized_head.py](regularized_head.py)
- [plda_features.py](plda_features.py)
- [enrich_plda.py](enrich_plda.py)
- [verify.py](verify.py)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
