# Model 9: Coverage-aware residual scoring

## Purpose and design motivation

The EDA linked missing-side/small-gallery coverage with poor matching, while Model 8 recovered almost no useful omitted candidates. This shifted the hypothesis toward interpretation of existing candidate evidence: a weak match might reflect inadequate references rather than a new animal. A bounded correction limits damage to an already strong baseline.

## Main model and pipeline

Freeze the original 15-candidate stack. Fit a regularized linear ranking head on candidate-relative coverage, SIFT, similarity and neighbour features; add a bounded tanh correction to log conditional identity scores. Fit a separate head that adds a bounded correction to novelty log odds. Use separate calibration identities and four identity-disjoint development folds to select blending strengths. Include zero correction as the baseline; use only observable candidate evidence at prediction time.

## Evaluation and what was learned

Development selected zero ranking correction and 0.5 novelty correction. Validation MAP@5 fell from 0.934272 to 0.932905. Neither the overall gate nor the intended coverage improvement was established; the frozen baseline remains preferred.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [coverage_heads.py](coverage_heads.py)
- [coverage_eval.py](coverage_eval.py)
- [development_strength_search.csv](development_strength_search.csv)
- [RESULTS.md](RESULTS.md)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
