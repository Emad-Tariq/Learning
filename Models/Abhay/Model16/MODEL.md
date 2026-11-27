# Model 16: Independent joint known/new scorer

## Purpose and design motivation

The Model 15 specialist preserved Model 14's first predictions, limiting how much a lower-rank correction could change the final outcome. Its reported public result also left a gap to the teammate comparator. Model 16 therefore trains a fresh known/new scorer, with a separate hypothesis that raw maxima can be distorted by the size of the comparison population.

## Main model and pipeline

Fit species-specific gallery PCA with partial whitening. Generate identity candidates using raw/projected similarity and geometric matches. Add same-flank support and optional two-reference convex reconstruction residuals. Train a LightGBM model with per-query multinomial loss over candidate identities plus new_individual and a diagonal Hessian majorizer. Include indirect cohort evidence without borrowing another sighting's predicted identity. A second arm replaces raw population maxima with exact expected maxima under uniform sampling of up to 128 photos, plus match rates. Select recipes on development before confirmation.

## Evaluation and what was learned

On the identical confirmation episodes, raw-population and fixed-exposure arms scored 0.86827 and 0.86883 versus 0.89374 for the Model 15 specialist. Fixed exposure helped novel animals while weakening known-animal matching. Both variants remained below the paired comparator; neither established a replacement.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [run.py](run.py)
- [reconstruction.py](reconstruction.py)
- [balanced.py](balanced.py)
- [audit.py](audit.py)
- [results/recipe.json](results/recipe.json)
- [results/confirmation_metrics.json](results/confirmation_metrics.json)
- [results/balanced_confirmation.json](results/balanced_confirmation.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
