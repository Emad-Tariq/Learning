# Model 6: Pairwise Identity Verification (PIV)

## Purpose and design motivation

Model 5 changed novelty placement but retained the baseline's known-animal matcher. Model 6 isolated a different question: can a small learned pair verifier improve identity matching before it is added to an open-set system? Separating this matching gate avoids confusing improved pair classification with improved final open-set ranking.

## Main model and pipeline

Split identities into 60% fit, 20% development and 20% evaluation before constructing pairs. Fit 128-dimensional PCA on fit images, normalize projections, and train an MLP with 64/32 hidden units and L2=0.01. Symmetric pair inputs combine absolute differences, elementwise products, raw/projected cosine and viewpoint indicators. Positives cross near-duplicate groups; negatives combine hard same-species neighbours and random examples. Rank each identity by its strongest gallery pair. Compare with raw cosine and LDA on identical closed-set gallery/query episodes.

## Evaluation and what was learned

Evaluation MAP@5 was 0.3046 for PIV versus 0.9091 for raw cosine. Correct and strongest incorrect identities both often received extreme probabilities. This suggests poor transfer and a mismatch between pair training and maximum-over-gallery retrieval; it does not prove one unique cause. The matching gate failed.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [matching.py](matching.py)
- [PIV_Local.ipynb](PIV_Local.ipynb)
- [PIV_Kaggle.ipynb](PIV_Kaggle.ipynb)
- [RESULTS.md](RESULTS.md)
- [FOLLOWUP_EXPERIMENTS.md](FOLLOWUP_EXPERIMENTS.md)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
