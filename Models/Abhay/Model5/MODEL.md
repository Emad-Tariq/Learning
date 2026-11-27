# Model 5: Open-Set Score Fusion (OSSF)

## Purpose and design motivation

Dalvin Model 3b already combined a new-animal detector with candidate scores using a fixed arithmetic rule. The design question was whether learning that final combination could place `new_individual` more effectively without changing known-identity retrieval. This is a baseline-driven motivation; the later 25 November EDA must not be presented as the original cause of this earlier experiment.

## Main model and pipeline

Normalize supplied MIEW-ID vectors and process species separately. Use PCA/shrinkage LDA, RootSIFT with geometric verification, a 15-identity shortlist, three LightGBM components and three sequential stacking stages from the adapted baseline. On separate calibration identities, fit median imputation, standard scaling and logistic regression (C=1) to detector/legacy probabilities, candidate margins, similarity, SIFT, viewpoint and gallery-support evidence. Rank the new-animal probability against conditional known-identity probabilities multiplied by the probability of being known. Keep known-identity order identical across the legacy-rule, detector-only and learned-fusion arms.

## Evaluation and what was learned

Identity-disjoint stage training, calibration and outer validation isolate the final fusion change. On 506 validation queries, baseline MAP@5 was 0.916206 and fusion was 0.911594. The small known-animal gain was outweighed by weaker new-animal detection. The learned fusion did not replace the baseline.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [train_eval.py](train_eval.py)
- [dalvin_core.py](dalvin_core.py)
- [Abhay_Local_PC.ipynb](Abhay_Local_PC.ipynb)
- [Abhay_OpenSet_Calibration.ipynb](Abhay_OpenSet_Calibration.ipynb)
- [RESULTS.md](RESULTS.md)
- [provenance.json](provenance.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
