# Model 7: Nuisance Evidence Retrieval (NER)

## Purpose and design motivation

After the Model 6 failure, the retrieval research explored simpler transformations rather than another high-dimensional pair network. Within-identity changes can reflect pose, lighting and viewpoint rather than identity. Residual-projection research motivated attenuating those directions while retaining the established open-set pipeline. This motivation predates the later EDA report.

## Main model and pipeline

Within each fitting identity, subtract its mean embedding and weight residuals by 1/sqrt(n-1), giving identities equal covariance weight. Fit species-specific residual directions, estimate 64 and retain the leading 16; subtract half the component along those directions and renormalize. Replace only raw similarity evidence in Dalvin Model 3d. Preserve original PCA/LDA retrieval, candidate identities, mutual RootSIFT/RANSAC, unknown detector, candidate scorers and three stacking stages. Both control and transformed variants are retrained.

## Evaluation and what was learned

Separate identities fit the transformation, the three ranking stages and validation. Frozen projection strength is 0.5. MAP@5 increased from 0.934272 to 0.935721, but one of three episodes worsened and the uncertainty interval included zero. The submission gate failed.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [paired_eval.py](paired_eval.py)
- [data_protocol.py](data_protocol.py)
- [dalvin_core.py](dalvin_core.py)
- [NER_Local.ipynb](NER_Local.ipynb)
- [NER_Kaggle.ipynb](NER_Kaggle.ipynb)
- [RESULTS.md](RESULTS.md)
- [results/paired_metrics.json](results/paired_metrics.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
