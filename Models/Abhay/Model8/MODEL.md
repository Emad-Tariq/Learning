# Model 8: Raw and LDA union shortlist

## Purpose and design motivation

EDA found weak retrieval when the correct animal lacked same-side references and when galleries were small. Raw and LDA similarities can retrieve different animals, suggesting that a correct identity might be lost before ranking. The experiment therefore targets candidate omission rather than immediately modifying novelty probabilities.

## Main model and pipeline

Retain the original 15 LDA/SIFT identities in order and append distinct raw-cosine top-15 identities. Up to 30 slots use validity masks; padded rows cannot train a model or receive ranking probability. Retrain each variant's complete three-stage LightGBM stack on matched episodes. Preserve PCA/LDA, mutual geometric verification, viewpoint evidence and the baseline novelty rule. Report candidate recall separately from final MAP@5, with missing-side and small-gallery slices.

## Evaluation and what was learned

MAP@5 rose from 0.934272 to 0.934910, but the gate failed. Only two query appearances newly gained their correct candidate, and neither reached the final top five. Missing-side and small-gallery candidate recall did not change. A larger shortlist alone did not establish a solution.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [union_eval.py](union_eval.py)
- [dalvin_core.py](dalvin_core.py)
- [RESULTS.md](RESULTS.md)
- [slice_results.csv](slice_results.csv)
- [diagnostics.json](diagnostics.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
