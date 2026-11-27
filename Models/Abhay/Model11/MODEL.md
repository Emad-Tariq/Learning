# Model 11: Full-vector MLP evidence for novelty detection

## Purpose and design motivation

Model 6's PCA pair verifier transferred poorly, while coverage corrections and difficult episodes did not establish a reliable gain. Model 11 revisits learned verification using full supplied descriptors, stronger regularization and identity-based stopping. Pair outputs enter a query-level calibrator because a pair match probability is not itself a probability that the animal is known.

## Main model and pipeline

Keep the baseline shortlist, known-identity ranking and three-stage stack. Train a 32/16 MLP on 4,305 symmetric inputs: full-vector absolute difference, elementwise product and cosine. Construct cross-group positives plus hard/random negatives; scale on pair-fitting data and clip at six. Use L2=1 and separate stopping identities. Aggregate pair scores across each candidate's references. On calibration identities, train logistic novelty calibrators with baseline summaries, with and without MLP evidence. The fixed primary arm averages baseline P(new) and MLP-calibrated P(new) equally.

## Evaluation and what was learned

Baseline MAP@5 was 0.934272; the MLP blend scored 0.932723. Known-animal ranking improved but new-animal performance fell. The summary-only control and full-replacement diagnostic help isolate calibration effects; the primary gate failed.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [verifier.py](verifier.py)
- [run.py](run.py)
- [persistence.py](persistence.py)
- [MLP_NewAnimal_Kaggle.ipynb](MLP_NewAnimal_Kaggle.ipynb)
- [Submission_Only_Kaggle.ipynb](Submission_Only_Kaggle.ipynb)
- [RESULTS.md](RESULTS.md)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
