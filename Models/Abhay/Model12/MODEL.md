# Model 12: Larger-gallery retraining

## Purpose and design motivation

Earlier experiments used identity-disjoint stages with relatively small galleries. Adding reference animals changes impostor competition, candidate margins and novelty evidence even when the query is unchanged. The larger-gallery diagnostic exposed a weakness in the original stack under expanded reference context. This motivated matching training conditions more closely to that context rather than adding another score correction.

## Main model and pipeline

Keep the original 36 training query episodes, truths, own-identity references, model settings and three stacking stages. Enlarge gallery context with the dedicated background identity role and completed earlier-stage identities. Background animals are references, never stage queries. Future-stage and validation identities are excluded from fitting context. Retrain all three baseline heads in each stage and compare the frozen original and retrained models on identical enlarged-gallery validation episodes. Package portable inference assets and an executable Kaggle notebook.

## Evaluation and what was learned

On the matched larger-gallery evaluation, MAP@5 improved from 0.878358 to 0.907654; all three episodes and pooled known/new groups improved. The paired animal-cluster interval for the gain was [0.016626, 0.046145]. The gate passed on this allocation. Known-leopard performance slipped slightly, so the result is not a universal matching fix or a leaderboard score.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [run.py](run.py)
- [interpretation.json](interpretation.json)
- [report.html](report.html)
- [submission.py](submission.py)
- [Model12_LargerGallery_Submission_Kaggle.ipynb](Model12_LargerGallery_Submission_Kaggle.ipynb)
- [verify.py](verify.py)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
