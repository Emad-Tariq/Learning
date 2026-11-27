# Model 15: Gallery-conditioned album matching and sparse-reference specialist

## Purpose and design motivation

Model 12 still struggled with known animals having few references. Model 14's small gain left room for complementary identity evidence. The EDA's sparse-identity and viewpoint findings, plus teammate graph/texture retrieval work, motivated judging an animal from an album of views and calibrating a reference photo against confusable animals.

## Main model and pipeline

Fit a gallery-conditioned pair verifier using raw similarity, geometric support and a reference photo's typical strongest wrong-animal matches by viewpoint. Aggregate identity support across broad viewpoint groups so nearly repeated views cannot dominate. Expand candidates using verifier scores, cohort-adjusted similarity and strong geometry. The broad arm blends learned known-identity scores at 30% while retaining Model 12's new-animal position. A later specialist retains Model 14 predictions except when its leading known identity has 1–3 references; then apply the album reranker at 30%, preserving new-animal placement.

## Evaluation and what was learned

The broad arm scored 0.89543 versus Model 12's 0.89565, although known-candidate recall increased from 94.86% to 95.81%. On three subsequent episode seeds, the specialist scored 0.89374 versus Model 14's 0.89346; the paired interval [-0.00027, 0.00093] included zero. This is a small targeted gain with unresolved uncertainty. The reported public score for the specialist was 0.84790, recorded in Model 16's history.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [album.py](album.py)
- [run.py](run.py)
- [specialist.py](specialist.py)
- [results/recipe.json](results/recipe.json)
- [results/specialist_recipe.json](results/specialist_recipe.json)
- [results/specialist_confirmation.json](results/specialist_confirmation.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
