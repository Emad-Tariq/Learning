# Model 17: Gallery-trained metric graph and learned sighting links

## Purpose and design motivation

Model 16's independent scorer did not close the gap to the specialist. Earlier experiments showed that geometry, reference coverage and repeated sightings provide complementary signals, while fixed neighbour thresholds and ordinary cosine alone can be brittle. Model 17 combines adaptive gallery metrics, graph evidence and learned same-animal links. This is a synthesis motivated by prior results, not evidence that every added component independently helps.

## Main model and pipeline

Fit 384-dimensional species PCA, then construct gallery-label discriminant coordinates with two covariance shrinkage settings and ridge identity scores. Combine raw/metric similarities, geometric evidence, centroids, reciprocal-neighbour overlap and graph evidence in candidate features. Simulate episodes with varied reference coverage and query-pool sizes. Blend grouped conditional ranker and binary classifier log probabilities with weights 0.54/0.46. A learned sighting-link model uses similarity, geometric evidence, neighbour-relative margins and viewpoint to pool posteriors across likely repeat sightings. Preserve the teammate simulator's identity allocation for its benchmark; geometry and confirmation have dedicated audit scripts.

## Evaluation and what was learned

Keep the saved small/large-gallery and linked/unlinked arms separate. The specialist confirmation covers 1,402 appearances from 95 animals: Model 17 MAP@5 0.890145 versus 0.893735 for the specialist, delta -0.003590 with interval [-0.013635, 0.006340]. It did not establish superiority. Other saved benchmark values use different episodes and are not directly interchangeable.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [graph_model.py](graph_model.py)
- [experiment.py](experiment.py)
- [sightings.py](sightings.py)
- [protocol.py](protocol.py)
- [geometry.py](geometry.py)
- [confirmation.py](confirmation.py)
- [verify.py](verify.py)
- [results/confirmation.json](results/confirmation.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
