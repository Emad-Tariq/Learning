# Model 14: Query-neighbour score sharing

## Purpose and design motivation

Model 12 provides a stronger reference stack, but a single sighting may still omit or underrank the correct identity. Repeated sightings can provide complementary evidence. A one-hop neighbour rule tests that opportunity while limiting error propagation through long chains.

## Main model and pipeline

Average Model 12's two conditional known-identity ranking heads, align scores by identity, and mix a query's evidence with the weighted mean of direct same-species neighbours. A neighbour can introduce an identity omitted from the original shortlist. Edges require cosine >=0.70 with >=20 geometric inliers, or cosine >=0.94 alone. Direct neighbours contribute half the final score. Select and freeze the recipe on development episodes, then compare on the separately trained seed-27613 larger-gallery checkpoint.

## Evaluation and what was learned

Confirmation MAP@5 increased from 0.89565 to 0.89711, with gains in all three episodes and both known/new groups. The paired interval [-0.00017, 0.00308] missed the promotion criterion. Top-one accuracy stayed 84.97%, and Model 12 remained the default under that gate.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [graph_model.py](graph_model.py)
- [run.py](run.py)
- [results/recipe.json](results/recipe.json)
- [results/validation_metrics.json](results/validation_metrics.json)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
