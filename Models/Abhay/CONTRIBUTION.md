# Abhay — project contributions

## Exploratory data analysis and research

- Built reproducible EDA scripts, CSV diagnostics, plots and an HTML report for identity frequency, sparse reference albums, viewpoints, near-duplicate groups, supplied embeddings and SIFT support.
- Compared raw and PCA/LDA retrieval, photo-weighted and identity-weighted results, same-side availability, small-gallery cases and simulated known/new score overlap. Identified the need to separate candidate omission, identity ranking and novelty placement.
- Investigated hubness corrections, within-identity nuisance projections, gallery calibration and geometric verification coverage. Retained negative results and uncertainty rather than claiming every hypothesis improved performance.

## Models and implementations

- **Model 5 — Open-Set Score Fusion (OSSF):** Normalize supplied MIEW-ID vectors and process species separately. See [design and evidence](Model5/MODEL.md).
- **Model 6 — Pairwise Identity Verification (PIV):** Split identities into 60% fit, 20% development and 20% evaluation before constructing pairs. See [design and evidence](Model6/MODEL.md).
- **Model 7 — Nuisance Evidence Retrieval (NER):** Within each fitting identity, subtract its mean embedding and weight residuals by 1/sqrt(n-1), giving identities equal covariance weight. See [design and evidence](Model7/MODEL.md).
- **Model 8 — Raw and LDA union shortlist:** Retain the original 15 LDA/SIFT identities in order and append distinct raw-cosine top-15 identities. See [design and evidence](Model8/MODEL.md).
- **Model 9 — Coverage-aware residual scoring:** Freeze the original 15-candidate stack. See [design and evidence](Model9/MODEL.md).
- **Model 10 — Difficult-episode retraining and coverage-input follow-up:** Retrain the baseline with 0%, 25% or 50% difficult episodes. See [design and evidence](Model10/MODEL.md).
- **Model 11 — Full-vector MLP evidence for novelty detection:** Keep the baseline shortlist, known-identity ranking and three-stage stack. See [design and evidence](Model11/MODEL.md).
- **Model 12 — Larger-gallery retraining:** Keep the original 36 training query episodes, truths, own-identity references, model settings and three stacking stages. See [design and evidence](Model12/MODEL.md).
- **Model 13 — Joint ranking, regularized heads and PLDA evidence:** Use Model 12 episode features and earlier-stage predictions. See [design and evidence](Model13/MODEL.md).
- **Model 14 — Query-neighbour score sharing:** Average Model 12's two conditional known-identity ranking heads, align scores by identity, and mix a query's evidence with the weighted mean of direct same-species neighbours. See [design and evidence](Model14/MODEL.md).
- **Model 15 — Gallery-conditioned album matching and sparse-reference specialist:** Fit a gallery-conditioned pair verifier using raw similarity, geometric support and a reference photo's typical strongest wrong-animal matches by viewpoint. See [design and evidence](Model15/MODEL.md).
- **Model 16 — Independent joint known/new scorer:** Fit species-specific gallery PCA with partial whitening. See [design and evidence](Model16/MODEL.md).
- **Model 17 — Gallery-trained metric graph and learned sighting links:** Fit 384-dimensional species PCA, then construct gallery-label discriminant coordinates with two covariance shrinkage settings and ridge identity scores. See [design and evidence](Model17/MODEL.md).

## Evaluation and reproducibility

- Implemented identity-disjoint fitting, calibration and evaluation roles; approximate encounter grouping; matched baseline comparisons; controlled ablations; and explicit development selection/promotion criteria where specified by each experiment.
- Added per-query/species/coverage diagnostics, candidate recall, known/new breakdowns, paired animal-cluster uncertainty estimates, checkpoint persistence, source/feature fingerprints and independent output audits across the model series.
- Built local and Kaggle notebooks, portable source bundles and inference helpers, and checked prediction file IDs, ordering, distinct labels and species consistency.
- Organized Models 5–17 into numbered folders and documented each pipeline, rationale, evidence and limitations. Model 5 moved from the Abhay root into `Model5/`.

## Results and attribution

Model 5 fusion, Model 6 pair verification and Models 8–11 did not establish an overall gain over their paired controls. Model 7's small gain missed its gate. Model 12 improved paired larger-gallery validation from 0.878358 to 0.907654. Models 14–15 produced small local gains with uncertainty including zero; Models 16–17 did not beat the specialist in their paired confirmation. Model 13 contains multiple investigated heads; the saved artifacts should be consulted rather than assuming a single winning result. Scores across different protocols are not a common leaderboard.

Dalvin retains authorship of the baseline retrieval, RootSIFT/PCA/LDA and LightGBM stacking components reused in these experiments. Abhay's contributions are the adaptations and experiments listed above. Emad's referenced graph/texture work and shared simulator remain credited to Emad. This contribution record does not assign teammate implementations to Abhay.
