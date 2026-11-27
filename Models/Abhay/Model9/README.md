# Model 9: coverage-aware residual ranking and unknown detection

This experiment adds bounded corrections to Dalvin Model 3d's final candidate probabilities and unknown score. It keeps the original 15-candidate shortlist, PCA/LDA, SIFT verification, viewpoint features, and three trained stacking stages unchanged. It uses the preserved baseline checkpoint, whose predictions were previously audited against a fresh training run. The residual heads are fitted using the previously unused calibration identity group. These corrections are experimental and do not replace the baseline automatically.

## Ranking correction

For each candidate, form 16 numerical features from gallery size, same-flank/exact-view coverage, geometric verification support, raw/LDA/centroid similarity and neighbour evidence. Express these relative to the baseline's highest-probability candidate. A regularized linear head adds `tanh(Xw)` to the log conditional identity probability, then normalizes the 15 candidate scores with softmax. The correction is bounded to ±1 log-score before blending.

Training minimizes weighted conditional cross-entropy on known queries whose correct identity is in the shortlist. Weights reduce dominance by repeatedly represented identities and upweight true missing-side/small-gallery cases. Those true coverage labels are used only during fitting; inference uses observable candidate evidence. Queries with an omitted correct candidate cannot provide a conditional ranking target and are excluded from this head's fit.

## Unknown correction

A second regularized linear head adds at most ±log(2) to the log odds of Dalvin's final unknown score. Inputs are baseline confidence, candidate probability entropy, raw candidate-score summaries, top-candidate and probability-weighted coverage/SIFT evidence, query-neighbour evidence, species and missing viewpoint. Query weights emphasize difficult-looking top candidates and reduce repeated-identity dominance, while preserving the total fitting weight of known versus unknown examples. The final known-vs-new ranking formula is unchanged.

Both heads use penalty 0.1, standardized/clipped numerical features and bounded corrections. Zero correction strength preserves the exact original probabilities.

## Development and evaluation

The original seed-2052 identity allocation supplies 330 stacking-training identities, 110 calibration/development identities and 110 validation identities. No head is fitted on a validation identity. The stack remains frozen. SIFT evidence is cached separately for development and validation roles.

Twelve fixed development episodes use seeds 42052–42063. Four identity-disjoint folds, stratified by species, produce out-of-fold residual corrections. Fit feature scales inside each fold. Select rank and unknown blending strengths from `{0, 0.25, 0.5, 1}` using only those out-of-fold predictions. The selection objective weights overall MAP@5 delta by 0.7, missing-side delta by 0.15 and small-gallery delta by 0.15. Eligible settings must not decrease development overall MAP@5 and may decrease unknown MAP@5 by at most 0.001. Ties prefer weaker corrections. A zero-strength baseline is included.

After selection, refit both heads on all development identities and freeze them. Compare baseline, rank-only, unknown-only and combined corrections on validation episodes 22052–22054. Ablation strengths are the selected strengths. If development selects zero strengths for both heads, the primary method remains the baseline; evaluate the fixed full-strength experimental arm only as a diagnostic negative-result check. Do not promote that diagnostic arm based on validation outcomes.

Report overall, known, unknown, missing-side, small-gallery and intersection results, counts of distinct queries/identities, per-episode results, and descriptive paired identity-cluster bootstrap intervals (3,000 draws, stratified by species). Missing side includes known left/right queries without a same-flank gallery photo of the correct identity. Small gallery means 1–3 true-identity gallery photos. The validation pass gate requires positive pooled and each-episode differences and no decline in known, unknown, missing-side or small-gallery MAP@5.

The validation identities and labelled dataset have already been examined in previous experiments and EDA. This is a controlled follow-up, not a pristine holdout or a leaderboard claim. Multiple episodes reuse identities and photos. Bootstrap intervals do not measure stack retraining variance. The calibration identity group provides new supervised examples to the experimental heads; this is part of the tested addition, not an equal-supervised-data architecture comparison. Galleries are smaller than final competition inference.

## Run

From the repository root:

```powershell
python Models/Abhay/Model9/test_coverage.py
python -u Models/Abhay/Model9/coverage_eval.py
```

The runner reads the supplied Downloads data, original Model 7 baseline checkpoint and completed Model 8 SIFT cache. It copies the cache into its own output directory, saves completed verifications, caches final-stage feature bundles, checks source/configuration consistency on resume and writes selection artifacts before creating validation features. No submission is generated.

`local_runs/coverage_2052` contains identity assignments, development fold assignments, strength search, fitted heads, source hashes, configuration, per-query predictions and metrics. Baseline predictions must exactly match the prior experiment across all validation episodes.

## Attribution

The unchanged baseline and episode helpers come from Dalvin Model 3d as preserved in `Model7/dalvin_core.py` and `Model7/data_protocol.py`. The residual heads, identity-level development selection and diagnostics are new experiment code. No external data or image models are used.
