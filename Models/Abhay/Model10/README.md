# Model 10: retraining with difficult episodes

This experiment retrains Dalvin Model 3d's complete three-stage pipeline with 0%, 25% and 50% difficult training episodes. The original PCA/LDA, 15-candidate LDA/SIFT shortlist, mutual RootSIFT verification, viewpoint features, all three LightGBM models per stage, neighbour stacking and final unknown-score rule are unchanged. No nuisance projection or residual correction head is used.

## Matched training

All three methods use the seed-2052 identity allocation and 12 episodes per stage, with 400-tree candidate scorers and a 300-tree unknown detector. There are three disjoint stage-training identity groups, one development identity group and one validation identity group. Every method's previous-stage predictions supply its own subsequent stacking features. The baseline is retrained and checked against the preserved checkpoint's validation predictions.

For each ordinary episode, the unchanged baseline splitter selects known/unknown identities and separates near-duplicate groups. The difficult version is constructed from that episode:

1. Select half of the gallery identities within each species.
2. For each selected identity, retain 1–3 gallery photos. If known left/right queries and differently oriented gallery photos exist, remove one queried flank from that identity's gallery before selecting the retained photos. The identity always retains at least one gallery photo.
3. Keep all original known queries unchanged. Gallery photos are only removed, never moved into query, so a duplicate group cannot newly cross gallery/query.
4. Keep each unknown identity entirely absent. Retain exactly the same number of unknown queries per identity, but choose the supplied photos with highest original-feature cosine to the reduced same-species gallery. This makes unknown examples at least as convincing under that similarity measure as a random subset of the same quota. No new identities or external data are introduced.

The known/unknown query counts are identical to the corresponding ordinary episode, including species-specific counts. Gallery identity sets are identical. Difficult episodes can include unmodified gallery identities and already-small galleries; realized case counts and gallery changes are exported.

The 25% method replaces exactly 3 of 12 episodes per stage; the 50% method replaces exactly 6. The former's difficult episode positions are a subset of the latter's. Both methods use the identical difficult episode at shared positions. Positions rotate by stage and are fixed in `config.json` before fitting. Total training episodes and query label counts stay matched.

## Selection and validation

Six fixed ordinary development episodes use seeds 52052–52057 on the original calibration identity role. Choose the method with highest development MAP@5 among settings with no overall loss, no missing-side/small-gallery decline and at most 0.001 unknown-query decline. Include baseline as fallback. Freeze this choice before scoring validation.

All three methods are reported on ordinary validation episodes 22052–22054, enabling direct comparison with earlier experiments. Those episodes are not artificially made difficult. Report overall, known, unknown, missing-side, small-gallery (1–3 true-identity gallery photos), their intersection and same-side/larger-gallery results. Missing side applies only to known left/right queries with no same-flank photo of the correct identity in the gallery.

The gate requires a positive pooled and each-episode delta and no known/unknown/missing-side/small-gallery decline. Report 3,000-draw paired identity-cluster bootstrap intervals, stratified by species, separately for both experimental fractions. Selection uses development results only; the better validation fraction is not retroactively declared development-selected.

The labelled dataset and validation identities were previously examined in EDA and other experiments. This is a controlled follow-up on one outer identity allocation, not an untouched dataset, evidence of retraining robustness across independent outer splits, or a Kaggle leaderboard claim. Episodes reuse photos and identities. A positive point estimate must be distinguished from a reliable improvement.

## Run and resume

From the repository root:

```powershell
python Models/Abhay/Model10/test_difficult.py
python -u Models/Abhay/Model10/difficult_eval.py
```

Data defaults to `~/Downloads/reid_student_data_main`, with supplied SIFT directories nearby. The completed Model 9 SIFT cache is copied into this run's own directory. Shared role-local SIFT evidence is used for every method. Completed stages are checkpointed atomically, source/configuration mismatches refuse resume, and ordinary/difficult episode manifests plus method schedules are exported. No competition submission is generated or submitted.

`local_runs/difficult_2052` stores configuration, source hashes, environment, models, role allocations, training distributions and split manifests, development selection and query-level results. The default output can be overridden with `--output`.

## Attribution

The unchanged core and data helpers are imported from `Model7/dalvin_core.py` and `Model7/data_protocol.py`, adapted from Dalvin Model 3d. New work is the controlled difficult-episode sampler, three-way retraining runner, development selection and comparative diagnostics. Tests verify exact episode proportions, nested schedules, label/identity conservation, deterministic sampling and unknown-example similarity.

## Separately labelled coverage-input follow-up

The requested episode-only run completed first. It did not improve the difficult validation slices. A separate exploratory follow-up adds 23 observable candidate-coverage features directly to the trained LightGBM unknown detector in the two difficult-episode variants. The baseline is still unchanged. The added features describe photo count, same-side/exact-view coverage, SIFT support fraction and weak-SIFT/missing-side interactions for the leading LDA, raw and SIFT candidates, plus top-three coverage fractions. No true identity, query known/unknown label or true-identity coverage is supplied at inference.

Candidate scorers do not receive these new fields directly; they retain the original feature lists. The unknown detector is trained with the new fields from stage one, and all three stages of each experimental pipeline are retrained so later stacking sees its own preceding predictions. This is a change to the actual tree detector, rather than the post-hoc linear correction used by Model 9.

The follow-up keeps the same fixed fractions, data roles, episode counts, sampler, development rule and validation comparisons. Its results are exploratory: the intervention was proposed after inspecting the episode-only outcome. It is saved separately and must not be described as the pure difficult-episode result or an untouched confirmatory test.

```powershell
python Models/Abhay/Model10/test_coverage_features.py
python -u Models/Abhay/Model10/coverage_train.py
```

`coverage_train.py` is a standalone source snapshot derived from `difficult_eval.py`; `make_coverage_runner.py` documents the exact source differences. The follow-up's output is `local_runs/coverage_difficult_2052`. `coverage_features.py` contains the new feature extraction and detector-specific fitting. Feature tests check truth invariance, the exact field count and removal of all coverage fields from legacy query feature lists.
