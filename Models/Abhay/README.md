# Model 5: Open-Set Score Fusion (OSSF)

Author contribution: Abhay. Baseline: Dalvin Model 3b (`Models/Dalvin/reid_pipeline.py`).

OSSF learns a logistic combination of existing unknown-animal evidence for the **final placement of `new_individual`**. Retrieval, conditional known-identity ranking and earlier stacking rounds are shared with the baseline. The comparison includes Dalvin's legacy rule and the existing detector alone.

The completed validation run did **not** improve overall MAP@5: baseline **0.916206**, detector-only **0.909947**, OSSF **0.911594**, delta **-0.0046113306982873414**. Fusion improves known-animal MAP@5 slightly but loses more on unseen animals. Dalvin's rule remains the strongest method on this split. Full analysis: [RESULTS.md](RESULTS.md). These are validation results, not leaderboard results.

## Method

1. Normalize supplied MIEW-ID descriptors; process each species separately.
2. Apply PCA (up to 384 dimensions) and shrinkage LDA fitted using episode gallery labels. PCA also sees unlabelled episode queries.
3. Compute raw/projected similarities, per-identity maximum and top-two scores, centroid similarity, viewpoint evidence and query-neighbour features.
4. Verify selected same-species neighbours using RootSIFT, ratio testing and partial-affine RANSAC. Retrieve 15 candidate identities using LDA similarity and SIFT evidence.
5. Train Dalvin's unknown detector, known-query candidate scorer and all-query candidate scorer. Apply three sequential stacking stages using neighbour predictions.
6. Fit median imputation, standardization and logistic regression (`C=1`) on a separate calibration identity group. Inputs include detector and legacy probabilities, candidate-score summaries, margins, SIFT support, viewpoint availability, gallery support and species.
7. Rank `new_individual` with score `p_new` against known candidates with scores `(1-p_new)*P(identity|known)`. Known identity order is shared across methods, with deterministic tie handling.

The legacy probability is `(detector_probability + clip(1-sum(binary_candidate_scores), 0, 1))/2`. OSSF tests whether learning this combination improves MAP@5. Logistic loss optimizes binary likelihood; a learned fusion is not guaranteed to improve calibration or ranking. No new image features, raw-image processing or pretrained models are introduced.

## Evaluation protocol

Identities are shuffled per species with seed 2026. Calibration receives 20%, outer validation 20%, and stack training 60%. The training identities are divided into three non-overlapping stage groups. Stage 1 models produce features for stage 2 identities; stages 1-2 produce features for stage 3 identities. Models from earlier stages have not trained on later-stage identities. This replaces the baseline's simulation-level folds to reduce repeated-identity overlap between supervised stages and final evaluation.

Within each episode, 20-35% of identities are removed from the gallery and labelled `new_individual` for evaluation. Remaining identities supply gallery and known-query sightings where possible. Near-duplicate connected components are held out whole; selected components are never truncated. Singletons and identities with one component may remain gallery-only. Training mixes four duplicate-grouping regimes. Calibration and outer evaluation use cosine >0.95 or >=100 verified SIFT inliers. Unverified pairs limit duplicate detection, so encounter separation is approximate.

The base stack is frozen before calibration. Repeated calibration sightings receive inverse occurrence weights so each image has total weight one. Fusion is frozen before outer validation. All three methods then use identical validation queries and trained stack outputs. Gallery labels are available to episode retrieval/LDA, as in the competition setup; query truth is excluded from predictor features. Competition test queries are excluded from the validation pipeline.

Default configuration: three rounds, 12 simulations per stage, four calibration episodes, 150 SIFT neighbours, 15 candidates, 400 candidate-scorer trees and up to 300 detector trees, two CPU workers. The pasted result did not include the complete runtime configuration or library versions; these defaults are not asserted as verified run metadata.

This protocol uses fewer training identities and smaller galleries per stage than the original full-data baseline. Its scores do not reproduce the team's public score or original simulated validation score. Gallery size and unknown prevalence can shift between calibration, evaluation and test. A single 506-query split does not establish statistical superiority. The reported evaluation has now been examined; further tuning requires calibration/development data and a new reserved outer evaluation for final claims.

## Files

| File | Purpose |
|---|---|
| `train_eval.py` | Training, final-stage fusion, controlled evaluation and optional submissions |
| `dalvin_core.py` | Adapted baseline components; attribution retained |
| `Abhay_OpenSet_Calibration.ipynb` | Self-contained Kaggle notebook |
| `Abhay_Local_PC.ipynb` | Self-contained Windows notebook |
| `test_pipeline.py` | Synthetic integration and identity-separation tests |
| `requirements.txt` | Runtime dependencies |
| `provenance.json` | Baseline source path, source hash and adaptation list |
| `RESULTS.md`, `results/` | Measured validation result and analysis |
| `CONTRIBUTION.md` | Contribution record and verification |
| `GIT_WORKFLOW.md` | Review, commit and push commands |

Notebook titles and embedded source identify Model 5. Internal method keys (`dalvin_rule`, `detector_only`, `abhay_fusion`) remain stable for compatibility with existing results. Notebook outputs and execution counters are cleared in the repository versions; recorded metrics are stored separately.

## Dataset

Competition: https://www.kaggle.com/competitions/projectwild3/data

```
<dataset parent>/
  reid_student_data_main/
    train.csv
    test.csv
    miewid_ids.csv
    miewid_features.npy
    sample_submission.csv
  sift_features_train.npz/
  sift_features_test.npz/
```

SIFT sources may be compressed NPZ archives or extracted directories containing `*_desc.npy` and `*_xy.npy` arrays. The runner searches the metadata directory and its two parent levels. Extracted nested NPY files are supported. Original image folders are unused. Global features are aligned through `miewid_ids.csv`, rather than assumed to follow concatenated CSV order.

## Kaggle execution

Import `Abhay_OpenSet_Calibration.ipynb`, attach the existing `projectwild3` competition input, and execute cells in order. The metadata directory is detected automatically when exactly one complete match exists. `DATA` supports an explicit override. Embedded scripts and outputs are written under `/kaggle/working`; input files remain read-only. A separate code or private data upload is unnecessary.

The optional install cell targets the active kernel. The optional reduced execution check writes separate artifacts and is not a quality benchmark. Full training uses CPU libraries; selecting a GPU does not automatically accelerate SIFT, PCA/LDA or the configured LightGBM models. Runtime has not been systematically benchmarked.

## Windows execution

The local notebook defaults to `Path.home()/"Downloads"` for the dataset parent. The project directory is resolved from the notebook working directory or its `Models/Abhay` child and can be overridden. Results are written to timestamped `local_runs` subdirectories.

From `Models/Abhay` in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt jupyterlab ipykernel
.\.venv\Scripts\python.exe -m jupyter lab
```

CLI equivalent:

```powershell
python train_eval.py --data "$env:USERPROFILE\Downloads\reid_student_data_main" --output ./abhay_output
python test_pipeline.py -v
```

`RUN_SMOKE=True` enables a reduced notebook execution check. `PREDICT_TEST=True` adds test CSV generation after validation. Neither switch changes the default full experiment unless enabled. A new timestamped local run starts a new cache; the same output directory is required for cache reuse. Interrupted model-stage training is not resumed automatically.

## Runtime artifacts

- `metrics.json`: MAP@5, known/unknown scores, rank-one new rates, candidate recall, delta, model name, configuration and library versions.
- `validation_predictions.csv`, `validation_slices.csv`: per-query scores, predictions and grouped error analysis.
- `fusion_coefficients.csv`: standardized coefficients; correlated features limit causal interpretation.
- `identity_roles.csv`, `episode_splits.csv`: reproducible role and episode assignments.
- `calibration_examples.csv`: targets and inverse-occurrence weights.
- `models.joblib`: frozen stack, fusion and feature schema.
- `sift_*.npz`: atomic pair cache keyed by IDs and normalized global features. Only role-local requested pairs are exposed during each phase.

Optional test inference keeps the learned weights frozen and uses the full labelled gallery. It exports `submission_dalvin_rule.csv`, `submission_detector_only.csv` and `submission_abhay_fusion.csv`, validating sample IDs/order, five distinct labels and species consistency. It does not submit to Kaggle or automatically declare a winning method. Full-gallery distribution shift remains a limitation.

## Verification and attribution

Synthetic tests cover feature-ID permutation, compressed and extracted SIFT layouts, empty descriptors, actual LightGBM training through three stacking stages, calibration, independent roles, preserved known-identity ordering and valid test submissions. Both notebooks' code cells compile and embedded modules match the standalone versions. The local data preflight verified 7,163 training sightings, 2,791 test sightings, a 9,954 x 2,152 feature matrix and both SIFT sources.

Dalvin's source attribution is retained in code and `provenance.json`. Adaptations include import-safe data loading; deterministic seeds and thread settings; bounded PCA dimension; guarded normalization; portable SIFT thread execution with locked ZIP reads; ID alignment; stable score ties; independent evaluation roles; final-stage fusion; and diagnostics. The original shortlist and zero treatment of unverified SIFT pairs are retained. No algorithm change is introduced by documenting the measured result.
