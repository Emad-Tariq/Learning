# Model 7: Nuisance Evidence Retrieval (NER)

Measured outcome: local open-set MAP@5 rises from 0.934272 to 0.935721, but one of three episodes worsens and the estimated improvement interval includes zero. The submission gate failed. No CSV was generated. See `RESULTS.md` and `results/` for the saved comparison. These are validation scores, not Kaggle scores.

This experiment tests whether a frozen nuisance projection improves Dalvin Model 3d's complete open-set pipeline. It replaces the raw similarity evidence with similarities between transformed vectors. It preserves the original PCA/LDA retrieval, candidate list, mutual RootSIFT/RANSAC verification, unknown detector, candidate scorers and three prediction-stacking stages. Both versions are retrained for this paired comparison.

## Transformation

For every fitting identity with multiple photos, subtract its mean descriptor from each photo. Scale its residual rows by `1/sqrt(n-1)` so each identity contributes equal covariance weight. Estimate 64 residual singular directions and retain the leading 16. Reduce each photo's projection along those 16 directions by half, preserve its other coordinates, and normalize the result. Fit one transformation per species. No image features are recomputed from pixels.

The dimensions (16) and strength (0.5) were frozen in the earlier matching research. The open-set comparison does not tune them or the baseline models using validation outcomes.

## Protocol

Split identities separately within each species, seed 2052: 20% final validation, 20% metric fitting, and three 20% ranking-training stages. Metric-fitting identities train only the nuisance transformation. Each stacking stage sees its own identity set; earlier models supply predictions to the next stage. Final validation labels are unavailable to every supervised fitting step.

Each stage uses 12 simulated open-set gallery/query episodes. Dalvin's three LightGBM components are fitted per stage: unknown detector, known-query candidate scorer and all-query candidate scorer. Candidate scorers use 400 trees; the original detector uses 300. Both variants share episode assignments, 15 candidate identities, 150 same-species SIFT neighbours and all random seeds. RootSIFT uses a mutual 0.85 ratio test and partial-affine RANSAC with 15-pixel tolerance.

Three predeclared episodes on validation identities contain known queries and entirely held-out unknown identities. Near-duplicate groups stay wholly in gallery or query. The projection affects only raw similarity features, not SIFT neighbour selection, duplicate grouping or original PCA/LDA features. The baseline and transformed versions must have identical candidate lists; runtime assertions enforce this. PCA sees unlabeled episode query features as in Dalvin's implementation; LDA sees gallery labels only.

This differs from the teammate's original training protocol, which trained stacking with out-of-fold simulated splits over all training identities. Our stricter outer identity separation produces smaller training galleries. It is a controlled comparison of the complete components, not a reproduction of the team's Kaggle score. Multiple validation episodes reuse identities and can reuse query images; their scores are correlated.

## Decision gate

Generate a submission only when the transformed version has positive pooled and per-episode MAP@5 differences, and pooled known-query and unknown-query MAP@5 do not fall. This gate is declared in the saved configuration before fitting. If it fails, the runner saves the negative result and skips submission generation. Passing does not guarantee a leaderboard improvement.

## Run

CPU is used throughout. From this directory:

```powershell
python -m pip install -r requirements.txt
python paired_eval.py --predict-test
```

Data defaults to `~/Downloads/reid_student_data_main`. SIFT archives or extracted NPY directories must be in the same directory or a parent, as supplied in Downloads. `--data` and `--output` override paths. Outputs default to `local_runs/paired_2052`, which is ignored by the parent `.gitignore`. First-run mutual SIFT verification can take substantial time; pairs are cached under `sift_mutual_*.npz` and shared by the baseline and transformed version. Older non-mutual Model 5 caches cannot be reused.

`NER_Local.ipynb` and `NER_Kaggle.ipynb` contain source snapshots and run the same command using the notebook kernel. The Kaggle notebook finds the competition metadata directory under `/kaggle/input`. It requires the competition SIFT files too. Neither notebook submits to Kaggle.

`identity_roles.csv`, `episode_splits.csv`, `config.json`, `paired_models.joblib`, `nuisance_directions.joblib`, `validation_predictions.csv` and `metrics.json` make each run auditable. `submission_nuisance.csv` appears only if the gate passes and `--predict-test` is supplied. The submission then uses the already-frozen stage models and projection with the full labeled gallery, so gallery-size shift remains a limitation.

## Attribution and checks

`dalvin_core.py` derives from `Models/Dalvin/reid_pipeline.py` (normalized source-text SHA256 `65fac657ffb6a63c687d27fbf8938b7bf67dac3d874c2d7569c9597f48ea9f6d`). Besides selecting the nuisance raw representation, the PCA seed is fixed and its dimensionality is bounded by feature count. Data/protocol helpers derive from the earlier Abhay Model 5 runner. Model 5's learned score fusion is not used.

`python -m unittest test_integration.py` checks exact baseline feature parity against the original Dalvin feature function and confirms that labels outside the metric-fitting set do not affect the learned transformation. The original feature function is retained in `baseline_species_features.py` for this check.

`python audit_results.py --run local_runs/paired_2052` independently recomputes scores, validates identity roles and prediction labels, and estimates uncertainty by resampling animal identities. Both notebooks run this output audit after evaluation.
