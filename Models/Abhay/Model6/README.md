# Model 6: Pairwise Identity Verification (PIV)

PIV tests whether a small MLP trained on pairs of supplied MIEW-ID descriptors can improve matching between **known** animal identities. This is a separate matching experiment. It does not perform open-set `new_individual` detection or create a competition submission. A gain in this experiment would justify a subsequent open-set design; it is not itself evidence of a higher Kaggle score.

The first identity-disjoint test did **not** pass the matching gate: evaluation MAP@5 was 0.3046 for PIV versus 0.9091 for original cosine. See `RESULTS.md` for the full comparison and failure analysis. The implementation is retained as a reproducible negative result.

## Method

1. Align descriptors to `train.csv` through `miewid_ids.csv`, then normalize each vector.
2. Split identities separately for hyenas and leopards: 60% fit, 20% development, 20% final evaluation. The split occurs before pair generation. No identity spans roles.
3. Fit a 128-dimensional PCA on fit-role images only. Normalize projected vectors. Fit species-specific LDA baselines on the same fit identities.
4. Form positive pairs from different near-duplicate groups of the same identity, preferring opposite flanks when available. Form hard negatives from other identities among the 30 most similar same-species images, plus random negatives. Images whose identity has only one group do not supply positive training pairs. This prevents near-identical sightings from making the task artificially easy.
5. Pair features are `|z_a-z_b|`, `z_a*z_b`, original cosine, projected cosine, and three simple viewpoint indicators. They are symmetric in pair order. Features are scaled using fit-role pairs only.
6. Fit an MLP with hidden layers (64, 32), L2 penalty 0.01 and fixed seed. Early stopping is disabled because a random pair-level validation split could share identities with training. Development identities are used to assess the fixed configuration; no automated hyperparameter search runs on them.
7. For each development and evaluation identity, hold out about 30% of whole near-duplicate groups as query sightings, capped at eight groups while retaining at least one gallery group. Identities with a single group remain gallery-only. Compare each query to all same-species gallery images. The maximum score per identity produces the ranking.

Cosine similarity, LDA cosine similarity, and PIV probability use identical gallery/query splits and identity aggregation. Report MAP@5, top-one accuracy, top-five recall and the fraction of queries for which a wrong identity scores at least as high as the correct one. Species scores and per-query predictions are exported.

The near-duplicate grouping uses only supplied global descriptor cosine similarity >=0.97 within an identity. It approximates encounter grouping; original encounter IDs are unavailable. The threshold is fixed before evaluation. On the supplied training set, this produces 215 development queries from 92 identities and 271 evaluation queries from 89 identities (seed 2026); the other identities have only one group and remain gallery-only. Gallery/query evaluation is *closed-set*: every evaluated query has a matching identity in its gallery. This isolates the proposed matching improvement. Positive-pair sampling excludes single-group identities and may underrepresent very small identities; this is a limitation to inspect in the results.

Pair examples are dependent: thousands of sampled pairs do not constitute thousands of independent animals. The evaluation identities, not pair rows, are held out. A single split still has variance; repeated predeclared identity splits would be needed before claiming a reliable improvement. The source training set has no original encounter IDs, so near-duplicate separation is approximate.

## Running

Kaggle: upload `PIV_Kaggle.ipynb`, attach the existing `projectwild3` competition input, and run the cells in order. Only `train.csv`, `miewid_ids.csv`, and `miewid_features.npy` are used. SIFT files and raw images are not needed for this first test. CPU is sufficient; scikit-learn's MLP has no GPU support.

Windows: open `PIV_Local.ipynb` in JupyterLab or VS Code. The dataset defaults to `Path.home()/"Downloads"/"reid_student_data_main"`. The notebook uses the active Python kernel and writes a new local run directory. Install dependencies from `requirements.txt` if the kernel lacks them.

CLI, from this directory:

```powershell
python -m pip install -r requirements.txt
python matching.py --data "$env:USERPROFILE\Downloads\reid_student_data_main" --output ./local_runs/piv_run
```

Results are written to `metrics.json`, `identity_roles.csv`, `development_split.csv`, `evaluation_split.csv`, per-query prediction CSVs and `model.joblib`. The split CSVs include identity, duplicate-group ID and gallery/query assignment so the grouping can be audited. These runtime artifacts are excluded from Git. Progress prints the pair count, model fit, and both score summaries. `--dimensions`, `--hard-k`, `--pairs-per-image`, and `--max-iter` permit development experiments; a final performance claim requires an evaluation split not used to select them. Run `python -m unittest test_matching.py` for the small synthetic pipeline and split-integrity checks.

## Decision rule

Proceed to open-set modelling only if PIV improves known-animal MAP@5 and top-one identity accuracy on evaluation identities, ideally for both species, with stable results across additional predeclared splits. If the improvement appears only on development identities or only on a small slice, the result does not support a more complex full pipeline. A promising verifier could then be combined with a separately validated `new_individual` detector; that second stage is outside Model 6's current scope.

## Competition rules and attribution

The project statement permits MLPs trained from scratch on supplied features, PCA, LDA and distance-based retrieval. PIV trains only on provided feature vectors and labels. It uses no pretrained models, images or external data. Dalvin's retrieval work motivated the comparison, but this implementation has separate pair construction, pair features, MLP training and evaluation. No baseline code is copied into Model 6.
