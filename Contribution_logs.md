Emad:
- **Model 7:** Ambiguity-aware gallery pair scoring and background-adjusted SIFT. Model 7 scored **0.85187** on the public Kaggle leaderboard, 0.00773 below Model 3d (0.85960). Full local simulated MAP@5 was 0.91301 for Model 7 and 0.91249 for the matched 3d control.
- **Model 1:** Implemented MIEW-ID cosine similarity baseline for species-aware identity retrieval. Achieved 0.53083 public leaderboard score.
- **Model 2:** Extended baseline with SIFT feature matching, RANSAC verification, and species-specific open-set thresholds. Achieved 0.765 public leaderboard score.
- **Model 6:** Tested joint open-set ranking with cross-flank retrieval and domain weighting. Public Kaggle MAP@5 was 0.83194, compared with 0.85960 for Model 3d.

Dalvin:
- **Model 3:** LDA-projected MIEW-ID retrieval, RootSIFT + RANSAC verification, and LightGBM open-set ranking (new-individual detector + candidate scorers) trained on simulated open-set splits, with 3 rounds of transductive stacking over nearest test neighbours. Simulated out-of-fold MAP@5: 0.916. Public leaderboard score: pending submission.

Dewang:
- **Model 4:** PCA-whitened MiewID retrieval with viewpoint-aware identity scoring (same-side bonus / opposite-side penalty),
  optional RootSIFT + RANSAC bonus, and a per-species logistic-regression new-individual detector whose probability places
  `new_individual` in the top-5 ranking. Shared open-set validation splits saved to Processed_Data/splits/.
  Public Scoreboard: 0.77431
- **Model 4b:** Upgraded Model 4 into a PCA → LDA retrieval pipeline with LightGBM open-set ranking. Replaced the logistic-regression
  detector with a query-level new-individual detector plus closed-set and all-query (query, candidate) scorers, trained on 16
  regime-diverse simulated open-set episodes (identity-disjoint, near-duplicate-aware group holdout, varying unknown fraction,
  query cap and linking threshold). Added 3 rounds of transductive stacking (each round's predictions for a query's nearest test
  neighbours feed the next round), richer features (same-view / same-flank similarity, per-identity viewpoint coverage, per-identity
  SIFT support counts, candidate centroid similarity, query-to-query embedding and SIFT neighbours), and mutual nearest-neighbour
  RootSIFT matching with RANSAC verification (SIFT pair results cached for reuse). Regularised LDA (shrinkage 0.3) and subsampled
  LightGBM used to limit overfitting. Public leaderboard score: 0.85373.
- **Model 4c:** Extended Model 4b with seed-bagged LightGBM (3 seeds), 24 simulated episodes instead of 16, a softmax /
  kernel-density identity posterior over all gallery images (with its maximum and entropy as open-set evidence), a PCA-whitened
  similarity signal (top 128 components) and stronger LightGBM regularisation. Added candidate recall@K and simulation-vs-test
  covariate-shift diagnostics. Public leaderboard score: 0.85952.
- **Model 4d:** Extended Model 4c with additional evidence features (linear metric learning with a supervised contrastive loss,
  label diffusion over an embedding and SIFT graph, k-reciprocal re-ranking, query expansion, query clustering and extra SIFT
  statistics), a LambdaRank candidate ranker, tuning of the `new_individual` score, and per-group ablation.
  Competition-data validation and public leaderboard score: not yet recorded.
- **Model 4e:** Extended Model 4c to address simulation-vs-test mismatch and validation leakage: viewpoint-matched simulated
  episodes, a test-vs-simulation classifier with importance weighting, identity-disjoint validation folds, a LambdaRank ranker
  with a `new_individual` candidate (compared with score fusion and a blend), down-weighting of generic SIFT keypoints, extra
  geometric-verification statistics, PLDA evidence and an optional new-individual prior estimate.
  Competition-data validation and public leaderboard score: not yet recorded.
  
Abhay:
- **EDA:** Implemented metadata, identity-frequency, sparse-gallery, viewpoint/side-coverage, near-duplicate, embedding retrieval and SIFT diagnostics, with reproducible scripts, CSV tables, plots and an HTML report. Compared photo- and identity-weighted retrieval and isolated candidate omission, ranking and new-animal placement failure modes.
- **Research:** Investigated retrieval hubness, nuisance projection, gallery calibration and geometric verification coverage; retained measured limitations and negative experiments.
- **Model 5 — Open-Set Score Fusion (OSSF):** Implemented and evaluated the approach documented in `Models/Abhay/Model5/MODEL.md`.
- **Model 6 — Pairwise Identity Verification (PIV):** Implemented and evaluated the approach documented in `Models/Abhay/Model6/MODEL.md`.
- **Model 7 — Nuisance Evidence Retrieval (NER):** Implemented and evaluated the approach documented in `Models/Abhay/Model7/MODEL.md`.
- **Model 8 — Raw and LDA union shortlist:** Implemented and evaluated the approach documented in `Models/Abhay/Model8/MODEL.md`.
- **Model 9 — Coverage-aware residual scoring:** Implemented and evaluated the approach documented in `Models/Abhay/Model9/MODEL.md`.
- **Model 10 — Difficult-episode retraining and coverage-input follow-up:** Implemented and evaluated the approach documented in `Models/Abhay/Model10/MODEL.md`.
- **Model 11 — Full-vector MLP evidence for novelty detection:** Implemented and evaluated the approach documented in `Models/Abhay/Model11/MODEL.md`.
- **Model 12 — Larger-gallery retraining:** Implemented and evaluated the approach documented in `Models/Abhay/Model12/MODEL.md`.
- **Model 13 — Joint ranking, regularized heads and PLDA evidence:** Implemented and evaluated the approach documented in `Models/Abhay/Model13/MODEL.md`.
- **Model 14 — Query-neighbour score sharing:** Implemented and evaluated the approach documented in `Models/Abhay/Model14/MODEL.md`.
- **Model 15 — Gallery-conditioned album matching and sparse-reference specialist:** Implemented and evaluated the approach documented in `Models/Abhay/Model15/MODEL.md`.
- **Model 16 — Independent joint known/new scorer:** Implemented and evaluated the approach documented in `Models/Abhay/Model16/MODEL.md`.
- **Model 17 — Gallery-trained metric graph and learned sighting links:** Implemented and evaluated the approach documented in `Models/Abhay/Model17/MODEL.md`.
- **Measured results:** Model 12 improved matched larger-gallery validation MAP@5 from 0.878358 to 0.907654. Model 5 scored 0.911594 versus its 0.916206 baseline; Model 6 and Models 8–11 did not establish overall gains. Models 14–15 had small local gains with uncertainty including zero; Models 16–17 did not beat the specialist on paired confirmation. Different evaluation protocols must not be compared as one leaderboard.
- **Reproducibility and documentation:** Added identity-disjoint roles, duplicate-group-aware episodes, matched ablations, per-query/species/coverage diagnostics, checkpointing, independent audits, local/Kaggle notebooks and prediction-file validation. Organized Models 5–17 and documented their pipelines and design rationale. Dalvin's reused baseline and Emad's referenced work retain attribution.
 
