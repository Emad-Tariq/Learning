Emad:
- **Model 1:** Implemented MIEW-ID cosine similarity baseline for species-aware identity retrieval. Achieved 0.53083 public leaderboard score.
- **Model 2:** Extended baseline with SIFT feature matching, RANSAC verification, and species-specific open-set thresholds. Achieved 0.765 public leaderboard score.

Dalvin:
- **Model 3:** LDA-projected MIEW-ID retrieval, RootSIFT + RANSAC verification, and LightGBM open-set ranking (new-individual detector + candidate scorers) trained on simulated open-set splits, with 3 rounds of transductive stacking over nearest test neighbours. Public leaderboard score: 0.84305.
- **Model 3b:** Same pipeline trained on a mix of simulated regimes (random holdout plus similarity-aware splits that hold out whole near-duplicate groups), PCA fitted on gallery and queries, and SIFT verification widened to 150 neighbours per image. Public leaderboard score: 0.85078 (simulated out-of-fold MAP@5 0.906 on the harder mix).
Dewang:
- **Model 4:** PCA-whitened MiewID retrieval with viewpoint-aware identity scoring (same-side bonus / opposite-side penalty),
  optional RootSIFT + RANSAC bonus, and a per-species logistic-regression new-individual detector whose probability places
  `new_individual` in the top-5 ranking. Shared open-set validation splits saved to Processed_Data/splits/.
  Public Scoreboard: 0.77431

Abhay:
- **Model 5 — Open-Set Score Fusion (OSSF):** Implemented final-stage logistic fusion of unknown-detector, candidate-ranking, similarity, viewpoint and SIFT evidence on top of Dalvin's Model 3b. Retrieval and conditional known-identity ranking remain shared across the legacy-rule, detector-only and learned-fusion comparisons.
- **Validation (24 November 2026):** Evaluated on 506 queries (349 known, 157 unknown) using identity-disjoint sequential stacking stages, calibration and outer validation. MAP@5: legacy rule **0.916206**, detector-only **0.909947**, OSSF **0.911594**. Fusion delta: **-0.00461133**. These are validation scores, not Kaggle leaderboard scores.
- **Error analysis:** Known-query MAP@5 increased from **0.887106** to **0.890449**, while unknown-query MAP@5 decreased from **0.980892** to **0.958599**. The loss on unseen animals outweighed the gain on known animals; the legacy rule remains strongest on this split. No overall improvement or leaderboard gain is claimed. Detailed results and limitations are recorded in `Models/Abhay/RESULTS.md`.
- **Reproducibility:** Added self-contained Kaggle and Windows notebooks, explicit feature-ID alignment, SIFT archive/folder support, split manifests, checkpoints, prediction/error diagnostics, dependency specifications and three passing synthetic tests. Baseline attribution is retained in `provenance.json`; runtime data and artifacts are excluded from Git.
