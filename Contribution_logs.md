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
- **Model 5 — Open-Set Score Fusion (OSSF):** Added final-stage logistic fusion over Dalvin Model 3b evidence, with legacy-rule and detector-only ablations on a shared identity-disjoint sequential-stacking evaluation. Validation (506 queries): legacy rule MAP@5 **0.916206**, detector-only **0.909947**, OSSF **0.911594**; fusion delta **-0.00461133**. Known-query MAP@5 increased slightly overall, but unknown-query performance decreased; no overall improvement or leaderboard gain is claimed. Added Kaggle/Windows notebooks, reproducible splits, diagnostics, tests and documented negative-result analysis in `Models/Abhay/RESULTS.md`.
