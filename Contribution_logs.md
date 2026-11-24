Emad:
- **Model 1:** Implemented MIEW-ID cosine similarity baseline for species-aware identity retrieval. Achieved 0.53083 public leaderboard score.
- **Model 2:** Extended baseline with SIFT feature matching, RANSAC verification, and species-specific open-set thresholds. Achieved 0.765 public leaderboard score.

Dalvin:
- **Model 3:** LDA-projected MIEW-ID retrieval, RootSIFT + RANSAC verification, and LightGBM open-set ranking (new-individual detector + candidate scorers) trained on simulated open-set splits, with 3 rounds of transductive stacking over nearest test neighbours. Public leaderboard score: 0.84305.
- **Model 3b:** Same pipeline trained on a mix of simulated regimes (random holdout plus similarity-aware splits that hold out whole near-duplicate groups), PCA fitted on gallery and queries, and SIFT verification widened to 150 neighbours per image. Public leaderboard score: 0.85078 (simulated out-of-fold MAP@5 0.906 on the harder mix).
- **Model 3c:** Model 3b plus viewpoint-coverage features (how many gallery images of a candidate share the query's viewpoint or flank). Simulated out-of-fold MAP@5 0.9073 (3b: 0.9056). Public leaderboard score: 0.84665 (no gain over 3b within noise).
- **Model 3d:** Model 3c with a stricter SIFT verifier (mutual nearest-neighbour check, ratio 0.85, 15px RANSAC), which detects about 23% of same-identity pairs vs 13.5% before at a 1% false-match rate. Simulated out-of-fold MAP@5 0.9132 (3c: 0.9073). Public leaderboard score: 0.85960 (best so far).

Dewang:
- **Model 4:** PCA-whitened MiewID retrieval with viewpoint-aware identity scoring (same-side bonus / opposite-side penalty),
  optional RootSIFT + RANSAC bonus, and a per-species logistic-regression new-individual detector whose probability places
  `new_individual` in the top-5 ranking. Shared open-set validation splits saved to Processed_Data/splits/.
  Public Scoreboard: 0.77431
