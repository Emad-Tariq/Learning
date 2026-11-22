Emad:
- **Model 1:** Implemented MIEW-ID cosine similarity baseline for species-aware identity retrieval. Achieved 0.53083 public leaderboard score.
- **Model 2:** Extended baseline with SIFT feature matching, RANSAC verification, and species-specific open-set thresholds. Achieved 0.765 public leaderboard score.

Dalvin:
- **Model 3:** LDA-projected MIEW-ID retrieval, RootSIFT + RANSAC verification, and LightGBM open-set ranking (new-individual detector + candidate scorers) trained on simulated open-set splits, with 3 rounds of transductive stacking over nearest test neighbours. Public leaderboard score: 0.84305.
- **Model 3b:** Same pipeline trained on a mix of simulated regimes (random holdout plus similarity-aware splits that hold out whole near-duplicate groups), PCA fitted on gallery and queries, and SIFT verification widened to 150 neighbours per image. Simulated out-of-fold MAP@5 is 0.906 on the harder mix (not comparable with Model 3's 0.916).
