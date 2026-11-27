# Model 10: Difficult-episode retraining and coverage-input follow-up

## Purpose and design motivation

Models 8 and 9 did not resolve the EDA's missing-side and sparse-reference failures. Post-hoc corrections may be insufficient if training rarely exposes the full detector to those cases. This model changes simulated training episodes while holding architecture, total episode count and query class counts fixed.

## Main model and pipeline

Retrain the baseline with 0%, 25% or 50% difficult episodes. Within selected gallery identities, retain 1–3 photos and remove a queried flank where feasible, keeping at least one reference. Preserve known queries and duplicate-group separation. For unknown identities, retain the same quotas but select photos most similar to the reduced gallery. Use 12 episodes per stage and three stacking stages. Select the fraction on separate development episodes. A separately saved exploratory follow-up adds 23 observable coverage features directly to the unknown detector and retrains the affected stacks.

## Evaluation and what was learned

Development retained the baseline. Validation MAP@5 was 0.934272, 0.934600 and 0.933743 for baseline, 25% and 50%; difficult slices did not improve reliably. The coverage-input follow-up is a distinct intervention and must be read separately from the episode-only result.

Scores above are local validation unless explicitly identified as reported public scores. Splits and galleries differ across experiments; compare paired arms within a model rather than sorting these values into one leaderboard. Repeated episodes can reuse animals and photos, and bootstrap intervals condition on fitted models. Later EDA observations are supporting context, not retroactive proof of an earlier design's motivation.

## Implementation and evidence

- [difficult_protocol.py](difficult_protocol.py)
- [difficult_eval.py](difficult_eval.py)
- [coverage_features.py](coverage_features.py)
- [coverage_train.py](coverage_train.py)
- [RESULTS.md](RESULTS.md)
- [coverage_followup/RESULTS.md](coverage_followup/RESULTS.md)

Abhay's contribution is the experimental change, implementation, evaluation and documentation described here. Dalvin retains credit for adapted baseline components; teammate work remains credited in the original sources. Shared modules and runtime requirements are described in [the model index](../README.md).
