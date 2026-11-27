# Model 10: difficult-episode full-pipeline retraining

Development selected the unchanged baseline. Neither experimental fraction qualified under the development constraints; validation results are reported for comparison, not used to change that choice.

Overall MAP@5: Dalvin baseline 0.934272, 25% difficult 0.934600, 50% difficult 0.933743.

The full pipeline was retrained for all three methods, with identical model settings and 36 total episodes each. Development uses separate identities and selects the mixture before validation. These validation identities and the labelled dataset have been inspected previously. This is one outer identity allocation with repeated photos across episodes; bootstrap intervals do not establish retraining robustness or a Kaggle gain.

Development-selected method: **Dalvin baseline**.

| Slice | Appearances | Unique photos | Baseline | 25% | Difference | 50% | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1829 | 1147 | 0.934272 | 0.934600 | +0.000328 | 0.933743 | -0.000529 |
| Known animals | 991 | 783 | 0.897191 | 0.898301 | +0.001110 | 0.896216 | -0.000975 |
| Unknown animals | 838 | 663 | 0.978123 | 0.977526 | -0.000597 | 0.978123 | +0.000000 |
| Missing side | 49 | 37 | 0.560204 | 0.540816 | -0.019388 | 0.545918 | -0.014286 |
| Small gallery (1–3) | 134 | 100 | 0.691169 | 0.682711 | -0.008458 | 0.691169 | +0.000000 |
| Missing side + small gallery | 41 | 30 | 0.542683 | 0.534553 | -0.008130 | 0.540650 | -0.002033 |
| Same side available | 659 | 518 | 0.950405 | 0.951416 | +0.001012 | 0.950860 | +0.000455 |
| 4+ gallery photos | 857 | 683 | 0.929405 | 0.932011 | +0.002606 | 0.928277 | -0.001128 |

## Reliability and parity

- 25% difficult overall paired delta interval: [-0.003036, +0.003341]; validation gate passed: False.
- 50% difficult overall paired delta interval: [-0.004552, +0.002954]; validation gate passed: False.

The retrained baseline reproduces all 1829 previous validation top-five predictions exactly. Sampler tests check nested proportions, class/identity conservation, deterministic behavior, gallery subset safety and harder unknown selection. Realized distributions confirm equal known/unknown query counts and exactly 9 or 18 difficult episodes out of 36.

No original model, competition submission or earlier experiment was overwritten. All three trained stacks, source hashes, configuration, identity roles, exact episode manifests, development selection and per-query predictions are saved in `local_runs/difficult_2052`. See `README.md` to reproduce.