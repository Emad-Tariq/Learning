# Model 10 follow-up: detector coverage inputs and difficult training

Development selected the unchanged baseline. Neither experimental fraction qualified under the development constraints; validation results are reported for comparison, not used to change that choice.

Overall MAP@5: Dalvin baseline 0.934272, 25% + detector coverage 0.935666, 50% + detector coverage 0.930691.

The full pipeline was retrained for all three methods, with identical model settings and 36 total episodes each. Development uses separate identities and selects the mixture before validation. These validation identities and the labelled dataset have been inspected previously. This is one outer identity allocation with repeated photos across episodes; bootstrap intervals do not establish retraining robustness or a Kaggle gain. This follow-up adds 23 observable candidate-coverage inputs to the trained tree unknown detector in the experimental variants only. It was designed after the episode-only outcome; it is exploratory and separate from the requested pure-mixture comparison.

Development-selected method: **Dalvin baseline**.

| Slice | Appearances | Unique photos | Baseline | 25% | Difference | 50% | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1829 | 1147 | 0.934272 | 0.935666 | +0.001394 | 0.930691 | -0.003581 |
| Known animals | 991 | 783 | 0.897191 | 0.899092 | +0.001900 | 0.896384 | -0.000807 |
| Unknown animals | 838 | 663 | 0.978123 | 0.978918 | +0.000796 | 0.971261 | -0.006862 |
| Missing side | 49 | 37 | 0.560204 | 0.518707 | -0.041497 | 0.554422 | -0.005782 |
| Small gallery (1–3) | 134 | 100 | 0.691169 | 0.680597 | -0.010572 | 0.694652 | +0.003483 |
| Missing side + small gallery | 41 | 30 | 0.542683 | 0.506098 | -0.036585 | 0.554878 | +0.012195 |
| Same side available | 659 | 518 | 0.950405 | 0.952681 | +0.002276 | 0.949039 | -0.001366 |
| 4+ gallery photos | 857 | 683 | 0.929405 | 0.933256 | +0.003851 | 0.927927 | -0.001478 |

## Reliability and parity

- 25% + detector coverage overall paired delta interval: [-0.002386, +0.004828]; validation gate passed: False.
- 50% + detector coverage overall paired delta interval: [-0.007668, -0.000217]; validation gate passed: False.

The retrained baseline reproduces all 1829 previous validation top-five predictions exactly. Sampler tests check nested proportions, class/identity conservation, deterministic behavior, gallery subset safety and harder unknown selection. Realized distributions confirm equal known/unknown query counts and exactly 9 or 18 difficult episodes out of 36.

No original model, competition submission or earlier experiment was overwritten. All three trained stacks, source hashes, configuration, identity roles, exact episode manifests, development selection and per-query predictions are saved in `../local_runs/coverage_difficult_2052`. See `README.md` to reproduce.