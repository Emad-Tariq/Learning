# Abhay: EDA and Models 5–17

Each numbered model has its own folder. Open its `MODEL.md` for how it was built, why that design was chosen, the evidence that motivated it and the recorded outcome. Existing source, notebooks, detailed run instructions and result artifacts remain alongside the model.

| Model | Main approach |
| --- | --- |
| [5](Model5/MODEL.md) | Open-Set Score Fusion (OSSF) |
| [6](Model6/MODEL.md) | Pairwise Identity Verification (PIV) |
| [7](Model7/MODEL.md) | Nuisance Evidence Retrieval (NER) |
| [8](Model8/MODEL.md) | Raw and LDA union shortlist |
| [9](Model9/MODEL.md) | Coverage-aware residual scoring |
| [10](Model10/MODEL.md) | Difficult-episode retraining and coverage-input follow-up |
| [11](Model11/MODEL.md) | Full-vector MLP evidence for novelty detection |
| [12](Model12/MODEL.md) | Larger-gallery retraining |
| [13](Model13/MODEL.md) | Joint ranking, regularized heads and PLDA evidence |
| [14](Model14/MODEL.md) | Query-neighbour score sharing |
| [15](Model15/MODEL.md) | Gallery-conditioned album matching and sparse-reference specialist |
| [16](Model16/MODEL.md) | Independent joint known/new scorer |
| [17](Model17/MODEL.md) | Gallery-trained metric graph and learned sighting links |

## EDA and research

[EDA design findings](EDA/DESIGN_FINDINGS.md) summarize the labelled-data observations used to motivate the later experiments. The EDA covers identity imbalance, sparse reference albums, near duplicates, viewpoint/side coverage, raw versus PCA/LDA retrieval, SIFT support and known/new score overlap. EDA scripts, tables and plots remain in `EDA/`. Retrieval, hubness, nuisance-projection and gallery-calibration investigations are retained in `Research/` and the existing research notes. The [SIFT coverage audit](sift_coverage/README.md) is supporting analysis, not a separately numbered model.

The 25 November EDA came after Models 5–7 and their earlier research. Its observations provide retrospective context for those models and direct motivation for later coverage experiments; the documentation distinguishes those roles.

## Layout and execution

Model 5's formerly root-level scripts, notebooks, requirements, results and local run folders are now in `Model5/`. For example, from the repository root:

```powershell
python Models/Abhay/Model5/train_eval.py --help
python Models/Abhay/Model5/test_pipeline.py -v
```

Models 6–17 retain their existing relative locations so sibling imports remain valid. These folders are experiment units and some share implementation: Models 8–13 use the preserved Model 7 baseline/protocol; later models also use earlier checkpoints. Models 12–13 refer to the repository's `analysis/domain_shift` helpers and saved manifests. Models 14–17 require original competition inputs and the assets/teammate sources described in their runners. Copying the model folders alone does not replace those external inputs. Existing absolute paths and checkpoint assumptions remain visible in the original code; this organization does not claim a new environment was provisioned or training was rerun.

Use each model's existing notebook/run instructions and dependency file where provided. Runtime caches, fitted weights and local runs remain associated with their model. `CONTRIBUTION.md` records Abhay's work across this collection; the project-wide contribution log is at `../../Contribution_logs.md`.
