# Model 15: Gallery-conditioned album matching and sparse-reference specialist

See [MODEL.md](MODEL.md) for the main pipeline, motivation from previous models and EDA, recorded validation results, and source evidence.

## Run

Run from the repository root with the original competition inputs and the earlier model assets required by the runner. Development commands protect frozen recipes; reuse the saved recipe for confirmation and prediction.

```powershell
python Models/Abhay/Model15/run.py develop
python Models/Abhay/Model15/run.py confirm
python Models/Abhay/Model15/specialist.py develop
python Models/Abhay/Model15/specialist.py confirm
python Models/Abhay/Model15/specialist.py predict
```

Models 14–15 require the portable Model 12 assets and the saved confirmation checkpoint. Model 15 also uses Model 14 as its specialist control. Model 16 fits an independent scorer and reuses geometric measurements with checked index mapping. Original paths and optional overrides are defined in the runners; model assets and raw data must be supplied separately.

Saved JSON metrics and frozen recipes are in `results/`. Prediction commands validate output IDs, label counts and species compatibility. Local validation scores and uncertainty must be interpreted within each paired protocol. Baseline and teammate source attribution is retained in the code and [model documentation](MODEL.md).
