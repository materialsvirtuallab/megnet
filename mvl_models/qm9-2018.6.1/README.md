These models are trained on the QM9 molecules crystals data set.
The details of this model and benchmarks are provided in our publication
["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294).

## Performance

| Property | Units      | MAE   |
|----------|------------|-------|
| HOMO     | eV         | 0.043 |
| LUMO     | eV         | 0.044 |
| Gap      | eV         | 0.066 |
| ZPVE     | meV        | 1.43  |
| µ        | Debye      | 0.05  |
| α        | Bohr^3     | 0.081 |
| \<R2\>   | Bohr^2     | 0.302 |
| U0       | eV         | 0.012 |
| U        | eV         | 0.013 |
| H        | eV         | 0.012 |
| G        | eV         | 0.012 |
| Cv       | cal/(molK) | 0.029|
| ω1       | cm^-1   | 1.18 |

It should be noted that for QM9 models, we do not expect transferability to
other molecules, since the QM9 dataset is limited in scope. Therefore please
only use it for testing QM9. Out of the 13 targets, we set `HOMO`, `LUMO`,
`gap`, and `omega1` to be intrinsic quantities and the models for them are
fitted on their scaled values. For other targets, however, the models are
fitted on `per_atom` quantities. We have a `scaler.json` file for the QM9
models that specifies the scaling factors. Please see `notebooks/qm9_pretrained.ipynb`
as examples of how to use it properly.
