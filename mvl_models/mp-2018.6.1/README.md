These models are trained on the 2018.6.1 Materials Project crystals data set.
The details of this model and benchmarks are provided in our publication
["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294).

The data file is shared here [https://figshare.com/articles/Graphs_of_materials_project/7451351](https://figshare.com/articles/Graphs_of_materials_project/7451351)

## Performance

| Property | Units      | MAE   | Data size |
|----------|------------|-------|-----------|
| Ef       | eV/atom    | 0.028 | 60,000    |
| Eg       | eV         | 0.33  | 36,720    |
| K_VRH    | log10(GPa) | 0.050 | 4,664     |
| G_VRH    | log10(GPa) | 0.079 | 4,664     |
| Metal classifier | - | 78.9% | 55,931     |
| Non-Metal classifier | - | 90.6% | 55,931     |
