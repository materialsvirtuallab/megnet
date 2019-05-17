These models are trained on the 2018.6.1 Materials Project crystals data set.
The details of this model and benchmarks are provided in our publication
["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294).

The data file is shared here [https://figshare.com/articles/Graphs_of_materials_project/7451351](https://figshare.com/articles/Graphs_of_materials_project/7451351)

## Performance

| Property | Units      | MAE   | Data size |
|----------|------------|-------|-----------|
| Ef       | eV/atom    | 0.028 | 69,239    |
| Eg       | eV         | 0.33  | 45,901    |
| K_VRH    | log10(GPa) | 0.050 | 5,831     |
| G_VRH    | log10(GPa) | 0.079 | 5,831     |
| Metal classifier | - | 78.9% | 69,239     |
| Non-Metal classifier | - | 90.6% | 69,239     |

The model for Ef was trained using 60,000 as training and the rest was splitted evenly as validation and test. All other models use 0.8/0.1/0.1 splits for train/validation/test. 
