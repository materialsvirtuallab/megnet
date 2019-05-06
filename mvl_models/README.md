## Pre-trained models from Materials Virtual Lab

This directory contains pre-trained models from Materials Virtual Lab. 

Currently, we provide models for

* QM9 molecules:
    - HOMO: Highest occupied molecular orbital energy
    - LUMO: Lowest unoccupied molecular orbital energy
    - Gap: energy gap
    - ZPVE: zero point vibrational energy
    - µ: dipole moment
    - α: isotropic polarizability
    - \<R2\>: electronic spatial extent
    - U0: internal energy at 0 K
    - U: internal energy at 298 K
    - H: enthalpy at 298 K
    - G: Gibbs free energy at 298 K
    - Cv: heat capacity at 298 K
    - ω1: highest vibrational frequency.
    
* Materials Project (MP) crystals:
    - Formation energy from the elements
    - Band gap
    - Log 10 of Bulk Modulus (K)
    - Log 10 of Shear Modulus (G)


### Performance of MP-2018.6.1

| Property | Units      | MAE   |
|----------|------------|-------|
| Ef       | meV/atom   | 0.028 |
| Eg       | eV         | 0.33  |
| K_VRH    | log10(GPa) | 0.050 |
| G_VRH    | log10(GPa) | 0.079 |

For the MP crystals, we provide all models for the original 2018.6.1 dataset
that was used in the publication, as well as updated models based on the
larger MP data set of 133,000 crystals as of 2019.4.1.

It should be noted that for QM9 models, we do not expect transferability to 
other molecules, since the QM9 dataset is limited in scope. Therefore please 
only use it for testing QM9. Out of the 13 targets, we set `HOMO`, `LUMO`, 
`gap`, and `omega1` to be intrinsic quantities and the models for them are 
fitted on their scaled values. For other targets, however, the models are 
fitted on `per_atom` quantities. We have a `scaler.json` file for the QM9
models that specifies the scaling factors. Please see `notebooks/qm9_pretrained.ipynb` 
as examples of how to use it properly.