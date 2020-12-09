## Multi-fidelity graph network models for band gaps

These models were trained using PBE, GLLB-SC, HSE and Experimental datasets.
The details of the models and benchmarks are provided in our publication
["Multi-fidelity Graph Networks for Deep Learning the Experimental Properties of Ordered and Disordered Materials"](https://arxiv.org/abs/2005.04338)[1].

The other multi-fidelity models (including SCAN) from 1-fi to 5-fi are available here [https://figshare.com/articles/software/Trained_models_for_Learning_Properties_of_Ordered_and_Disordered_Materials_from_Multi-fidelity_Data/13350686](https://figshare.com/articles/software/Trained_models_for_Learning_Properties_of_Ordered_and_Disordered_Materials_from_Multi-fidelity_Data/13350686)

The full data (without ICSD structures) are here [https://figshare.com/articles/dataset/Learning_Properties_of_Ordered_and_Disordered_Materials_from_Multi-fidelity_Data/13040330](https://figshare.com/articles/dataset/Learning_Properties_of_Ordered_and_Disordered_Materials_from_Multi-fidelity_Data/13040330).
## Performance
Ordered model `pbe_gllb_hse_exp`

| Fidelity | PBE (eV)   |GLLB-SC (eV)|  HSE (eV) |Exp (eV)      |
|----------|------------|------------|-----------|--------------| 
| Errors   | 0.28±0.01  | 0.48±0.04  | 0.31±0.03 | 0.38±0.03    |


Disordered model `pbe_gllb_hse_exp_disorder`

| Fidelity | PBE (eV)   |GLLB-SC (eV)|  HSE (eV) |Exp. Ord. (eV) | Exp. Disord. (eV)|
|----------|------------|------------|-----------|--------------|------------------|
| Errors   | 0.27±0.01  | 0.47±0.03  | 0.30±0.03 | 0.37±0.02    | 0.51±0.11        |


## Data source

The PBE dataset comprising 52,348 crystal structures with band gaps were obtained from Materials Project[2] on Jun 1 2019.
The GLLB-SC band gaps were from Castelli et al.[3]. The total number of GLLB-SC band gaps is 2,290 after filtering out materials that do not have structures in the current Materials Project database and those that failed the graph computations due to abnormally long bond (> 5 Angstrom). 
The GLLB-SC data all have positive band gaps due to the constraints applied in the structure selection in the previous work. 
The  SCAN band gaps for 472 nonmagnetic materials were obtained from Borlido et al.[4]. 
The HSE band gaps with corresponding Materials Project structures were downloaded from the MaterialGo website [5].
After filtering out ill-converged calculations and those that have a much smaller HSE band gap compared to the PBE band gaps, 6,030 data points remain, of which 2,775 are metallic. 
Finally, the experimental band gaps were obtained from the work by Zhuo et al.[6]. 
As this data set only contains compositions, the experimental crystal structure for each composition was obtained by looking up the lowest energy polymorph for a given formula in the Materials Project, followed by cross-referencing with the corresponding Inorganic Crystal Structure Database (ICSD) entry. 
Further, as multiple band gap can be reported for the same composition in this data set, the band gaps for the duplicated entries were averaged. 
In total, 2,703 ordered (938 binary, 1306 ternary and 459 quaternary) and 278 disordered (41 binary, 132 ternary and 105 quaternary) structure-band gap pairs were obtained. All data sets are publicly available.


## References 
[1] Chen C.; Zuo Y.; Ye W.; Li X.G; Ong S.P., Multi-fidelity Graph Networks for Deep Learning the Experimental Properties of Ordered and Disordered Materials, https://arxiv.org/abs/2005.04338

[2] Jain, A.;  Ong, S. P.;  Hautier, G.;  Chen, W.;  Richards, W. D.;  Dacek, S.;  Cholia, S.;Gunter, D.; Skinner, D.; Ceder, G. et al. Commentary:  The Materials Project:  A Materials Genome Approach to Accelerating Materials Innovation. APL Materials 2013,1, 011002.

[3] Castelli,  I.  E.;  H ̈user,  F.;  Pandey,  M.;  Li,  H.;  Thygesen,  K.  S.;  Seger,  B.;  Jain,  A.;Persson,  K.  A.;  Ceder,  G.;  Jacobsen,  K.  W.  New  Light-Harvesting  Materials  UsingAccurate  and  Efficient  Bandgap  Calculations. Advanced Energy Materials2015,5,1400915

[4] Borlido, P.; Aull, T.; Huran, A. W.; Tran, F.; Marques, M. A. L.; Botti, S. Large-Scale Benchmark of Exchange–Correlation Functionals for the Determination of ElectronicBand Gaps of Solids.Journal of Chemical Theory and Computation 2019,15, 5069–5079.

[5] Jie, J.; Weng, M.; Li, S.; Chen, D.; Li, S.; Xiao, W.; Zheng, J.; Pan, F.; Wang, L. A New MaterialGo Database and Its Comparison with Other High-Throughput Electronic Structure Databases for Their Predicted Energy Band Gaps.Science China Technological Sciences 2019.

[6] Zhuo,  Y.;  Mansouri  Tehrani,  A.;  Brgoch,  J.  Predicting  the  Band  Gaps  of  InorganicSolids by Machine Learning. The Journal of Physical Chemistry Letters 2018,9, 1668–1673