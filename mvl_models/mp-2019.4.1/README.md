These are updated models are trained on the 2019.4.1 Materials Project crystals data set. The data is available at [https://figshare.com/articles/Graphs_of_Materials_Project_20190401/8097992](https://figshare.com/articles/Graphs_of_Materials_Project_20190401/8097992).

## Performance

| Property | Units      | MAE   | Data size |
|----------|------------|-------|-----------|
| Ef       | eV/atom    | 0.026 | 133,420   |
| Efermi   | eV         | 0.288 | 66,680    | 
| log10(K) | log10(GPa) | 0.071 | 12,179    |
| log10(G) | log10(GPa) | 0.124 | 12,179    |
  
The models are trained with train:validation:test ratios of 0.8:0.1:0.1. 
 


