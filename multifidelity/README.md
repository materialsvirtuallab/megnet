## Multifidelity Graph Networks


This repository along with the MatErials Graph Network (`megnet`) package repository contain the data and code implementations of our recent work:


**Multi-fidelity Graph Networks for Deep Learning the Experimental Properties of Ordered and Disordered Materials**, Chen, C., Zuo, Y., Ye, W., Li, X.G. & Ong, S.P., arXiv preprint arXiv:2005.04338 (2020)
[https://arxiv.org/abs/2005.04338](https://arxiv.org/abs/2005.04338 "https://arxiv.org/abs/2005.04338"). 


The model construction has already been included in the recent `megnet v1.1.8` release. The current repository demonstrates the use of multi-fidelity `megnet` in the multi-fidelity materials data modeling.


### Table of contents
- [Overview](#overview)
- [Installation guides](#install)
	- [Pypi install](#pypi)
	- [Github install](#github)
- [System requirements](#sys)
	- [Hardware](#hard)
	- [Software](#soft)
- [Documentation](#doc)
	- [Multi-fidelity band gap data](#data)
	- [Model training](#train)
	- [Codes for plots](#plot)
- [License](#license)
- [Issues](#issues)

<a name="overview"></a>
### Overview

In recent years, machine learning has attracted considerable interests from the materials science community due to its powerfulness in modeling accurately the structure-property relationships, by-passing the time consuming experimental measurements or expensive first-principles calculations.  Among the structure-property relationship models, graph-based models, in particular MatErials Graph Network (`megnet`), have achieved remarkable accuracy in various property prediction tasks. However, the main drawbacks of such models lie in the requirements of sufficiently large and high-quality materials data, which are not available in most cases. The developments of multi-fidelity models solve this dilemma by fusing information from large, inaccurate data to the small, accurate materials data. 

The current repository leverages the existing `megnet` package developed by the same authors and extends the `megnet` capability to the modeling of multi-fidelity data sets. This repository will share the multi-fidelity band gap data used in the publication and show examples to run the model fitting for mult-fidelity data sets. 

<a name="install"></a>
### Installation guide


The following dependencies are needed. 

```
pymatgen>=2020.7.18
pandas>=1.0.5
tensorflow>=2.0.0
numpy>=1.19.1
monty>=3.0.4
megnet>=1.1.8
```
Users may change to `tensorflow-gpu>=2.0.0` to use the GPU version of tensorflow if your machine has Nvidia GPUs.

<a name="pypi"></a>
#### Pypi install
Users can install all the dependencies by using the `requirements.txt` file in the current repository.

```
pip install -r requirements.txt  # or requirements-gpu.txt
```

<a name="github"></a>
#### Github install

```
git clone https://github.com/materialsvirtuallab/megnet.git
cd megnet
pip install -r requirements.txt  # or requirements-gpu.txt
python setup.py install
```

<a name="sys"></a>
### System requirements

<a name="hard"></a>
#### Hardware requirements
New inferences using the models can be run on any standard computer. However, we suggest using GPUs with at least 8 Gb memory to run model fitting.

<a name="soft"></a>
#### Software requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:

- macOS: Catalina 10.15.5
- Linux: Ubuntu 18.04 (with tensorflow-gpu==2.0.0)


<a name="doc"></a>
### Documentations
We will show the data used in the multi-fidelity graph network paper and also the fitting procedures.

<a name="data"></a>
#### Multi-fidelity band gap data
The full band gap data used in the paper is located at `data_no_structs.json.gz`. Users can use the following code to extract it. 

```
import gzip
import json

with gzip.open("data_no_structs.json.gz", "rb") as f:
	data = json.loads(f.read())
```

`data` is a dictionary with the following format

```
- "pbe"
	- mp_id
		- PBE band gap
	- ...
- "hse"
	- mp_id
		- HSE band gap
	- ...
- "gllb-sc"
	- mp_id
		- GLLB-SC band gap
	- ...
- "scan"
	- mp_id
		- SCAN band gap
	- ...
- "ordered_exp"
	- icsd_id
		- Exp band gap
	- ...
- "disordered_exp"
	- icsd_id
		- Exp band gap
	- ...
```
where `mp_id` is the Materials Project materials ID for the material, and `icsd_id` is the ICSD materials ID. For example, the PBE band gap of NaCl (mp-22862, band gap 5.003 eV) can be accessed by `data['pbe']['mp-22862']`. Note that the Materials Project database is evolving with time and it is possible that certain ID is removed in latest release and there may also be some band gap value change for the same material. 

To get the structure that corresponds to the specific material id in Materials Project, users can use the `pymatgen` REST API. 

1. Register at Materials Project [https://www.materialsproject.org](https://www.materialsproject.org) and get an `API` key.
2. In python, do the following to get the corresponding computational structure.

	```
	from pymatgen import MPRester
	mpr = MPRester(#Your API Key)
	structure = mpr.get_structure_by_material_id(#mp_id)
	```
A dump of all the material ids and structures for 2019.04.01 MP version is provided here: [https://ndownloader.figshare.com/files/15108200](https://ndownloader.figshare.com/files/15108200). Users can download the file and extract the `material_id` and `structure` from this file for all materials. The `structure` in this case is a `cif` file. Users can use again `pymatgen` to read the cif string and get the structure. 

```
from pymatgen.core import Structure
structure = Structure.from_str(#cif_string, fmt='cif')
```

For the ICSD structures, the users are required to have commercial ICSD access. Hence the structures will not be provided here.

<a name="train"></a>
#### Model training

A example training script is provided as `train.py` for the four-fidelity model (PBE/GLLB-SC/HSE/SCAN). The users can run `bash runall.sh` to download the data and run `train.py` automatically. 

The outcome of the fitting is an optimized model `best_model.hdf5`, with the configurations `best_model.hdf5.json`. The test errors are written in `test_errors.txt` and the fitting log will be saved in `log.txt`.

<a name="plot"></a>
#### Codes for plots
The codes for plots are shared in the `codes_for_plots` subfolder.


<a name="license"></a>
### License

This project is covered under the BSD 3-clause License.

<a name="issues"></a>
### Issues

1. For fitting models that include the experimental data, the model requires experimental structures. We take them from the ICSD commercial database. The users are required to have access to ICSD to repeat the fitting. The models however use the same principle as shown in the example script `train.py`.

2. On some systems, you may find errors like `SystemError: google/protobuf/pyext/...`. These seem to be bugs related to tensorflow. A workaround is by putting `import tensorflow as tf` at the beginning of your training script. 
