#!/usr/bin/env python
# coding: utf-8


# Author: Chi Chen
# Email: chc273@eng.ucsd.edu

# 1. Set up
##  We use PBE + GLLB-SC + HSE + SCAN as training data
##  GLLB-SC + HSE + SCAN as validation data
##  PBE + GLLB-SC + HSE + SCAN as test data
##  We do not use PBE as validation data. It may bias the model to fit PBE very accurately, 
##  at the sacrifice of other fidelities, since the PBE data set is much larger than the rest.

## Experimental data are not included here since it requires access to commercial ICSD database
## for the structures. 
import tensorflow as tf
ALL_FIDELITIES = ['pbe', 'gllb-sc', 'hse', 'scan']
TRAIN_FIDELITIES= ['pbe', 'gllb-sc', 'hse', 'scan']
VAL_FIDELITIES = ['gllb-sc', 'hse', 'scan']
TEST_FIDELITIES= ['pbe', 'gllb-sc', 'hse', 'scan']

##  Set the number of maximum epochs to 1500
EPOCHS = 1500

##  Random seed
SEED = 42

##  Use the GPU with index 0. I use GTX 1080Ti, so one GPU is enough. 
##  By default, tensorflow will use all GPUs at maximum capability, this is not what we need.
GPU_INDEX = '0'


from copy import deepcopy
from glob import glob
import gzip
from itertools import chain
import json
import os
import pickle

import numpy as np
from pymatgen.core import Structure
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

## Import megnet related modules
from megnet.callbacks import ReduceLRUponNan, ManualStop
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GraphBatchDistanceConvert, GaussianDistance
from megnet.models import MEGNetModel

## Set GPU
os.environ['CUDA_VISIBLE_DEVICES']  = '0'



#  2. Model construction
##  Graph converter 
crystal_graph = CrystalGraph(
        bond_converter=GaussianDistance(centers=np.linspace(0, 6, 100),
                                        width=0.5),
        cutoff=5.0)
## model setup
model = MEGNetModel(nfeat_edge=100, nfeat_global=None, ngvocal=len(TRAIN_FIDELITIES), 
                    global_embedding_dim=16,  nblocks=3, nvocal=95, 
                    npass=2, graph_converter=crystal_graph, lr=1e-3)


#  3. Data loading and processing
##  load data

##  Structure data for all materials project materials

if not os.path.isfile('mp.2019.04.01.json'):
    raise RuntimeError("Please download the data first! Use runall.sh in this directory if needed.")

with open('mp.2019.04.01.json', 'r') as f:
    structure_data = {i['material_id']: i['structure'] for i in json.load(f)}
print('All structures in mp.2019.04.01.json contain %d structures' % len(structure_data))


##  Band gap data
with gzip.open('data_no_structs.json.gz', 'rb') as f:
    bandgap_data = json.loads(f.read())

useful_ids = set.union(*[set(bandgap_data[i].keys()) for i in ALL_FIDELITIES])  # mp ids that are used in training
print('Only %d structures are used' % len(useful_ids))
print('Calculating the graphs for all structures... this may take minutes.')
structure_data = {i: structure_data[i] for i in useful_ids}
structure_data = {i: crystal_graph.convert(Structure.from_str(j, fmt='cif')) 
                  for i, j in structure_data.items()}

##  Generate graphs with fidelity information
graphs = []
targets = []
material_ids = []

for fidelity_id, fidelity in enumerate(ALL_FIDELITIES):
    for mp_id in bandgap_data[fidelity]:
        graph = deepcopy(structure_data[mp_id])
        
        # The fidelity information is included here by changing the state attributes
        # PBE: 0, GLLB-SC: 1, HSE: 2, SCAN: 3
        graph['state'] = [fidelity_id]  
        graphs.append(graph)
        targets.append(bandgap_data[fidelity][mp_id])
        # the new id is of the form mp-id_fidelity, e.g., mp-1234_pbe
        material_ids.append('%s_%s' % (mp_id, fidelity))
        
final_graphs = {i:j for i, j in zip(material_ids, graphs)}
final_targets = {i:j for i, j in zip(material_ids, targets)}


#  4. Data splits

from sklearn.model_selection import train_test_split

##  train:val:test = 8:1:1
fidelity_list = [i.split('_')[1] for i in material_ids]
train_val_ids, test_ids = train_test_split(material_ids, stratify=fidelity_list, 
                                           test_size=0.1, random_state=SEED)
fidelity_list = [i.split('_')[1] for i in train_val_ids]
train_ids, val_ids = train_test_split(train_val_ids, stratify=fidelity_list, 
                                      test_size=0.1/0.9, random_state=SEED)

##  remove pbe from validation
val_ids = [i for i in val_ids if not i.endswith('pbe')]


print("Train, val and test data sizes are ", len(train_ids), len(val_ids), len(test_ids))


## Get the train, val and test graph-target pairs
def get_graphs_targets(ids):
    """
    Get graphs and targets list from the ids
    
    Args:
        ids (List): list of ids
    
    Returns:
        list of graphs and list of target values
    """
    ids = [i for i in ids if i in final_graphs]
    return [final_graphs[i] for i in ids], [final_targets[i] for i in ids]

train_graphs, train_targets = get_graphs_targets(train_ids)
val_graphs, val_targets = get_graphs_targets(val_ids)


#  5. Model training
callbacks = [ReduceLRUponNan(patience=500), ManualStop()]
model.train_from_graphs(train_graphs, train_targets, val_graphs, val_targets, 
                        epochs=EPOCHS, verbose=2, initial_epoch=0, callbacks=callbacks)

#  6. Model testing

##  load the best model with lowest validation error
files = glob('./callback/*.hdf5')
best_model = sorted(files, key=os.path.getctime)[-1]

model.load_weights(best_model)
model.save_model('best_model.hdf5')


def evaluate(test_graphs, test_targets):
    """
    Evaluate the test errors using test_graphs and test_targets
    
    Args:
        test_graphs (list): list of graphs
        test_targets (list): list of target properties
        
    Returns:
        mean absolute errors
    """
    test_data = model.graph_converter.get_flat_data(test_graphs, test_targets)
    gen = GraphBatchDistanceConvert(*test_data, distance_converter=model.graph_converter.bond_converter,
                                    batch_size=128)
    preds = []
    trues = []
    for i in range(len(gen)):
        d = gen[i]
        preds.extend(model.predict(d[0]).ravel().tolist())
        trues.extend(d[1].ravel().tolist())
    return np.mean(np.abs(np.array(preds) - np.array(trues)))


## Calculate the errors on each fidelity
test_errors = []
for fidelity in TEST_FIDELITIES:
    test_ids_fidelity = [i for i in test_ids if i.endswith(fidelity)]
    test_graphs, test_targets = get_graphs_targets(test_ids_fidelity)
    test_error = evaluate(test_graphs, test_targets)
    test_errors.append(test_error)

## Save errors
with open('test_errors.txt', 'w') as f:
    line = ['%s: %.3f eV\n' % (i, j) for i, j in zip(TEST_FIDELITIES, test_errors)]
    f.write(''.join(line))

