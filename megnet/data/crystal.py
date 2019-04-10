from megnet.data.graph import StructureGraph, GaussianDistance
import numpy as np


class CrystalGraph(StructureGraph):
    """
    Convert a crystal into a graph with z as atomic feature and distance as bond feature
    one can optionally include state features
    """
    def __init__(self,
                 nn_strategy='MinimumDistanceNNAll',
                 atom_convertor=None,
                 bond_convertor=None,
                 cutoff=4.0
                 ):
        if bond_convertor is None:
            bond_convertor = GaussianDistance(np.linspace(0, 5, 100), 0.5)
        super(CrystalGraph, self).__init__(nn_strategy=nn_strategy,
                                           atom_convertor=atom_convertor,
                                           bond_convertor=bond_convertor,
                                           cutoff=cutoff)




