from megnet.data.graph import StructureGraph
import numpy as np
from megnet.data.graph import GaussianDistance
from monty.serialization import loadfn
from pathlib import Path


MODULE_DIR = Path(__file__).parent.absolute()


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
        super().__init__(nn_strategy=nn_strategy, atom_convertor=atom_convertor,
                         bond_convertor=bond_convertor, cutoff=cutoff)


def get_elemental_embeddings():
    """
    Provides the pre-trained elemental embeddings using formation energies,
    which can be used to speed up the training of other models. The embeddings
    are also extremely useful elemental descriptors that encode chemical
    similarity that may be used in other ways. See

    "Graph Networks as a Universal Machine Learning Framework for Molecules
    and Crystals", https://arxiv.org/abs/1812.05055

    :return: Dict of elemental embeddings as {symbol: length 16 string}
    """
    return loadfn(MODULE_DIR / "resources" /
                  "elemental_embedding_1MEGNet_layer.json")

