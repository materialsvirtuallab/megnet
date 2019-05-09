from operator import itemgetter
import numpy as np
import threading
from megnet.utils.general_utils import expand_1st
from monty.json import MSONable
from megnet.data import local_env
from inspect import signature
from pymatgen.analysis.local_env import NearNeighbors
from keras.utils import Sequence


class StructureGraph(MSONable):
    """
    This is a base class for converting converting structure into graphs or model inputs
    Methods to be implemented are follows:
        1. convert(self, structure)
            This is to convert a structure into a graph dictionary
        2. get_input(self, structure)
            This method convert a structure directly to a model input
        3. get_flat_data(self, graphs, targets)
            This method process graphs and targets pairs and output model input list.

    """
    def __init__(self,
                 nn_strategy,
                 atom_convertor=None,
                 bond_convertor=None,
                 **kwargs):

        if isinstance(nn_strategy, str):
            strategy = local_env.get(nn_strategy)
            parameters = signature(strategy).parameters
            param_dict = {i: j.default for i, j in parameters.items()}
            for i, j in kwargs.items():
                if i in param_dict:
                    setattr(self, i, j)
                    param_dict.update({i: j})
            self.nn_strategy = strategy(**param_dict)
        elif isinstance(nn_strategy, NearNeighbors):
            self.nn_strategy = nn_strategy
        else:
            raise RuntimeError("Strategy not valid")

        self.atom_convertor = atom_convertor
        self.bond_convertor = bond_convertor
        if self.atom_convertor is None:
            self.atom_convertor = self._get_dummy_convertor()
        if self.bond_convertor is None:
            self.bond_convertor = self._get_dummy_convertor()

    def convert(self, structure, state_attributes=None):
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance

        Args:
            structure: (pymatgen structure)
            state_attributes: (list) state attributes
        Returns:
            (dictionary)
        """
        state_attributes = state_attributes or [[0, 0]]
        index1 = []
        index2 = []
        bonds = []
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            for neighbor in neighbors:
                index2.append(neighbor['site_index'])
                bonds.append(neighbor['weight'])

        atoms = [i.specie.Z for i in structure]

        if np.size(np.unique(index1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")
        else:
            return {'atom': np.array(atoms, dtype='int32').tolist(),
                    'bond': bonds,
                    'state': state_attributes,
                    'index1': index1,
                    'index2': index2
                    }

    def __call__(self, structure, state_attributes=None):
        return self.convert(structure, state_attributes)

    def get_input(self, structure):
        """
        Turns a structure into model input
        """
        graph = self.convert(structure)
        return self.graph_to_input(graph)

    def graph_to_input(self, graph):
        """
        Turns a graph into model input
        """
        gnode = [0] * len(graph['atom'])
        gbond = [0] * len(graph['index1'])

        return [expand_1st(self.atom_convertor.convert(graph['atom'])),
                expand_1st(self.bond_convertor.convert(graph['bond'])),
                expand_1st(np.array(graph['state'])),
                expand_1st(np.array(graph['index1'])),
                expand_1st(np.array(graph['index2'])),
                expand_1st(np.array(gnode)),
                expand_1st(np.array(gbond)),
                ]

    def get_flat_data(self, graphs, targets):
        """
        Expand the graph dictionary to form a list of features and targets tensors
        This is useful when the model is trained on assembled graphs on the fly

        Args:
            graphs: (list of dictionary) list of graph dictionary for each structure
            targets: (list of float) correpsonding target values for each structure

        Returns:
            tuple(node_features, edges_features, global_values, index1, index2, targets)
        """
        atoms = []
        bonds = []
        states = []
        index1 = []
        index2 = []

        final_targets = []
        for g, t in zip(graphs, targets):
            if isinstance(g, dict):
                atoms.append(np.array(g['atom']))
                bonds.append(np.array(g['bond']))
                states.append(g['state'])
                index1.append(g['index1'])
                index2.append(g['index2'])
                final_targets.append([t])
        return atoms, bonds, states, index1, index2, final_targets

    def _get_dummy_convertor(self):
        return DummyConvertor()

    def as_dict(self):
        all_dict = super().as_dict()
        all_dict.pop('nn_strategy')
        return all_dict


class DistanceConvertor(MSONable):
    """
    Base class for distance conversion. The class needs to have a convert method.
    """
    def convert(self, d):
        raise NotImplementedError


class DummyConvertor(DistanceConvertor):
    """
    Dummy convertor as a placeholder
    """
    def convert(self, d):
        return d


class GaussianDistance(DistanceConvertor):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args:
        centers: (np.array)
        width: (float)
    """

    def __init__(self, centers=np.linspace(0, 5, 100), width=0.5):
        self.centers = centers
        self.width = width

    def convert(self, d):
        """
        expand distance vector d with given parameters

        Args:
            d: (1d array) distance array

        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-(d[:, None] - self.centers[None, :]) ** 2 / self.width ** 2)


class GraphBatchGenerator(Sequence):
    """
    A generator class that assembles several structures (indicated by
    batch_size) and form (x, y) pairs for model training

    Args:
        atom_features: (list of np.array) list of atom feature matrix,
        bond_features: (list of np.array) list of bond features matrix
        state_features: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
        index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for different structures
        index2_list: (list of integer) list of (M, ) the other side atomic
            index of the bond, M is different for different structures, but it has to be the same as the correponding index1.
        targets: (numpy array), N*1, where N is the number of structures
        batch_size: (int) number of samples in a batch
    """
    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets,
                 batch_size=128,
                 is_shuffle=True):
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.state_features = state_features
        self.index1_list = index1_list
        self.index2_list = index2_list
        self.targets = targets
        self.batch_size = batch_size
        self.total_n = len(atom_features)
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.mol_index = list(range(self.total_n))
        self.mol_index = np.random.permutation(self.mol_index)
        self.is_shuffle = is_shuffle

    def __len__(self):
        return self.max_step

    def __getitem__(self, index):
        batch_index = self.mol_index[index * self.batch_size:(index + 1) * self.batch_size]
        it = itemgetter(*batch_index)
        # get atom features from  a batch of structures
        feature_list_temp = itemgetter_list(self.atom_features, batch_index)
        # get atom's structure id
        gnode = []
        for i, j in enumerate(feature_list_temp):
            gnode += [i] * len(j)

        # get bond features from a batch of structures
        connection_list_temp = itemgetter_list(self.bond_features, batch_index)
        # get bond's structure id
        gbond = []
        for i, j in enumerate(connection_list_temp):
            gbond += [i] * len(j)

        global_list_temp = itemgetter_list(self.state_features, batch_index)
        # assemble atom features together
        feature_list_temp = np.concatenate(feature_list_temp, axis=0)
        feature_list_temp = self.process_atom_feature(feature_list_temp)
        # assemble bond feature together
        connection_list_temp = np.concatenate(connection_list_temp, axis=0)
        connection_list_temp = self.process_bond_feature(connection_list_temp)
        # assemble state feature together
        global_list_temp = np.concatenate(global_list_temp, axis=0)
        global_list_temp = self.process_state_feature(global_list_temp)

        # assemble bond indices
        index1_temp = it(self.index1_list)
        index2_temp = it(self.index2_list)
        index1 = []
        index2 = []
        offset_ind = 0
        for ind1, ind2 in zip(index1_temp, index2_temp):
            index1 += [i + offset_ind for i in ind1]
            index2 += [i + offset_ind for i in ind2]
            offset_ind += (max(ind1) + 1)
        # get targets
        target_temp = it(self.targets)
        target_temp = np.atleast_2d(target_temp)

        return [expand_1st(feature_list_temp),
                expand_1st(connection_list_temp),
                expand_1st(global_list_temp),
                expand_1st(index1),
                expand_1st(index2),
                expand_1st(gnode),
                expand_1st(gbond)], expand_1st(target_temp)

    def on_epoch_end(self):
        if self.is_shuffle:
            self.mol_index = np.random.permutation(self.mol_index)

    def process_atom_feature(self, x):
        return x

    def process_bond_feature(self, x):
        return x

    def process_state_feature(self, x):
        return x


class GraphBatchDistanceConvert(GraphBatchGenerator):
    """
    Generate batch of structures with bond distance being expanded using a Expansor

    Args:
        atom_features: (list of np.array) list of atom feature matrix,
        bond_features: (list of np.array) list of bond features matrix
        state_features: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
        index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for differentstructures
        index2_list: (list of integer) list of (M, ) the other side atomic index of the bond, M is different for different
            structures, but it has to be the same as the correponding index1.
        targets: (numpy array), N*1, where N is the number of structures
        batch_size: (int) number of samples in a batch
        is_shuffle: (bool) whether to shuffle the structure, default to True
        distance_convertor: (bool) convertor for processing the distances

    """
    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets,
                 batch_size=128,
                 is_shuffle=True,
                 distance_convertor=None):
        super().__init__(atom_features=atom_features,
                         bond_features=bond_features,
                         state_features=state_features,
                         index1_list=index1_list,
                         index2_list=index2_list,
                         targets=targets,
                         batch_size=batch_size,
                         is_shuffle=is_shuffle)
        self.distance_convertor = distance_convertor

    def process_bond_feature(self, x):
        return self.distance_convertor.convert(x)


def itemgetter_list(l, indices):
    """
    Get indices of l and return a tuple

    Args:
        l:  (list)
        indices: (list) indices

    Returns:
        (tuple)
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return it(l),
    else:
        return it(l)
